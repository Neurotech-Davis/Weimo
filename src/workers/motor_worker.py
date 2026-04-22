"""
workers/motor_worker.py
Windows PC → pyserial (USB) → Pico → Motor HAT → motors

Pico protocol:
  TURN:<angle_deg>\n   → rotate to angle, Pico responds OK:DONE
  DRIVE:<dist_mm>\n    → drive forward dist_mm, Pico responds OK:DONE
  STOP\n               → immediate stop, Pico responds OK:STOPPED

State machine:
  IDLE    → on MOVE prediction + valid target → DRIVING
  DRIVING → on arrival → IDLE
  DRIVING → on jaw_clench at any point → IDLE (emergency stop)
  DRIVING → on manual UI command → IDLE (manual override)
"""

import serial
import time
from enum import Enum


PICO_PORT = "/dev/ttyACM0"  # Windows: COM3 etc.
BAUD_RATE = 115200

MOVE_CONFIDENCE_THRESHOLD = 0.95
DIST_THRESHOLD_M = 0.150
ANGLE_THRESHOLD_DEG = 10.0


class MotorState(Enum):
    IDLE = "idle"
    DRIVING = "driving"


# ── Serial helpers ────────────────────────────────────────────────────────────


def connect(port: str, baud: int, retries: int = 5) -> serial.Serial:
    for attempt in range(1, retries + 1):
        try:
            ser = serial.Serial(port, baud, timeout=1)
            print(f"[motor_worker] connected to {port} on attempt {attempt}")
            return ser
        except serial.SerialException as e:
            print(f"[motor_worker] attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"[motor_worker] could not open {port} after {retries} attempts")


def send(ser: serial.Serial, cmd: str) -> str:
    """Send command, block until Pico acks."""
    ser.write((cmd + "\n").encode())
    try:
        ack = ser.readline().decode().strip()
        print(f"[motor_worker] sent={cmd!r}  ack={ack!r}")
        return ack
    except Exception as e:
        print(f"[motor_worker] ack read failed: {e}")
        return ""


def cmd_stop(ser):
    return send(ser, "STOP")


def cmd_turn(ser, angle_deg: float) -> str:
    return send(ser, f"TURN:{angle_deg:.1f}")


def cmd_drive(ser, dist_mm: float) -> str:
    return send(ser, f"DRIVE:{dist_mm:.0f}")


# ── Jaw clench check — call between blocking operations ───────────────────────


def jaw_clench_detected(shared_state) -> bool:
    return shared_state.prediction.value == 2


# ── Navigation primitives ─────────────────────────────────────────────────────


def navigate_to(ser, h_angle: float, dist_mm: float, shared_state) -> bool:
    """
    Rotate then drive to target.
    Checks for jaw_clench between each step.
    Returns True if arrived, False if aborted by jaw_clench.
    """
    print(f"[motor_worker] navigating → angle={h_angle:.1f}° dist={dist_mm:.0f}mm")

    # step 1 — rotate to align
    if abs(h_angle) > ANGLE_THRESHOLD_DEG:
        if jaw_clench_detected(shared_state):
            print("[motor_worker] jaw_clench — aborting before rotate")
            cmd_stop(ser)
            return False
        print(f"[motor_worker] rotating {h_angle:.1f}°")
        cmd_turn(ser, h_angle)  # blocks until Pico acks OK:DONE

    # step 2 — check again before driving
    if jaw_clench_detected(shared_state):
        print("[motor_worker] jaw_clench — aborting before drive")
        cmd_stop(ser)
        return False

    # step 3 — drive forward
    if dist_mm > DIST_THRESHOLD_M:
        print(f"[motor_worker] driving {dist_mm:.0f}mm")
        cmd_drive(ser, dist_mm)  # blocks until Pico acks OK:DONE

    print("[motor_worker] arrived at target")
    return True


# ── Worker ────────────────────────────────────────────────────────────────────


def motor_worker(shared_state):
    ser = None
    try:
        ser = connect(PICO_PORT, BAUD_RATE)
        shared_state.motor_running.value = True

        state = MotorState.IDLE
        print(f"[motor_worker] state=IDLE, ready")

        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            # ── manual UI override — always highest priority ──
            manual_cmd = shared_state.motor_command.value
            if manual_cmd != 0:
                if state == MotorState.DRIVING:
                    print("[motor_worker] manual override — aborting navigation")
                cmd_stop(ser)
                state = MotorState.IDLE
                shared_state.motor_state.value = 0
                shared_state.motor_command.value = 0  # consume
                time.sleep(0.05)
                continue

            # ── jaw clench emergency stop ──
            if jaw_clench_detected(shared_state):
                if state == MotorState.DRIVING:
                    print("[motor_worker] jaw_clench — emergency stop")
                    cmd_stop(ser)
                    state = MotorState.IDLE
                    shared_state.motor_state.value = 0
                    shared_state.prediction.value = 0  # consume
                time.sleep(0.05)
                continue

            # ── state machine ──
            if state == MotorState.IDLE:
                pred = shared_state.prediction.value
                conf = shared_state.pred_confidence.value

                if pred == 1 and conf >= MOVE_CONFIDENCE_THRESHOLD:
                    # latch target at moment of move prediction
                    h_angle = shared_state.target_angle.value
                    dist = shared_state.target_dist.value
                    shared_state.prediction.value = 0  # consume

                    if dist > DIST_THRESHOLD_M:
                        state = MotorState.DRIVING
                        shared_state.motor_state.value = 1
                        print(f"[motor_worker] IDLE → DRIVING")

                        arrived = navigate_to(ser, h_angle, dist, shared_state)

                        state = MotorState.IDLE
                        shared_state.motor_state.value = 0
                        print(f"[motor_worker] DRIVING → IDLE  (arrived={arrived})")
                    else:
                        print("[motor_worker] target too close, ignoring")

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, 0.05 - elapsed))

    except Exception as e:
        print(f"[motor_worker] fatal error: {e}")
        shared_state.motor_error.set()

    finally:
        shared_state.motor_running.value = False
        if ser and ser.is_open:
            cmd_stop(ser)
            time.sleep(0.1)
            ser.close()
        print("[motor_worker] shutdown complete")


# ── Manual test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Motor Worker Manual Test ===")
    print(f"Connecting to {PICO_PORT} at {BAUD_RATE} baud...")

    try:
        ser = connect(PICO_PORT, BAUD_RATE)
    except RuntimeError as e:
        print(e)
        exit(1)

    print(
        "Commands: navigate <angle> <dist> | turn <angle> | drive <dist> | stop | quit"
    )

    try:
        while True:
            try:
                raw = input("> ").strip().lower()
            except EOFError:
                break
            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0]

            if cmd == "quit":
                break
            elif cmd == "stop":
                print(cmd_stop(ser))
            elif cmd == "turn" and len(parts) == 2:
                print(cmd_turn(ser, float(parts[1])))
            elif cmd == "drive" and len(parts) == 2:
                print(cmd_drive(ser, float(parts[1])))
            elif cmd == "navigate" and len(parts) == 3:

                class _FakeState:
                    class prediction:
                        value = 0

                navigate_to(ser, float(parts[1]), float(parts[2]), _FakeState())
            else:
                print("unknown command")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cmd_stop(ser)
        ser.close()
        print("Serial closed.")
