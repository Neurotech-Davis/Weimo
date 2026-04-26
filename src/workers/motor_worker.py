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
import math
from enum import Enum


BAUD_RATE = 115200
INFINITE_DRIVE_FALLBACK_LENGTH = 1000  # mm

MOVE_CONFIDENCE_THRESHOLD = 0.95
DIST_THRESHOLD_MM = 150
ANGLE_THRESHOLD_DEG = 10.0

LIDAR_STOP = True
YOLO_STOP = True


class MotorState(Enum):
    IDLE = "idle"
    DRIVING = "driving"


# ── Serial helpers ────────────────────────────────────────────────────────────


def force_pico_reset(ser):
    """Sends Ctrl+C to break hangs and Ctrl+D to soft-reboot the Pico."""
    print("[motor_worker] Initializing Pico state...")
    ser.write(b"\x03")  # Ctrl+C (Interrupt any running code/REPL)
    time.sleep(0.1)
    ser.write(b"\x03")  # Second Ctrl+C just in case
    time.sleep(0.1)
    ser.write(b"\x04")  # Ctrl+D (Soft Reboot)
    time.sleep(1.5)  # Wait for MicroPython to reboot and run main.py
    ser.reset_input_buffer()
    print("[motor_worker] Pico ready.")


def connect(port: str, baud: int, retries: int = 5) -> serial.Serial:
    for attempt in range(1, retries + 1):
        try:
            ser = serial.Serial(port, baud, timeout=2)
            print(
                f"[motor_worker] port open on attempt {attempt}, waiting for Pico boot..."
            )
            # Drain until we see the ready banner or 4s elapses
            deadline = time.time() + 4.0
            while time.time() < deadline:
                line = ser.readline().decode(errors="replace").strip()
                if "Motor Hat Initialized" in line:
                    print("[motor_worker] Pico boot confirmed — ready")
                    break
            else:
                print("[motor_worker] boot banner not seen, proceeding anyway")
            ser.reset_input_buffer()
            return ser
        except serial.SerialException as e:
            print(f"[motor_worker] attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"[motor_worker] could not open {port} after {retries} attempts")


# def send(ser, cmd, shared_state=None):
#     try:
#         ser.write((cmd + "\n").encode())
#     except (serial.SerialException, OSError) as e:
#         print(f"[motor_worker] Write failure: {e}. Attempting recovery...")
#         force_pico_reset(ser)
#         return "OK:STOPPED"  # Fail safe
#
#     deadline = time.time() + 30.0
#     while time.time() < deadline:
#         if shared_state is not None:
#             if shared_state.motor_command.value != 0 or jaw_clench_detected(
#                 shared_state
#             ):
#                 ser.write(b"STOP\n")
#                 ser.flush()
#                 # ... rest of your drain logic ...
#                 return "OK:STOPPED"
#
#         try:
#             line = ser.readline().decode().strip()
#         except (serial.SerialException, OSError):
#             print("[motor_worker] Read failure during command. Resetting...")
#             force_pico_reset(ser)
#             return "OK:STOPPED"
#
#         if line in ("OK:DONE", "OK:STOPPED"):
#             return line
#         elif line:
#             print(f"[pico] {line}")
#
#     return ""
def send(ser, cmd, shared_state=None):
    ser.write((cmd + "\n").encode())
    is_drive_cmd = cmd.startswith("DRIVE")
    deadline = time.time() + 30.0

    while time.time() < deadline:
        if shared_state is not None:
            obstacle_blocking = is_drive_cmd and obstacle_detected(shared_state, LIDAR_STOP, YOLO_STOP)
            if (
                shared_state.motor_command.value != 0
                or jaw_clench_detected(shared_state)
                or obstacle_blocking
            ):
                ser.write(b"STOP\n")
                ser.flush()
                drain_deadline = time.time() + 2.0
                while time.time() < drain_deadline:
                    line = ser.readline().decode().strip()
                    if line in ("OK:DONE", "OK:STOPPED"):
                        return "OK:STOPPED"
                return "OK:STOPPED"

        line = ser.readline().decode().strip()
        if "Motor Hat Initialized" in line:
            print("[motor_worker] !! PICO REBOOT DETECTED mid-session !!")
            # Pico rebooted under us — treat as a serial error
            raise serial.SerialException("Pico rebooted mid-command")
        if line in ("OK:DONE", "OK:STOPPED"):
            return line
        elif line:
            print(f"[pico] {line}")

    # Deadline expired — don't silently return, force reconnect
    raise serial.SerialTimeoutException(
        f"send() 30s timeout waiting for ack on: {cmd!r}"
    )


def cmd_stop(ser):
    return send(ser, "STOP")


def cmd_turn(ser, angle_deg: float, shared_state=None) -> str:
    return send(ser, f"TURN:{angle_deg:.1f}", shared_state)


def cmd_drive(ser, dist_m: float, shared_state=None) -> str:
    return send(ser, f"DRIVE:{dist_m:.3f}", shared_state)


def cmd_stop_immediate(ser):
    """Fire-and-forget stop — doesn't wait for ack."""
    ser.write(b"STOP\n")
    ser.flush()


# ── Jaw clench check — call between blocking operations ───────────────────────


def obstacle_detected(shared_state, lidar_stop=True, yolo_stop=True):
    emergency_stop = False
    if lidar_stop:
        emergency_stop = emergency_stop or shared_state.lidar_obstacle_detected.value
    if yolo_stop:
        emergency_stop = emergency_stop or shared_state.yolo_obstacle_detected.value
    return emergency_stop


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
        cmd_turn(ser, h_angle, shared_state)  # blocks until Pico acks OK:DONE

    # step 3 — drive forward
    if not math.isfinite(dist_mm):
        print(f"[motor_worker] dist={dist_mm:.0f}mm is not finite — skipping drive")
        dist_mm = INFINITE_DRIVE_FALLBACK_LENGTH  # Currently 700. 1500 / 1000 ?

    if dist_mm > DIST_THRESHOLD_MM:
        print(f"[motor_worker] driving {dist_mm:.0f}mm")
        dist_m = dist_mm / 1000.0
        cmd_drive(ser, dist_m, shared_state)

    print("[motor_worker] arrived at target")
    return True


def process_ui_turn_logic(ser, shared_state) -> bool:
    """
    Checks for a pending turn command from the UI.
    Returns True if a turn was executed, False otherwise.
    """
    turn_deg = shared_state.turn_command.value
    if turn_deg != 0.0:
        # Consume immediately to prevent double-triggering
        shared_state.turn_command.value = 0.0

        print(f"[motor_worker] Executing UI Zone Turn: {turn_deg}°")
        shared_state.motor_state.value = 1  # Update UI label to "DRIVING"

        # Execute the blocking serial command
        cmd_turn(ser, turn_deg, shared_state)
        shared_state.motor_state.value = 0  # Return to "IDLE"

        return True
    return False


# ── Worker ────────────────────────────────────────────────────────────────────


def motor_worker(shared_state):
    PICO_PORT = "/dev/ttyACM0" if shared_state.on_linux else "COM8"
    ser = None

    try:
        # ── outer loop: reconnection ──
        while not shared_state.shutdown.is_set():
            try:
                if ser is None or not ser.is_open:
                    print("[motor_worker] (re)connecting to Pico...")
                    ser = connect(PICO_PORT, BAUD_RATE)
                    shared_state.motor_running.value = True

                state = MotorState.IDLE
                print("[motor_worker] state=IDLE, ready")

                # ── inner loop: normal operation ──
                while not shared_state.shutdown.is_set():
                    t_start = time.perf_counter()

                    # ── manual UI override — always highest priority ──
                    manual_cmd = shared_state.motor_command.value
                    if manual_cmd != 0:
                        if state == MotorState.DRIVING:
                            print(
                                "[motor_worker] manual override — aborting navigation"
                            )
                        cmd_stop(ser)
                        state = MotorState.IDLE
                        shared_state.motor_state.value = 0
                        shared_state.motor_command.value = 0
                        time.sleep(0.05)
                        continue

                    if not shared_state.tracker_running:
                        time.sleep(2)
                        continue

                    # ── jaw clench emergency stop ──
                    if jaw_clench_detected(shared_state):
                        if state == MotorState.DRIVING:
                            print("[motor_worker] jaw_clench — emergency stop")
                            cmd_stop(ser)
                            state = MotorState.IDLE
                            shared_state.motor_state.value = 0
                            shared_state.prediction.value = 0
                        time.sleep(0.05)
                        continue

                    # ── state machine ──
                    if state == MotorState.IDLE:
                        if process_ui_turn_logic(ser, shared_state):
                            continue

                        pred = shared_state.prediction.value
                        conf = shared_state.pred_confidence.value
                        dist = shared_state.target_dist.value

                        if pred == 1 and conf >= MOVE_CONFIDENCE_THRESHOLD:
                            if dist > DIST_THRESHOLD_MM:
                                shared_state.prediction.value = 0
                                h_angle = shared_state.target_angle.value
                                frozen_dist = shared_state.target_dist.value
                                shared_state.committed_angle.value = h_angle
                                shared_state.committed_dist.value = frozen_dist

                                state = MotorState.DRIVING
                                shared_state.motor_state.value = 1
                                print("[motor_worker] IDLE → DRIVING")
                                arrived = navigate_to(
                                    ser, h_angle, frozen_dist, shared_state
                                )

                                state = MotorState.IDLE
                                shared_state.motor_state.value = 0
                                print(
                                    f"[motor_worker] DRIVING → IDLE  (arrived={arrived})"
                                )
                            else:
                                print(
                                    "[motor_worker] target too close, ignoring. Handing off to UI?"
                                )

                    elapsed = time.perf_counter() - t_start
                    time.sleep(max(0.0, 0.05 - elapsed))

            except (
                serial.SerialException,
                serial.SerialTimeoutException,
                OSError,
            ) as e:
                print(f"[motor_worker] serial error: {e} — attempting reconnect in 2s")
                shared_state.motor_state.value = 0
                shared_state.turn_command.value = 0.0
                shared_state.motor_command.value = 0
                if ser and ser.is_open:
                    try:
                        ser.close()
                    except Exception:
                        pass
                ser = None
                time.sleep(2.0)

    except Exception as e:
        print(f"[motor_worker] fatal error: {e}")
        shared_state.motor_error.set()

    finally:
        shared_state.motor_running.value = False
        if ser and ser.is_open:
            try:
                cmd_stop(ser)
                time.sleep(0.1)
            except Exception:
                pass
            ser.close()
        print("[motor_worker] shutdown complete")


# ── Manual test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PICO_PORT = "/dev/ttyACM0"

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
