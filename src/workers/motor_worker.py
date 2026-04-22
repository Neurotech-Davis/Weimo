"""
workers/motor_worker.py
Windows PC → pyserial (USB) → Pico (PicoFolder/main.py) → Motor HAT → motors

Protocol (must match PicoFolder/main.py):
  MECANUM:fwd:strafe:turn\n  → set motor speeds, Pico responds OK:MOVING
  STOP\n                     → stop all motors, Pico responds OK:STOPPED
"""

import serial
import time


PICO_PORT = "/dev/ttyACM0"  # Windows: check Device Manager
BAUD_RATE = 115200
SPEED = 1500  # PWM value 0-4095, tune for your buggy

# Tuning constants — adjust these against real hardware
ANGLE_THRESHOLD_DEG = 10.0  # within this, consider aligned
DIST_THRESHOLD_MM = 150.0  # within this, consider arrived
DEG_PER_SEC = 90.0  # how fast buggy rotates — measure this
MM_PER_SEC = 300.0  # how fast buggy drives forward — measure this


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
    """Send a command string, return Pico's ack line."""
    ser.write((cmd + "\n").encode())
    try:
        ack = ser.readline().decode().strip()
        return ack
    except Exception:
        return ""


def cmd_stop(ser):
    return send(ser, "STOP")


def cmd_mecanum(ser, fwd: int, strafe: int, turn: int):
    return send(ser, f"MECANUM:{fwd}:{strafe}:{turn}")


# ── Navigation primitives ─────────────────────────────────────────────────────


def rotate_to_angle(ser, h_angle: float):
    """Rotate in place until aligned with target angle."""
    if abs(h_angle) <= ANGLE_THRESHOLD_DEG:
        return

    duration = abs(h_angle) / DEG_PER_SEC
    turn_speed = SPEED if h_angle > 0 else -SPEED

    print(f"[motor] rotating {h_angle:.1f}° — {duration:.2f}s")
    # cmd_mecanum(ser, 0, 0, turn_speed)
    send(ser, f"TURN:{h_angle}")
    time.sleep(duration)
    cmd_stop(ser)
    time.sleep(0.1)  # brief settle


def drive_distance(ser, dist_mm: float):
    """Drive forward for estimated time to cover dist_mm."""
    if dist_mm <= DIST_THRESHOLD_MM:
        print("[motor] already at target distance")
        return

    duration = dist_mm / MM_PER_SEC
    print(f"[motor] driving {dist_mm:.0f}mm — {duration:.2f}s")
    # cmd_mecanum(ser, SPEED, 0, 0)
    send(ser, f"DRIVE:{dist_mm}")
    time.sleep(duration)
    cmd_stop(ser)
    time.sleep(0.1)


def navigate_to(ser, h_angle: float, dist_mm: float):
    """Rotate to align, then drive to target. Sequential, blocking."""
    print(f"[motor] navigating → angle={h_angle:.1f}° dist={dist_mm:.0f}mm")
    rotate_to_angle(ser, h_angle)
    drive_distance(ser, dist_mm)
    print("[motor] arrived at target")


# ── Worker (called by main_process.py) ───────────────────────────────────────


def motor_worker(shared_state):
    ser = None
    try:
        ser = connect(PICO_PORT, BAUD_RATE)
        shared_state.motor_running.value = True
        last_cmd = -1
        navigating = False

        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            # manual UI command takes priority — cancels any active navigation
            manual_cmd = shared_state.motor_command.value
            if manual_cmd != 0:
                navigating = False
                if manual_cmd != last_cmd:
                    fwd, strafe, turn = _cmd_to_vectors(manual_cmd)
                    cmd_mecanum(ser, fwd, strafe, turn)
                    shared_state.motor_state.value = manual_cmd
                    last_cmd = manual_cmd

            elif not navigating and shared_state.prediction.value == 1:  # 1 = move
                # latch current target and navigate
                h_angle = shared_state.target_angle.value
                dist = shared_state.target_dist.value
                navigating = True
                navigate_to(ser, h_angle, dist)
                navigating = False
                shared_state.motor_command.value = 0

            else:
                if last_cmd != 0:
                    cmd_stop(ser)
                    shared_state.motor_state.value = 0
                    last_cmd = 0

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, 0.05 - elapsed))

    except Exception as e:
        print(f"[motor] fatal error: {e}")
    finally:
        shared_state.motor_running.value = False
        if ser and ser.is_open:
            cmd_stop(ser)
            time.sleep(0.1)
            ser.close()
        print("[motor] shutdown complete")


def _cmd_to_vectors(cmd_id: int):
    """Map UI command ID to (fwd, strafe, turn) vectors."""
    return {
        0: (0, 0, 0),  # stop
        1: (SPEED, 0, 0),  # forward
        2: (-SPEED, 0, 0),  # backward
        3: (0, 0, -SPEED),  # rotate left
        4: (0, 0, SPEED),  # rotate right
        5: (0, -SPEED, 0),  # strafe left
        6: (0, SPEED, 0),  # strafe right
    }.get(cmd_id, (0, 0, 0))


# ── Manual test script ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Motor Worker Manual Test ===")
    print(f"Connecting to {PICO_PORT} at {BAUD_RATE} baud...")

    try:
        ser = connect(PICO_PORT, BAUD_RATE)
    except RuntimeError as e:
        print(e)
        exit(1)

    print("Connected. Commands:")
    print("  navigate <angle_deg> <dist_mm>  — e.g. 'navigate 30 500'")
    print("  forward  <dist_mm>              — e.g. 'forward 300'")
    print("  rotate   <angle_deg>            — e.g. 'rotate -45'")
    print("  stop")
    print("  quit")
    print()

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
                ack = cmd_stop(ser)
                print(f"  ack: {ack}")

            elif cmd == "forward" and len(parts) == 2:
                dist = float(parts[1])
                drive_distance(ser, dist)

            elif cmd == "rotate" and len(parts) == 2:
                angle = float(parts[1])
                rotate_to_angle(ser, angle)

            elif cmd == "navigate" and len(parts) == 3:
                angle = float(parts[1])
                dist = float(parts[2])
                navigate_to(ser, angle, dist)

            else:
                print("  unknown command — try 'navigate 30 500'")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        cmd_stop(ser)
        ser.close()
        print("Serial closed.")
