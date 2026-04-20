"""
This is the motor translation worker, it takes inputs and speaks to the pico
"""

import serial
import time

COMMANDS = {
    0: "z",  # stop
    1: "w",  # forward
    2: "s",  # backward
    3: "a",  # rotate left
    4: "d",  # rotate right
    5: "q",  # strafe left
    6: "e",  # strafe right
}

# For actual demo, this will have to be windows COM port (ex: COM3)
PICO_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200


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


def motor_worker(shared_state):
    # SETUP
    ser = None
    try:
        ser = connect(PICO_PORT, BAUD_RATE)
        shared_state.motor_running.value = True
        last_cmd = -1

        # LOOP
        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            cmd_id = shared_state.motor_command.value
            if cmd_id != last_cmd:
                char = COMMANDS.get(cmd_id, "z")
                ser.write(char.encode())
                shared_state.motor_state.value = cmd_id
                last_cmd = cmd_id

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, 0.05 - elapsed))

    # TEARDOW
    except Exception as e:
        print(f"[motor] fatal error: {e}")

    finally:
        # TEARDOWN
        shared_state.motor_running.value = False
        if ser and ser.is_open:
            ser.write(b"z")  # safety stop
            time.sleep(0.1)
            ser.close()
