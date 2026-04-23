"""
test_pico.py — validate Pico serial connection
run from src/: python workers/test_pico.py
"""

import serial
import time

PICO_PORT = "/dev/ttyACM0"  # Windows: COM3
BAUD_RATE = 115200

print(f"Connecting to {PICO_PORT}...")
try:
    ser = serial.Serial(PICO_PORT, BAUD_RATE, timeout=5)
    print("Connected.\n")
except serial.SerialException as e:
    print(f"Failed: {e}")
    exit(1)


def send_and_watch(cmd: str, timeout: float = 15.0):
    print(f">>> {cmd}")
    ser.write((cmd + "\n").encode())
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode().strip()
        if line:
            print(f"    {line}")
        if line in ("OK:DONE", "OK:STOPPED"):
            break


try:
    print("--- Test 1: STOP ---")
    send_and_watch("STOP", timeout=3)

    print("\n--- Test 2: Short drive (0.1m) ---")
    send_and_watch("DRIVE:0.1")

    print("\n--- Test 3: 45 degree turn ---")
    send_and_watch("TURN:45")

    print("\n--- Test 4: Turn back ---")
    send_and_watch("TURN:-45")

    print("\nAll tests complete.")

except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    ser.write(b"STOP\n")
    ser.close()
    print("Serial closed.")
