import serial
import time

PORT = "COM3"
BAUD = 115200

# RPLIDAR command: GET_HEALTH = 0xA5 0x52
GET_HEALTH = bytes([0xA5, 0x52])

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(GET_HEALTH)
    ser.flush()

    desc = ser.read(7)     # response descriptor (binary)
    payload = ser.read(3)  # health payload is typically 3 bytes

    print("DESC:", desc)
    print("PAYLOAD:", payload)
