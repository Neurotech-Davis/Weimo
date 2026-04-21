from machine import Pin, I2C
import time

# Using your exact new I2C pins
print("Initializing I2C Bus...")
i2c = I2C(0, sda=Pin(16), scl=Pin(17), freq=100000)

print("Scanning for devices...")
devices = i2c.scan()

if len(devices) == 0:
    print("Result: No I2C devices found! (Check wiring, 3.3V, GND, or swapped SDA/SCL)")
else:
    print(f"Result: {len(devices)} device(s) found!")
    for device in devices:
        print(f"Device Decimal Address: {device} | Hex Address: {hex(device)}")