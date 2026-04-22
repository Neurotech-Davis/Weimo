from machine import I2C, Pin
i2c = I2C(0, sda=Pin(16), scl=Pin(17))
print("Scanning I2C bus...")
devices = i2c.scan()
print("Found devices at:", [hex(d) for d in devices])