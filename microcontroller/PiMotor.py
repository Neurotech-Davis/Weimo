import time
from adafruit_motorkit import MotorKit

# Connect to the Adafruit Motor HAT
try:
    kit = MotorKit()
    print("Motor HAT successfully connected!")
except Exception as e:
    print(f"Error finding Motor HAT: {e}")
    print("Check your I2C settings and ensure the HAT is seated properly.")
    exit()

print("--- Starting Waymo Car Test ---")

# The throttle ranges from -1.0 (full reverse) to 1.0 (full forward)
# 0.0 is stop.

print("Moving Forward...")
# Setting both motors to 50% power
kit.motor1.throttle = 0.4  
kit.motor2.throttle = 0.4  

# Let them run for 1 second
time.sleep(2.0)  

print("Stopping...")
kit.motor1.throttle = 0.0
kit.motor2.throttle = 0.0

# Pause before ending the script
time.sleep(1.0) 

print("Test Complete. Check if the wheels moved!")