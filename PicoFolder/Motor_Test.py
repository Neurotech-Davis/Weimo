from machine import Pin, I2C
import time

# --- I2C & HAT SETUP ---
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)
PCA9685_ADDR = 0x60

# Wake up HAT & Turn on Auto-Increment
i2c.writeto_mem(PCA9685_ADDR, 0x00, b'\x20') 
time.sleep(0.05)
i2c.writeto_mem(PCA9685_ADDR, 0x01, b'\x04') 

def set_pwm(channel, on, off):
    data = bytearray([on & 0xFF, on >> 8, off & 0xFF, off >> 8])
    i2c.writeto_mem(PCA9685_ADDR, 0x06 + 4 * channel, data)

def set_motor(motor_num, speed):
    pins = {1: (8,10,9), 2: (13,11,12), 3: (2,4,3), 4: (7,5,6)}
    pwm_pin, in1_pin, in2_pin = pins[motor_num]
    
    speed = max(-4095, min(4095, int(speed))) 
    
    if speed == 0:
        set_pwm(in1_pin, 0, 4095)
        set_pwm(in2_pin, 0, 4095)
        set_pwm(pwm_pin, 0, 0)
    elif speed > 0:
        set_pwm(in1_pin, 4095, 0)
        set_pwm(in2_pin, 0, 4095)
        set_pwm(pwm_pin, 0, speed)
    else:
        set_pwm(in1_pin, 0, 4095)
        set_pwm(in2_pin, 4095, 0)
        set_pwm(pwm_pin, 0, abs(speed))

# --- THE DIAGNOSTIC SEQUENCE ---
motor_names = {
    1: "Front Right (M1)",
    2: "Back Right (M2)",
    3: "Back Left (M3)",
    4: "Front Left (M4)"
}

test_speed = 2000 # Roughly 50% power

print("--- STARTING BENCH TEST ---")
print("Make sure the robot is propped up!")
time.sleep(3)

for m_num in range(1, 5):
    print(f"\nTesting {motor_names[m_num]}...")
    
    print("  -> Spinning Code FORWARD...")
    set_motor(m_num, test_speed)
    time.sleep(3)
    
    print("  -> Spinning Code REVERSE...")
    set_motor(m_num, -test_speed)
    time.sleep(3)
    
    print("  -> STOP")
    set_motor(m_num, 0)
    time.sleep(2)

print("\n--- BENCH TEST COMPLETE ---")