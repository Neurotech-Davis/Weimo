from machine import Pin, I2C
import time
import math
import sys
import select

# --- HARDWARE CONSTANTS ---
WHEEL_DIAMETER_M = 0.066  
WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER_M
TICKS_PER_REV = 247  
METERS_PER_TICK = WHEEL_CIRCUMFERENCE / TICKS_PER_REV
DISTANCE_MULTIPLIER = 0.80 
TRACK_WIDTH_M = 0.17 

# --- I2C & MOTOR HAT SETUP ---
i2c = I2C(0, sda=Pin(16), scl=Pin(17), freq=100000)
PCA9685_ADDR = 0x60

def init_motor_hat():
    try:
        i2c.writeto_mem(PCA9685_ADDR, 0x00, b'\x20') 
        time.sleep(0.05)
        i2c.writeto_mem(PCA9685_ADDR, 0x01, b'\x04') 
        print("STATUS: Motor Hat Initialized")
    except OSError:
        print("ERROR: Motor Hat communication failed")

def set_pwm(channel, on, off):
    data = bytearray([on & 0xFF, on >> 8, off & 0xFF, off >> 8])
    try:
        i2c.writeto_mem(PCA9685_ADDR, 0x06 + 4 * channel, data)
    except OSError:
        pass

def set_pin(channel, state):
    if state == 1: set_pwm(channel, 4096, 0) 
    else: set_pwm(channel, 0, 4096) 

def set_motor(motor_num, speed):
    # Mapping based on your "Perfect Mapping"
    pins = {1: (8,10,9), 2: (13,11,12), 3: (2,4,3), 4: (7,5,6)}
    pwm_pin, in1_pin, in2_pin = pins[motor_num]
    speed = max(-4095, min(4095, int(speed))) 
    
    if speed == 0:
        set_pin(in1_pin, 0); set_pin(in2_pin, 0); set_pwm(pwm_pin, 0, 0)
    elif speed > 0:
        set_pin(in1_pin, 0); set_pin(in2_pin, 1); set_pwm(pwm_pin, 0, speed)
    else:
        set_pin(in1_pin, 1); set_pin(in2_pin, 0); set_pwm(pwm_pin, 0, abs(speed))

# --- ENCODER SETUP (UPDATED PINS) ---
ticks = {1: 0, 2: 0, 3: 0, 4: 0}

def make_handler(motor_id, pin_b, reverse_polarity=False):
    def handler(pin):
        val = pin_b.value()
        if reverse_polarity: val = not val
        if val: ticks[motor_id] += 1
        else: ticks[motor_id] -= 1
    return handler

# M1: EncA=2, EncB=3, Rev=False
m1_a = Pin(2, Pin.IN, Pin.PULL_UP); m1_b = Pin(3, Pin.IN, Pin.PULL_UP)
m1_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(1, m1_b, False))

# M2: EncA=19, EncB=18, Rev=False
m2_a = Pin(19, Pin.IN, Pin.PULL_UP); m2_b = Pin(18, Pin.IN, Pin.PULL_UP)
m2_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(2, m2_b, False))

# M3: EncA=15, EncB=14, Rev=True
m3_a = Pin(15, Pin.IN, Pin.PULL_UP); m3_b = Pin(14, Pin.IN, Pin.PULL_UP)
m3_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(3, m3_b, True))

# M4: EncA=0, EncB=1, Rev=True
m4_a = Pin(0, Pin.IN, Pin.PULL_UP); m4_b = Pin(1, Pin.IN, Pin.PULL_UP)
m4_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(4, m4_b, True))

poll_obj = select.poll()
poll_obj.register(sys.stdin, select.POLLIN)

def stop_all_motors():
    for i in range(1, 5): set_motor(i, 0)

# --- MOTION ENGINE ---
def drive_distance(target_meters):
    target_ticks = (target_meters / METERS_PER_TICK) * DISTANCE_MULTIPLIER
    for i in range(1, 5): ticks[i] = 0 
    
    Kp, max_speed, min_speed, current_speed, accel_rate = 5.0, 1500, 800, 0, 50
    print(f"Driving {target_meters}m...")
    
    while True:
        if poll_obj.poll(0):
            if sys.stdin.readline().strip() == "STOP": break

        current_ticks = sorted([abs(t) for t in ticks.values()])
        avg_ticks = (current_ticks[1] + current_ticks[2]) / 2.0
        error = target_ticks - avg_ticks
        
        if abs(error) < 25: break
            
        desired_speed = error * Kp
        current_speed = min(abs(desired_speed), current_speed + accel_rate)
        output_pwm = min(max_speed, max(min_speed, current_speed))
        final_pwm = output_pwm if error > 0 else -output_pwm

        for i in range(1, 5):
            power = final_pwm
            if i == 1 or i == 3: power = int(final_pwm * 0.90) # Steering Bias
            set_motor(i, power)
        time.sleep(0.02) 

    stop_all_motors()
    print("Done.")

def turn_robot(degrees):
    arc_length = (math.pi * TRACK_WIDTH_M * abs(degrees)) / 360
    target_ticks = (arc_length / METERS_PER_TICK) * DISTANCE_MULTIPLIER
    for i in range(1, 5): ticks[i] = 0 
    
    Kp, max_speed, min_speed, current_speed, accel_rate = 5.0, 1500, 800, 0, 50
    print(f"Turning {degrees} degrees...")
    
    while True:
        if poll_obj.poll(0):
            if sys.stdin.readline().strip() == "STOP": break

        current_ticks = sorted([abs(t) for t in ticks.values()])
        avg_ticks = (current_ticks[1] + current_ticks[2]) / 2.0
        error = target_ticks - avg_ticks
        
        if abs(error) < 25: break
            
        desired_speed = error * Kp
        current_speed = min(abs(desired_speed), current_speed + accel_rate)
        output_pwm = min(max_speed, max(min_speed, current_speed))
        final_pwm = output_pwm if degrees > 0 else -output_pwm

        # Pivot: Right sides (1,2) opposite of Left sides (3,4)
        set_motor(1, -final_pwm); set_motor(2, -final_pwm)
        set_motor(3, final_pwm); set_motor(4, final_pwm)
        time.sleep(0.02) 

    stop_all_motors()

# --- MAIN LOOP ---
init_motor_hat()
try:
    while True:
        if poll_obj.poll(0):
            cmd = sys.stdin.readline().strip()
            if cmd.startswith("DRIVE:"):
                drive_distance(float(cmd.split(":")[1]))
            elif cmd.startswith("TURN:"):
                turn_robot(float(cmd.split(":")[1]))
            elif cmd == "STOP":
                stop_all_motors()
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_all_motors()