from machine import Pin, I2C
import time
import math
import sys
import select

# --- HARDWARE CONSTANTS ---
WHEEL_DIAMETER_M = 0.066
WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER_M
TICKS_PER_REV = 990
METERS_PER_TICK = WHEEL_CIRCUMFERENCE / TICKS_PER_REV
TRACK_WIDTH_M = 0.125

# --- I2C & MOTOR HAT SETUP ---
i2c = I2C(0, sda=Pin(16), scl=Pin(17), freq=100000)
PCA9685_ADDR = 0x60


def init_motor_hat():
    try:
        i2c.writeto_mem(PCA9685_ADDR, 0x00, b"\x20")
        time.sleep(0.05)
        i2c.writeto_mem(PCA9685_ADDR, 0x01, b"\x04")
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
    if state == 1:
        set_pwm(channel, 4096, 0)
    else:
        set_pwm(channel, 0, 4096)


def set_motor(motor_num, speed):
    pins = {1: (8, 10, 9), 2: (13, 11, 12), 3: (2, 4, 3), 4: (7, 5, 6)}
    pwm_pin, in1_pin, in2_pin = pins[motor_num]
    speed = max(-4095, min(4095, int(speed)))

    if speed == 0:
        set_pin(in1_pin, 0)
        set_pin(in2_pin, 0)
        set_pwm(pwm_pin, 0, 0)
    elif speed > 0:
        set_pin(in1_pin, 0)
        set_pin(in2_pin, 1)
        set_pwm(pwm_pin, 0, speed)
    else:
        set_pin(in1_pin, 1)
        set_pin(in2_pin, 0)
        set_pwm(pwm_pin, 0, abs(speed))


# --- ENCODER SETUP ---
ticks = {1: 0, 2: 0, 3: 0, 4: 0}


def make_handler(motor_id, pin_b, reverse_polarity=False):
    def handler(pin):
        val = pin_b.value()
        if reverse_polarity:
            val = not val
        if val:
            ticks[motor_id] += 1
        else:
            ticks[motor_id] -= 1

    return handler


# M1: EncA=2, EncB=3
m1_a = Pin(2, Pin.IN, Pin.PULL_UP)
m1_b = Pin(3, Pin.IN, Pin.PULL_UP)
m1_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(1, m1_b, False))

# M2: EncA=19, EncB=18
m2_a = Pin(19, Pin.IN, Pin.PULL_UP)
m2_b = Pin(18, Pin.IN, Pin.PULL_UP)
m2_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(2, m2_b, False))

# M3: EncA=15, EncB=14, Reversed
m3_a = Pin(15, Pin.IN, Pin.PULL_UP)
m3_b = Pin(14, Pin.IN, Pin.PULL_UP)
m3_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(3, m3_b, True))

# M4: EncA=0, EncB=1, Reversed
m4_a = Pin(0, Pin.IN, Pin.PULL_UP)
m4_b = Pin(1, Pin.IN, Pin.PULL_UP)
m4_a.irq(trigger=Pin.IRQ_RISING, handler=make_handler(4, m4_b, True))

poll_obj = select.poll()
poll_obj.register(sys.stdin, select.POLLIN)


def stop_all_motors():
    for i in range(1, 5):
        set_motor(i, 0)


# --- ROBUST TICK AVERAGE ---
def robust_avg_ticks():
    vals = sorted([abs(ticks[i]) for i in range(1, 5)])
    return (vals[1] + vals[2]) / 2.0


# --- DEMO-DAY DYNAMIC SCALING ---
def get_drive_multiplier(target_meters):
    val = abs(target_meters)
    if val <= 0.10:
        return 0.71
    elif val <= 0.15:
        return 0.58
    elif val <= 0.20:
        return 0.56
    elif val <= 0.30:
        return 0.53
    elif val <= 0.40:
        return 0.50
    elif val <= 0.50:
        return 0.47
    elif val <= 0.60:
        return 0.47
    elif val <= 0.70:
        return 0.46
    elif val <= 0.80:
        return 0.45
    else:
        return 0.39


def get_turn_multiplier(degrees):
    # Turns behave differently mechanically.
    # Testing proved a 1.0 multiplier yields between 45 and 55 degrees.
    return 0.90


# --- MOTION ENGINE ---
def drive_distance(target_meters):
    current_multiplier = get_drive_multiplier(target_meters)
    target_ticks = (target_meters / METERS_PER_TICK) * current_multiplier

    for i in range(1, 5):
        ticks[i] = 0

    Kp = 5.0
    max_speed = 1100  # Lowered for stable SNR
    min_speed = 800
    accel_rate = 50
    current_speed = 0
    loop_count = 0

    timeout_ms = max(3000, int(abs(target_meters) * 7000))
    start_time = time.ticks_ms()

    direction = 1 if target_meters >= 0 else -1
    print(
        f"Driving {target_meters}m (target={target_ticks:.0f} ticks, multiplier={current_multiplier})..."
    )

    while True:
        if poll_obj.poll(0):
            if sys.stdin.readline().strip() == "STOP":
                stop_all_motors()
                print("OK:STOPPED")
                return

        avg_ticks = robust_avg_ticks()
        error = target_ticks - avg_ticks

        loop_count += 1
        if loop_count % 5 == 0:
            print(
                f"TICKS | M1(R):{abs(ticks[1]):>4} M2(R):{abs(ticks[2]):>4} "
                f"M3(L):{abs(ticks[3]):>4} M4(L):{abs(ticks[4]):>4} "
                f"| Dist: {avg_ticks:.0f}/{target_ticks:.0f}"
            )

        if abs(error) < 25:
            break

        desired_speed = min(max_speed, error * Kp)
        current_speed = min(desired_speed, current_speed + accel_rate)
        output_pwm = max(min_speed, current_speed)
        final_pwm = output_pwm * direction

        for i in range(1, 5):
            power = final_pwm
            # Gentle 5% bias to the right side (M1 & M2)
            if i == 1 or i == 2:
                power = int(final_pwm * 0.95)
            set_motor(i, power)

        time.sleep(0.02)

    stop_all_motors()
    print("OK:DONE")


def turn_robot(degrees):
    arc_length = (math.pi * TRACK_WIDTH_M * abs(degrees)) / 360
    current_multiplier = get_turn_multiplier(degrees)
    target_ticks = (arc_length / METERS_PER_TICK) * current_multiplier

    for i in range(1, 5):
        ticks[i] = 0

    Kp = 5.0
    max_speed = 1100  # Lowered for stable SNR
    min_speed = 800
    accel_rate = 50
    current_speed = 0
    loop_count = 0

    timeout_ms = max(3000, int(abs(degrees) * 50))
    start_time = time.ticks_ms()

    print(
        f"Turning {degrees} degrees (target={target_ticks:.0f} ticks, multiplier={current_multiplier})..."
    )

    while True:
        if poll_obj.poll(0):
            if sys.stdin.readline().strip() == "STOP":
                break

        avg_ticks = robust_avg_ticks()
        error = target_ticks - avg_ticks

        loop_count += 1
        if loop_count % 5 == 0:
            print(
                f"TICKS | M1(R):{abs(ticks[1]):>4} M2(R):{abs(ticks[2]):>4} "
                f"M3(L):{abs(ticks[3]):>4} M4(L):{abs(ticks[4]):>4} "
                f"| Turn: {avg_ticks:.0f}/{target_ticks:.0f}"
            )

        if abs(error) < 25:
            break

        desired_speed = min(max_speed, error * Kp)
        current_speed = min(desired_speed, current_speed + accel_rate)
        output_pwm = max(min_speed, current_speed)

        if degrees > 0:
            set_motor(1, -output_pwm)
            set_motor(2, -output_pwm)
            set_motor(3, output_pwm)
            set_motor(4, output_pwm)
        else:
            set_motor(1, output_pwm)
            set_motor(2, output_pwm)
            set_motor(3, -output_pwm)
            set_motor(4, -output_pwm)

        time.sleep(0.02)

    stop_all_motors()
    print("OK:DONE")


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
                print("OK:STOPPED")
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_all_motors()
