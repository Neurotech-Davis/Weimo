from machine import Pin, I2C, time_pulse_us
import time
import math
import sys
import select

# --- HARDWARE CONSTANTS ---
WHEEL_DIAMETER_M = 0.066  
WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER_M
TICKS_PER_REV = 247  
METERS_PER_TICK = WHEEL_CIRCUMFERENCE / TICKS_PER_REV

# --- I2C MOTOR HAT DRIVER ---
class MotorHat:
    def __init__(self, i2c_bus, addr=0x60):
        self.i2c = i2c_bus
        self.addr = addr
        self._init_pca9685()

    def _init_pca9685(self):
        try:
            self.i2c.writeto_mem(self.addr, 0x00, b'\x20') 
            time.sleep(0.05)
            self.i2c.writeto_mem(self.addr, 0x01, b'\x04') 
        except OSError:
            print("[!] Motor Hat not detected on I2C bus.")

    def set_pwm(self, channel, on, off):
        data = bytearray([on & 0xFF, on >> 8, off & 0xFF, off >> 8])
        try:
            self.i2c.writeto_mem(self.addr, 0x06 + 4 * channel, data)
        except OSError:
            pass

    def set_pin(self, channel, state):
        if state == 1:
            self.set_pwm(channel, 4096, 0) 
        else:
            self.set_pwm(channel, 0, 4096) 

# --- INDIVIDUAL WHEEL CLASS ---
class Wheel:
    def __init__(self, motor_id, hat, pwm_pin, in1, in2, enc_a_pin, enc_b_pin, reverse_enc=False):
        self.motor_id = motor_id
        self.hat = hat
        self.pwm_pin = pwm_pin
        self.in1 = in1
        self.in2 = in2
        
        # Software variables
        self.ticks = 0
        self.reverse_enc = reverse_enc
        self.power_bias = 1.0 # 1.0 = 100% power, 0.9 = 90% power
        
        # Hardware Pins
        self.enc_a = Pin(enc_a_pin, Pin.IN, Pin.PULL_UP)
        self.enc_b = Pin(enc_b_pin, Pin.IN, Pin.PULL_UP)
        
        # Attach Interrupt
        self.enc_a.irq(trigger=Pin.IRQ_RISING, handler=self._enc_handler)

    def _enc_handler(self, pin):
        # A lightweight handler to minimize CPU interrupt time
        val = self.enc_b.value()
        if self.reverse_enc:
            val = not val
        if val:
            self.ticks += 1
        else:
            self.ticks -= 1

    def reset_ticks(self):
        self.ticks = 0

    def drive(self, speed):
        # Apply the individual wheel bias
        adjusted_speed = int(speed * self.power_bias)
        adjusted_speed = max(-4095, min(4095, adjusted_speed))
        
        if adjusted_speed == 0:
            self.hat.set_pin(self.in1, 0) 
            self.hat.set_pin(self.in2, 0)
            self.hat.set_pwm(self.pwm_pin, 0, 0)
        elif adjusted_speed > 0:
            self.hat.set_pin(self.in1, 0) 
            self.hat.set_pin(self.in2, 1)
            self.hat.set_pwm(self.pwm_pin, 0, adjusted_speed)
        else:
            self.hat.set_pin(self.in1, 1) 
            self.hat.set_pin(self.in2, 0)
            self.hat.set_pwm(self.pwm_pin, 0, abs(adjusted_speed))

    def stop(self):
        self.drive(0)


# --- SETUP HARDWARE ---
i2c = I2C(0, sda=Pin(16), scl=Pin(17), freq=100000)
hat = MotorHat(i2c)

# FIXED CROSS-MATCHED POWER & ENCODERS
# Wheels: (ID, Hat, PWM, IN1, IN2, EncA, EncB, ReverseEncoder)

# M1 pairs Driver Port 8 with Encoder Pins 2 & 3
m1 = Wheel(1, hat, 8, 10, 9,  0, 1,  False)  
m2 = Wheel(2, hat, 13, 11, 12, 19, 18, False) 
m3 = Wheel(3, hat, 2, 4, 3,  15, 14, True)    
m4 = Wheel(4, hat, 7, 5, 6,  2, 3,  True)     

wheels = [m1, m2, m3, m4]

# TUNE: Apply steering bias directly to the objects if needed
m1.power_bias = 1.0
m2.power_bias = 1.0
m3.power_bias = 1.0
m4.power_bias = 1.0

# --- HELPER FUNCTIONS ---
def stop_all():
    for w in wheels:
        w.stop()

def reset_all_ticks():
    for w in wheels:
        w.reset_ticks()

def print_telemetry():
    print(f"TICKS | M1:{m1.ticks:>5} | M2:{m2.ticks:>5} | M3:{m3.ticks:>5} | M4:{m4.ticks:>5}")

# --- COMMAND PARSER ---
poll_obj = select.poll()
poll_obj.register(sys.stdin, select.POLLIN)

print("\n--- PICO ROBOTICS CORE ---")
print("Commands available:")
print("  TEST:1   -> Runs M1 only for 1 second (Tests hardware)")
print("  DRIVE:1500 -> Drives all motors at PWM 1500 for 1 second")
print("  STOP     -> Kills motors")

try:
    while True:
        if poll_obj.poll(0):
            command = sys.stdin.readline().strip().upper()
            
            if command == "STOP":
                stop_all()
                print("Motors Stopped.")
                
            elif command.startswith("TEST:"):
                # Run a single motor for diagnostic testing
                motor_id = int(command.split(":")[1])
                target_wheel = next((w for w in wheels if w.motor_id == motor_id), None)
                
                if target_wheel:
                    reset_all_ticks()
                    print(f"Testing M{motor_id} at 1500 PWM for 1 second...")
                    target_wheel.drive(1500)
                    time.sleep(1)
                    target_wheel.stop()
                    print_telemetry()
                else:
                    print("Invalid Motor ID")

            elif command.startswith("DRIVE:"):
                # Drive all motors for a short burst to check telemetry
                pwm_val = int(command.split(":")[1])
                reset_all_ticks()
                print(f"Driving all motors at {pwm_val} for 1 second...")
                for w in wheels:
                    w.drive(pwm_val)
                time.sleep(1)
                stop_all()
                print_telemetry()
                
        time.sleep(0.05)

except KeyboardInterrupt:
    stop_all()
    print("\nProgram exit.")