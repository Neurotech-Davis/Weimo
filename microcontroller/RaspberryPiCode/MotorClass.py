import time
from adafruit_motorkit import MotorKit
import board

class WaymoController:
    def __init__(self, max_accel_step=0.05):
        """
        Initializes the motor controller.
        :param max_accel_step: The maximum change in throttle per update loop.
                               Lower = smoother/slower, Higher = jerkier/faster.
        """
        # Initialize the I2C motor hat
        self.kit = MotorKit(i2c=board.I2C())
        
        # Map logical positions to physical motor ports.
        # UPDATE THESE based on your diagnostic script results!
        self.motor_fl = self.kit.motor1  # Front-Left
        self.motor_fr = self.kit.motor2  # Front-Right
        self.motor_bl = self.kit.motor3  # Back-Left
        self.motor_br = self.kit.motor4  # Back-Right

        # State tracking for ramping
        self.max_accel_step = max_accel_step
        self.current_throttles = [0.0, 0.0, 0.0, 0.0] # FL, FR, BL, BR
        self.target_throttles = [0.0, 0.0, 0.0, 0.0]
        
    def drive(self, forward, strafe, turn):
        """
        Sets the target vectors for the robot.
        :param forward: -1.0 (backward) to 1.0 (forward)
        :param strafe:  -1.0 (left) to 1.0 (right)
        :param turn:    -1.0 (counter-clockwise) to 1.0 (clockwise)
        """
        # Mecanum kinematic equations
        fl = forward + strafe + turn
        fr = forward - strafe - turn
        bl = forward - strafe + turn
        br = forward + strafe - turn
        
        # Normalize the speeds so no motor is asked to go past 1.0 or -1.0
        # We include 1.0 in the max() calculation to avoid dividing by 0 or scaling up small numbers
        max_val = max(abs(fl), abs(fr), abs(bl), abs(br), 1.0)
        
        self.target_throttles = [
            fl / max_val,
            fr / max_val,
            bl / max_val,
            br / max_val
        ]

    def update(self):
        """
        Applies the ramping logic. This MUST be called frequently in your main loop.
        """
        for i in range(4):
            # Calculate the difference between where we are and where we want to be
            diff = self.target_throttles[i] - self.current_throttles[i]
            
            # Clamp the change to our maximum acceleration step
            if abs(diff) > self.max_accel_step:
                step = self.max_accel_step if diff > 0 else -self.max_accel_step
            else:
                step = diff
                
            self.current_throttles[i] += step

        # Apply the smoothed throttles to the hardware
        self.motor_fl.throttle = self.current_throttles[0]
        self.motor_fr.throttle = self.current_throttles[1]
        self.motor_bl.throttle = self.current_throttles[2]
        self.motor_br.throttle = self.current_throttles[3]

    def stop(self):
        """Immediately halts all motors (bypasses ramping)."""
        self.target_throttles = [0.0, 0.0, 0.0, 0.0]
        self.current_throttles = [0.0, 0.0, 0.0, 0.0]
        self.motor_fl.throttle = 0.0
        self.motor_fr.throttle = 0.0
        self.motor_bl.throttle = 0.0
        self.motor_br.throttle = 0.0