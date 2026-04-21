import sys
import termios
import tty
import select
import board
from adafruit_motorkit import MotorKit

# Initialize the motor hat
kit = MotorKit(i2c=board.I2C())

def get_key(timeout=0.4):
    """Reads a single keypress from standard input without needing 'Enter'."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Set terminal to 'raw' mode to capture single keystrokes
        tty.setraw(sys.stdin.fileno())
        
        # Wait for input, but timeout if nothing is pressed
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
        else:
            ch = None
    finally:
        # Restore normal terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("--- Raspberry Pi Real-Time RC Control ---")
print("Press and hold W/A/S/D to move.")
print("Release the key to stop.")
print("Press 'Q' to quit.\n")

# Motor speeds (0.0 to 1.0)
SPEED = 0.8
TURN_SPEED = 0.6

try:
    current_action = None
    while True:
        # 0.4s timeout handles the natural gap in SSH key-repeat rates
        key = get_key(timeout=0.4)
        
        if key in ['q', 'Q']:
            print("\r\nQuitting...")
            break
            
        elif key in ['w', 'W']:
            kit.motor1.throttle = SPEED
            kit.motor2.throttle = SPEED
            if current_action != 'W':
                print("\rMoving Forward...      ", end="")
                current_action = 'W'
                
        elif key in ['s', 'S']:
            kit.motor1.throttle = -SPEED
            kit.motor2.throttle = -SPEED
            if current_action != 'S':
                print("\rMoving Backward...     ", end="")
                current_action = 'S'
                
        elif key in ['a', 'A']:
            kit.motor1.throttle = -TURN_SPEED
            kit.motor2.throttle = TURN_SPEED
            if current_action != 'A':
                print("\rTurning Left...        ", end="")
                current_action = 'A'
                
        elif key in ['d', 'D']:
            kit.motor1.throttle = TURN_SPEED
            kit.motor2.throttle = -TURN_SPEED
            if current_action != 'D':
                print("\rTurning Right...       ", end="")
                current_action = 'D'
                
        elif key is None:
            # If no key is received within 0.4 seconds, stop the motors
            kit.motor1.throttle = 0.0
            kit.motor2.throttle = 0.0
            if current_action != 'STOP':
                print("\rStopped.               ", end="")
                current_action = 'STOP'

except KeyboardInterrupt:
    print("\r\nProgram interrupted by user.")
finally:
    # FAILSAFE: Always ensure motors are stopped when the script crashes or exits
    kit.motor1.throttle = 0.0
    kit.motor2.throttle = 0.0
    print("\r\nMotors stopped. Goodbye!")