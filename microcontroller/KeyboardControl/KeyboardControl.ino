#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 
Adafruit_DCMotor *fl = AFMS.getMotor(1); // Front Left
Adafruit_DCMotor *fr = AFMS.getMotor(2); // Front Right
Adafruit_DCMotor *bl = AFMS.getMotor(3); // Back Left
Adafruit_DCMotor *br = AFMS.getMotor(4); // Back Right

void setup() {
  Serial.begin(115200); // Faster communication speed
  Serial.println("Spresense Mecanum Controller Ready.");
  Serial.println("Controls: W,A,S,D (Move) | Q,E (Strafe) | Z (Stop)");

  if (!AFMS.begin()) {
    Serial.println("Could not find Motor Shield.");
    while (1);
  }
}

// THIS is the Engineering part: A dedicated Kinematics Function
// It takes desired X, Y, and Rotation speeds (-255 to 255)
void setMecanumSpeed(int y, int x, int rot) {
  
  // 1. Calculate motor speeds using vector mixing
  // Note: You might need to flip +/- signs depending on your specific wiring!
  int fl_speed = y + x + rot;
  int fr_speed = y - x - rot;
  int bl_speed = y - x + rot;
  int br_speed = y + x - rot;

  // 2. Normalize speeds (ensure we don't exceed 255)
  // If we calculate 300, we must scale everything down so the max is 255
  int max_val = max(abs(fl_speed), max(abs(fr_speed), max(abs(bl_speed), abs(br_speed))));
  if (max_val > 255) {
    fl_speed = map(fl_speed, -max_val, max_val, -255, 255);
    fr_speed = map(fr_speed, -max_val, max_val, -255, 255);
    bl_speed = map(bl_speed, -max_val, max_val, -255, 255);
    br_speed = map(br_speed, -max_val, max_val, -255, 255);
  }

  // 3. Send commands to motors
  setMotor(fl, fl_speed);
  setMotor(fr, fr_speed);
  setMotor(bl, bl_speed);
  setMotor(br, br_speed);
}

// Helper function to handle Direction + Speed automatically
void setMotor(Adafruit_DCMotor *motor, int speed) {
  if (speed > 0) {
    motor->run(FORWARD);
    motor->setSpeed(speed);
  } else if (speed < 0) {
    motor->run(BACKWARD);
    motor->setSpeed(abs(speed)); // Speed must be positive
  } else {
    motor->run(RELEASE);
  }
}

void loop() {
  // CHECK FOR INPUT
  if (Serial.available() > 0) {
    char cmd = Serial.read(); // Read the typed key

    // Define Base Speed
    int s = 150; 

    // INPUT LAYER: Map Keys to Vectors (Y, X, Rot)
    switch(cmd) {
      case 'w': setMecanumSpeed(s, 0, 0);  break; // Forward
      case 's': setMecanumSpeed(-s, 0, 0); break; // Backward
      case 'a': setMecanumSpeed(0, 0, s);  break; // Rotate Left (Pivot)
      case 'd': setMecanumSpeed(0, 0, -s); break; // Rotate Right (Pivot)
      case 'q': setMecanumSpeed(0, -s, 0); break; // Strafe Left
      case 'e': setMecanumSpeed(0, s, 0);  break; // Strafe Right
      case 'z': setMecanumSpeed(0, 0, 0);  break; // Stop
    }
  }
}