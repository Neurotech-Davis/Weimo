#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Create the motor shield object with the default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 

// Define the 4 motors
// initialize motor pointers
Adafruit_DCMotor *motor1 = AFMS.getMotor(1);
Adafruit_DCMotor *motor2 = AFMS.getMotor(2);
Adafruit_DCMotor *motor3 = AFMS.getMotor(3);
Adafruit_DCMotor *motor4 = AFMS.getMotor(4);

void setup() {
Serial.begin(9600);           // Start Serial Monitor for debugging
  Serial.println("Motor Test!");

  if (!AFMS.begin()) {         // Start the shield
    Serial.println("Could not find Motor Shield. Check connections.");
    while (1); // Stop here if shield is not found
  }
  Serial.println("Motor Shield found!");
  
  // Set speed (0-255)
  motor1->setSpeed(150);
  motor2->setSpeed(150);
  motor3->setSpeed(150);
  motor4->setSpeed(150);
}
void loop() {
Serial.println("Running Forward...");

  // Run motors
  motor1->run(FORWARD);
  motor2->run(FORWARD);
  motor3->run(FORWARD);
  motor4->run(FORWARD);

  delay(1000); // Wait 1 second

  Serial.println("Stopping...");

  // Stop motors
  motor1->run(RELEASE);
  motor2->run(RELEASE);
  motor3->run(RELEASE);
  motor4->run(RELEASE);

  delay(2000); // Wait 2 seconds before repeating
}
