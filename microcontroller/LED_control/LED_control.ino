// I defined which Arduino pins were connected to the LED and the buzzer.
const int ledPin = 13;
const int buzzerPin = 12; // Just a guess btw
char inputChar;  // This variable stored the incoming character received from MATLAB.


void setup() {
  // I configured the LED and buzzer pins as OUTPUT so I could control their voltage levels.
  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
  // I started serial communication at 9600 baud, which matched the settings in MATLAB.
  Serial.begin(9600);
}

void loop() {
 // I continuously checked whether any data was available from MATLAB on the serial port.
  if (Serial.available() > 0) {
    // I read the incoming character from the serial buffer.
    inputChar = Serial.read();
    
    // If I received '1', I turned on both the LED and the buzzer by setting their pins HIGH.
    if (inputChar == '1') {
      digitalWrite(ledPin, HIGH);
      digitalWrite(buzzerPin, HIGH);
    }
    // If I received '0', I turned off the LED and the buzzer by setting their pins LOW.
    else if (inputChar == '0') {
      digitalWrite(ledPin, LOW);
      digitalWrite(buzzerPin, LOW);
    }
  }
}