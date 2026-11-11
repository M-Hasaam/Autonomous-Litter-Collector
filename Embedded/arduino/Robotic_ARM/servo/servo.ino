#include <Servo.h>

// Global configuration
const int NUM_SERVOS = 4;
const int SERVO_PINS[NUM_SERVOS] = {6, 9, 10, 11}; // Signal pins

Servo servos[NUM_SERVOS];
String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  inputBuffer.reserve(32);
  
  // Attach all servos and set them to 90 degrees neutral position
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(90);
  }
}

void loop() {
  // Read incoming characters until a newline is reached
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        int servoIndex = -1;
        int targetAngle = -1;
        if (sscanf(inputBuffer.c_str(), "%d %d", &servoIndex, &targetAngle) == 2) {
          if (servoIndex >= 0 && servoIndex < NUM_SERVOS && targetAngle >= 0 && targetAngle <= 180) {
            servos[servoIndex].write(targetAngle);
          }
        }
      }
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}