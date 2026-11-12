#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <string.h>

BLEServer* pServer = nullptr;
BLECharacteristic* pCharacteristic = nullptr;

bool deviceConnected = false;

const int IN1 = 4;
const int IN2 = 5;
const int IN3 = 6;
const int IN4 = 7;

// Keep the existing wheel mapping unchanged.
const int LU = IN2;
const int LD = IN1;
const int RU = IN3;
const int RD = IN4;

const int NUM_SERVOS = 4;
const int SERVO_PINS[NUM_SERVOS] = {15, 16, 17, 18};

// Servo PWM: 50 Hz, 14-bit, pulse 500..2400 us (same range as before).
const int SERVO_FREQ_HZ = 50;
const int SERVO_PWM_BITS = 14;
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2400;

void writeServo(int index, int angle) {
  uint32_t us = SERVO_MIN_US + (uint32_t)angle * (SERVO_MAX_US - SERVO_MIN_US) / 180;
  uint32_t duty = (uint64_t)us * ((1UL << SERVO_PWM_BITS) - 1) / (1000000UL / SERVO_FREQ_HZ);
  ledcWrite(SERVO_PINS[index], duty);
}

int currentAngles[NUM_SERVOS] = {90, 90, 90, 90};
int targetAngles[NUM_SERVOS] = {90, 90, 90, 90};
int servoSpeeds[NUM_SERVOS] = {60, 60, 60, 60};  // degrees per second
unsigned long lastServoUpdate[NUM_SERVOS] = {0, 0, 0, 0};

void motors(int lu, int ld, int ru, int rd) {
  digitalWrite(LU, LOW);
  digitalWrite(LD, LOW);
  digitalWrite(RU, LOW);
  digitalWrite(RD, LOW);

  digitalWrite(LU, lu);
  digitalWrite(LD, ld);
  digitalWrite(RU, ru);
  digitalWrite(RD, rd);
}

void handleCommand(String message) {
  message.trim();
  if (message.length() == 0) return;

  // Motor commands: U, D, L, R, S
  if (message.length() == 1) {
    char command = message.charAt(0);
    Serial.print("Motor command: ");
    Serial.println(command);

    switch (command) {
      case 'U': motors(1, 0, 1, 0); break;
      case 'D': motors(0, 1, 0, 1); break;
      case 'L': motors(0, 1, 1, 0); break;
      case 'R': motors(1, 0, 0, 1); break;
      case 'S': motors(0, 0, 0, 0); break;
      default: Serial.println("Unknown motor command"); break;
    }
    return;
  }

  // Servo command format: V<servo-index> <angle> <speed deg/sec>
  // Examples: V0 90 60, V1 120 30, V3 160 120
  if (message.charAt(0) == 'V') {
    int servoIndex = -1;
    int angle = -1;
    int speed = -1;

    if (sscanf(message.c_str(), "V%d %d %d", &servoIndex, &angle, &speed) == 3) {
      if (servoIndex >= 0 && servoIndex < NUM_SERVOS &&
          angle >= 0 && angle <= 180 &&
          speed >= 1 && speed <= 180) {
        targetAngles[servoIndex] = angle;
        servoSpeeds[servoIndex] = speed;
        Serial.printf("Servo %d | Target: %d | Speed: %d deg/sec\n",
                      servoIndex, angle, speed);
      } else {
        Serial.println("Servo index, angle or speed out of range");
      }
    } else {
      Serial.println("Invalid servo command. Use: V0 90 60");
    }
  }
}

// Moves each servo 1 degree at a time toward its target at its set speed.
void updateServos() {
  unsigned long now = millis();

  for (int i = 0; i < NUM_SERVOS; i++) {
    if (currentAngles[i] == targetAngles[i]) continue;

    unsigned long interval = 1000UL / servoSpeeds[i];

    if (now - lastServoUpdate[i] >= interval) {
      lastServoUpdate[i] = now;

      if (currentAngles[i] < targetAngles[i]) currentAngles[i]++;
      else currentAngles[i]--;

      writeServo(i, currentAngles[i]);
    }
  }
}

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    deviceConnected = true;
    Serial.println("BLE Client Connected");
  }

  void onDisconnect(BLEServer* server) override {
    deviceConnected = false;
    motors(0, 0, 0, 0);
    Serial.println("BLE Client Disconnected");
    BLEDevice::startAdvertising();
  }
};

class CharacteristicCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) override {
    String received = characteristic->getValue();

    if (received.length() == 0) return;

    handleCommand(received);
  }
};

void setup() {
  Serial.begin(115200);

  pinMode(LU, OUTPUT);
  pinMode(LD, OUTPUT);
  pinMode(RU, OUTPUT);
  pinMode(RD, OUTPUT);
  motors(0, 0, 0, 0);

  // Servo PWM setup (50 Hz, direct LEDC). Servos must use an external 5V supply.
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (!ledcAttach(SERVO_PINS[i], SERVO_FREQ_HZ, SERVO_PWM_BITS)) {
      Serial.printf("ERROR: could not attach servo %d on pin %d\n", i, SERVO_PINS[i]);
    }
    writeServo(i, 90);
  }

  BLEDevice::init("AI_Car");

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService* pService =
    pServer->createService("12345678-1234-1234-1234-1234567890AB");

  pCharacteristic = pService->createCharacteristic(
    "87654321-4321-4321-4321-BA0987654321",
    BLECharacteristic::PROPERTY_WRITE |
    BLECharacteristic::PROPERTY_WRITE_NR
  );

  pCharacteristic->setCallbacks(new CharacteristicCallbacks());

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID("12345678-1234-1234-1234-1234567890AB");
  pAdvertising->start();

  Serial.println("AI_Car BLE Ready");
  Serial.println("Motor commands: U, D, L, R, S");
  Serial.println("Servo commands: V<0-3> <angle 0-180> <speed 1-180 deg/sec>, e.g. V0 90 60");
}

void loop() {
  updateServos();
  delay(2);
}
