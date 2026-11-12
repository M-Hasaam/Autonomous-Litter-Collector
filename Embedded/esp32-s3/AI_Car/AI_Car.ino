#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <string.h>
#include <Wire.h>
#include <math.h>

// HC-SR04 ultrasonic sensor pins.
// ECHO is 5V, so it goes through a divider (1k in series, 2k to GND) before the pin.
const int TRIG_PIN = 10;
const int ECHO_PIN = 11;

// Active buzzer: + to BUZZER_PIN, - to GND. Beeps while an obstacle is closer than BUZZER_ON_CM.
// (An active buzzer has its own tone generator; all 8 PWM channels are already used by wheels and servos.)
const int BUZZER_PIN = 14;
const float BUZZER_ON_CM = 20.0f;                 // alarm starts below this distance
const float BUZZER_OFF_CM = 23.0f;                // alarm stops above this (gap avoids flicker at the limit)
const unsigned long BUZZER_BEEP_MS = 120;         // beep on / off time while alarming

const char* DISTANCE_CHAR_UUID = "11223344-5566-7788-99AA-BBCCDDEEFF00";
const unsigned long DISTANCE_INTERVAL_MS = 100;   // time between distance readings
const unsigned long ECHO_TIMEOUT_US = 25000;      // ~4.3 m max range

// MPU-6050 (I2C). AD0 -> GND gives address 0x68.
const int MPU_SDA_PIN = 41;
const int MPU_SCL_PIN = 42;
const uint8_t MPU_ADDR = 0x68;
const unsigned long MPU_READ_INTERVAL_MS = 20;    // sensor read / filter update
const unsigned long MPU_NOTIFY_INTERVAL_MS = 100; // BLE + Serial output
const float MPU_FILTER_ALPHA = 0.98f;             // complementary filter: gyro vs accel weight
const float MPU_GYRO_LSB_PER_DPS = 65.5f;         // gyro range +-500 deg/s
const float MPU_YAW_DEADBAND_DPS = 0.3f;          // ignore tiny gyro noise so heading doesn't creep
const int MPU_CAL_SAMPLES = 200;                  // gyro offset calibration (keep the car still)
const char* MPU_CHAR_UUID = "22334455-6677-8899-AABB-CCDDEEFF0011";

BLEServer* pServer = nullptr;
BLECharacteristic* pCharacteristic = nullptr;
BLECharacteristic* pDistanceCharacteristic = nullptr;
BLECharacteristic* pMpuCharacteristic = nullptr;

bool deviceConnected = false;
unsigned long lastDistanceRead = 0;
bool distanceAlarm = false;                       // true while an obstacle is too close

bool mpuOk = false;
float mpuPitch = 0, mpuRoll = 0;                  // degrees
float mpuYaw = 0;                                 // heading change in degrees, + = counter-clockwise from above
float gyroBiasX = 0, gyroBiasY = 0, gyroBiasZ = 0; // deg/s measured while still
volatile bool mpuCalRequested = false;
volatile bool mpuYawResetRequested = false;
float mpuAx = 0, mpuAy = 0, mpuAz = 0;            // g
float mpuGx = 0, mpuGy = 0, mpuGz = 0;            // deg/s
unsigned long lastMpuRead = 0, lastMpuNotify = 0, lastMpuMicros = 0, lastMpuRetry = 0;

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

// Wheel speed: PWM on the motor-driver pins, one setting for all wheels (command P<0-100>, percent).
const int MOTOR_PINS[4] = {LU, LD, RU, RD};
const int MOTOR_PWM_FREQ_HZ = 1000;
const int MOTOR_PWM_BITS = 8;
const uint32_t MOTOR_PWM_MAX = (1UL << MOTOR_PWM_BITS) - 1;
int motorSpeedPct = 100;
int motorState[4] = {0, 0, 0, 0};  // last requested direction (1 = on) for LU, LD, RU, RD

int currentAngles[NUM_SERVOS] = {90, 90, 90, 90};
int targetAngles[NUM_SERVOS] = {90, 90, 90, 90};
int servoSpeeds[NUM_SERVOS] = {60, 60, 60, 60};  // degrees per second
unsigned long lastServoUpdate[NUM_SERVOS] = {0, 0, 0, 0};

void motors(int lu, int ld, int ru, int rd) {
  motorState[0] = lu;
  motorState[1] = ld;
  motorState[2] = ru;
  motorState[3] = rd;

  uint32_t duty = MOTOR_PWM_MAX * motorSpeedPct / 100;
  for (int i = 0; i < 4; i++) {
    ledcWrite(MOTOR_PINS[i], motorState[i] ? duty : 0);
  }
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
      case 'Z': mpuYawResetRequested = true; break;  // heading back to 0
      case 'C': mpuCalRequested = true; break;       // re-calibrate gyro (car must be still)
      default: Serial.println("Unknown motor command"); break;
    }
    return;
  }

  // Wheel speed: P<0-100> (percent, same for all wheels). Also applies to wheels that are running right now.
  if (message.charAt(0) == 'P') {
    int pct = -1;
    if (sscanf(message.c_str(), "P%d", &pct) == 1 && pct >= 0 && pct <= 100) {
      motorSpeedPct = pct;
      motors(motorState[0], motorState[1], motorState[2], motorState[3]);
      Serial.printf("Wheel speed: %d%%\n", pct);
    } else {
      Serial.println("Invalid speed. Use: P75");
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

// Returns distance in cm, or -1 if no echo was received (out of range).
float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (duration == 0) return -1;
  return duration * 0.0343f / 2.0f;
}

// Reads the sensor periodically, prints it and notifies the BLE client.
void updateDistance() {
  unsigned long now = millis();
  if (now - lastDistanceRead < DISTANCE_INTERVAL_MS) return;
  lastDistanceRead = now;

  float cm = readDistanceCm();

  // Alarm: on below BUZZER_ON_CM, off above BUZZER_OFF_CM, off when nothing is in range (-1).
  if (cm < 0) distanceAlarm = false;
  else if (cm < BUZZER_ON_CM) distanceAlarm = true;
  else if (cm > BUZZER_OFF_CM) distanceAlarm = false;

  char buf[16];
  if (cm < 0) snprintf(buf, sizeof(buf), "-1");
  else snprintf(buf, sizeof(buf), "%.1f", cm);

  Serial.printf("Distance: %s cm\n", buf);

  if (deviceConnected && pDistanceCharacteristic) {
    pDistanceCharacteristic->setValue((uint8_t*)buf, strlen(buf));
    pDistanceCharacteristic->notify();
  }
}

// Beeps (non-blocking) while distanceAlarm is set, silent otherwise.
void updateBuzzer() {
  static unsigned long lastToggle = 0;
  static bool on = false;
  if (!distanceAlarm) {
    if (on) { on = false; digitalWrite(BUZZER_PIN, LOW); }
    return;
  }
  unsigned long now = millis();
  if (now - lastToggle >= BUZZER_BEEP_MS) {  // alternate beep / silence
    on = !on;
    digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
    lastToggle = now;
  }
}

// Averages the gyro while the car is still to find its zero offset. Takes ~1 s.
void calibrateGyro() {
  float sx = 0, sy = 0, sz = 0;
  int n = 0;
  for (int i = 0; i < MPU_CAL_SAMPLES; i++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x43);  // GYRO_XOUT_H
    if (Wire.endTransmission(false) == 0 && Wire.requestFrom(MPU_ADDR, (uint8_t)6) == 6) {
      int16_t g[3];
      for (int k = 0; k < 3; k++) {
        uint8_t hi = Wire.read();
        uint8_t lo = Wire.read();
        g[k] = (int16_t)((hi << 8) | lo);
      }
      sx += g[0]; sy += g[1]; sz += g[2];
      n++;
    }
    delay(5);
  }
  if (n > 0) {
    gyroBiasX = sx / n / MPU_GYRO_LSB_PER_DPS;
    gyroBiasY = sy / n / MPU_GYRO_LSB_PER_DPS;
    gyroBiasZ = sz / n / MPU_GYRO_LSB_PER_DPS;
  }
  mpuYaw = 0;
  Serial.printf("Gyro bias: %.2f %.2f %.2f deg/s\n", gyroBiasX, gyroBiasY, gyroBiasZ);
}

bool mpuInit() {
  Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);
  Wire.setClock(400000);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);  // WHO_AM_I
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)1) != 1) return false;
  uint8_t who = Wire.read();
  Serial.printf("MPU WHO_AM_I = 0x%02X\n", who);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1: wake up, use X-gyro clock
  Wire.write(0x01);
  if (Wire.endTransmission() != 0) return false;

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1A);  // CONFIG: digital low-pass filter ~44 Hz (less noise)
  Wire.write(0x03);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);  // GYRO_CONFIG: +-500 deg/s so fast hand turns don't clip
  Wire.write(0x08);
  Wire.endTransmission();
  // Accel stays at default +-2g (16384 LSB/g).

  delay(100);
  calibrateGyro();
  return true;
}

// Reads the MPU and updates the tilt angles (complementary filter).
void updateMpu() {
  unsigned long now = millis();
  if (mpuOk && mpuCalRequested) {
    mpuCalRequested = false;
    calibrateGyro();
    lastMpuMicros = micros();
  }
  if (mpuYawResetRequested) {
    mpuYawResetRequested = false;
    mpuYaw = 0;
  }
  if (!mpuOk) {
    // Not found yet: retry every 2 s (e.g. wired after boot) and list I2C devices.
    if (now - lastMpuRetry >= 2000) {
      lastMpuRetry = now;
      mpuOk = mpuInit();
      lastMpuMicros = micros();
      if (mpuOk) {
        Serial.println("MPU-6050 ready");
      } else {
        Serial.print("MPU not found. I2C devices:");
        int found = 0;
        for (uint8_t a = 1; a < 127; a++) {
          Wire.beginTransmission(a);
          if (Wire.endTransmission() == 0) { Serial.printf(" 0x%02X", a); found++; }
        }
        Serial.println(found ? "" : " none");
      }
    }
    return;
  }
  if (now - lastMpuRead >= MPU_READ_INTERVAL_MS) {
    lastMpuRead = now;

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);  // ACCEL_XOUT_H
    if (Wire.endTransmission(false) == 0 && Wire.requestFrom(MPU_ADDR, (uint8_t)14) == 14) {
      int16_t raw[7];
      for (int i = 0; i < 7; i++) {
        uint8_t hi = Wire.read();
        uint8_t lo = Wire.read();
        raw[i] = (int16_t)((hi << 8) | lo);
      }
      mpuAx = raw[0] / 16384.0f;
      mpuAy = raw[1] / 16384.0f;
      mpuAz = raw[2] / 16384.0f;
      mpuGx = raw[4] / MPU_GYRO_LSB_PER_DPS - gyroBiasX;
      mpuGy = raw[5] / MPU_GYRO_LSB_PER_DPS - gyroBiasY;
      mpuGz = raw[6] / MPU_GYRO_LSB_PER_DPS - gyroBiasZ;

      unsigned long us = micros();
      float dt = (us - lastMpuMicros) / 1e6f;
      lastMpuMicros = us;
      if (dt <= 0 || dt > 0.5f) dt = MPU_READ_INTERVAL_MS / 1000.0f;

      float accRoll = atan2f(mpuAy, mpuAz) * 180.0f / M_PI;
      float accPitch = atan2f(-mpuAx, sqrtf(mpuAy * mpuAy + mpuAz * mpuAz)) * 180.0f / M_PI;
      if (fabsf(mpuGz) > MPU_YAW_DEADBAND_DPS) {
        mpuYaw += mpuGz * dt;
        if (mpuYaw > 180.0f) mpuYaw -= 360.0f;
        if (mpuYaw < -180.0f) mpuYaw += 360.0f;
      }
      mpuRoll = MPU_FILTER_ALPHA * (mpuRoll + mpuGx * dt) + (1 - MPU_FILTER_ALPHA) * accRoll;
      mpuPitch = MPU_FILTER_ALPHA * (mpuPitch + mpuGy * dt) + (1 - MPU_FILTER_ALPHA) * accPitch;
    }
  }
}

// Sends "yaw,pitch,roll,gx,gy,gz,ax,ay,az" (deg x3, deg/s x3, g x3) over BLE.
void notifyMpu() {
  unsigned long now = millis();
  if (now - lastMpuNotify < MPU_NOTIFY_INTERVAL_MS) return;
  lastMpuNotify = now;

  char buf[96];
  if (mpuOk) {
    snprintf(buf, sizeof(buf), "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f",
             mpuYaw, mpuPitch, mpuRoll, mpuGx, mpuGy, mpuGz, mpuAx, mpuAy, mpuAz);
  } else {
    snprintf(buf, sizeof(buf), "NA");
  }
  Serial.printf("MPU: %s\n", buf);

  if (deviceConnected && pMpuCharacteristic) {
    pMpuCharacteristic->setValue((uint8_t*)buf, strlen(buf));
    pMpuCharacteristic->notify();
  }
}

class ServerCallbacks: public BLEServerCallbacks {
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

  // Wheel PWM (1 kHz, 8-bit). Uses 4 of the 8 LEDC channels; the 4 servos use the other 4.
  for (int i = 0; i < 4; i++) {
    if (!ledcAttach(MOTOR_PINS[i], MOTOR_PWM_FREQ_HZ, MOTOR_PWM_BITS)) {
      Serial.printf("ERROR: could not attach motor pin %d\n", MOTOR_PINS[i]);
    }
  }
  motors(0, 0, 0, 0);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  mpuOk = mpuInit();
  lastMpuMicros = micros();
  Serial.println(mpuOk ? "MPU-6050 ready (keep the car still while it calibrates)" : "ERROR: MPU-6050 not found (check SDA/SCL/power)");

  // Servo PWM setup (50 Hz, direct LEDC). Servos must use an external 5V supply.
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (!ledcAttach(SERVO_PINS[i], SERVO_FREQ_HZ, SERVO_PWM_BITS)) {
      Serial.printf("ERROR: could not attach servo %d on pin %d\n", i, SERVO_PINS[i]);
    }
    writeServo(i, 90);
  }

  BLEDevice::init("AI_Car");
  BLEDevice::setMTU(185);  // MPU notifications are longer than the default 20 bytes

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

  // Distance (cm, ASCII text, "-1" = no echo) notified to the client.
  pDistanceCharacteristic = pService->createCharacteristic(
    DISTANCE_CHAR_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pDistanceCharacteristic->addDescriptor(new BLE2902());

  // MPU-6050 data, ASCII "pitch,roll,ax,ay,az,gx,gy,gz" ("NA" = sensor not found).
  pMpuCharacteristic = pService->createCharacteristic(
    MPU_CHAR_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pMpuCharacteristic->addDescriptor(new BLE2902());

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
  updateDistance();
  updateBuzzer();
  updateMpu();
  notifyMpu();
  delay(2);
}
