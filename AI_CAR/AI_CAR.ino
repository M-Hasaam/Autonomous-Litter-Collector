
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

BLEServer* pServer = nullptr;
BLECharacteristic* pCharacteristic = nullptr;

bool deviceConnected = false;

const int IN1 = 15;
const int IN2 = 16;
const int IN3 = 17;
const int IN4 = 18;

// Left and Right motor directions
const int LU = IN2;
const int LD = IN1;
const int RU = IN3;
const int RD = IN4;

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
    String value = characteristic->getValue().c_str();

    for (int i = 0; i < value.length(); i++) {
      char command = value[i];
      Serial.println(command);

      switch (command) {
        case 'U':
          motors(1, 0, 1, 0);
          break;

        case 'D':
          motors(0, 1, 0, 1);
          break;

        case 'L':
          motors(0, 1, 1, 0);
          break;

        case 'R':
          motors(1, 0, 0, 1);
          break;

        case 'S':
          motors(0, 0, 0, 0);
          break;
      }
    }
  }
};

void setup() {
  Serial.begin(115200);

  pinMode(LU, OUTPUT);
  pinMode(LD, OUTPUT);
  pinMode(RU, OUTPUT);
  pinMode(RD, OUTPUT);

  motors(0, 0, 0, 0);

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
}

void loop() {
  delay(10);
}
