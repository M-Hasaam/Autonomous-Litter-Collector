// ESP32 (WROOM-32 dev board) WiFi bridge for the ESP32-CAM wired stream.
//
//   ESP32-CAM  U0T (GPIO1) ----> bridge GPIO34 (input-only pin, "G34")      common GND required
//   (camera firmware: USE_SERIAL_STREAM 1, 921600 baud)
//
// The bridge creates its own WiFi network. Join it from the laptop and run:
//   py camera_serial.py tcp:192.168.4.1:8080
//
// Frame format from the camera: A5 5A C3 3C | uint32 length (little-endian) | JPEG bytes.
// The bridge re-assembles whole frames, drops any frame that is already out of date
// (a newer one is waiting), and forwards the rest unchanged.
#include <WiFi.h>

const char *AP_SSID = "ESP32-BRIDGE";
const char *AP_PASS = "12345678";  // at least 8 characters
#define AP_CHANNEL 1
#define CAM_BAUD 3000000
#define RX_PIN 34  // connect to ESP32-CAM U0T (GPIO34 is input-only, which is fine for receiving)
#define TX_PIN 17  // not used
#define MAX_FRAME 60000
#define TCP_PORT 8080
#define MAGIC 0xA55AC33Cu

WiFiServer server(TCP_PORT);
WiFiClient client;
static uint8_t frameBuf[MAX_FRAME];
static uint8_t rxTmp[2048];

enum State { S_SYNC, S_LEN, S_BODY };
static State st = S_SYNC;
static uint32_t win = 0;
static uint8_t lenBytes[4];
static int lenGot = 0;
static uint32_t need = 0, got = 0;
static uint32_t lastByteMs = 0;

uint32_t framesIn = 0, framesSent = 0, framesDropped = 0, framesBad = 0, rawBytes = 0;
uint32_t lastStats = 0;
volatile uint32_t uartErr[8] = {0};
static uint8_t snap[24];  // first bytes received in each 2 s window, to see what is on the wire
static int snapN = 0;

static void resetParser() {
  st = S_SYNC;
  win = 0;
  lenGot = 0;
  got = 0;
}

static uint32_t curBaud = CAM_BAUD;
static int rxPin = RX_PIN;
static uint32_t lastGoodMs = 0;  // last time a valid frame arrived
static uint32_t lastScanMs = 0;

// The UART hardware FIFO holds 128 bytes and the core raises the "FIFO full" interrupt at 120, leaving only
// 8 bytes (~27 us at 3 Mbaud) for the ISR to react. Raising the interrupt earlier gives the ISR much more time.
#define RX_FIFO_FULL_BYTES 32  // headroom 96 bytes = ~320 us at 3 Mbaud

static void startCamUart(int pin) {
  Serial2.end();
  Serial2.setRxBufferSize(32768);
  Serial2.begin(curBaud, SERIAL_8N1, pin, TX_PIN);
  Serial2.setRxFIFOFull(RX_FIFO_FULL_BYTES);  // must be set after begin(): end() resets it
  Serial2.onReceiveError([](hardwareSerial_error_t e) {
    if ((int)e < 8) uartErr[(int)e]++;
  });
  rxPin = pin;
  resetParser();
}

// Listen on every free GPIO for the camera's data stream and return the pin that is toggling the most.
// A UART stream at 460800 baud changes level thousands of times in 30 ms; a floating or idle pin barely moves.
static int scanPins() {
  static const int cand[] = {0, 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33, 34, 35, 36, 39};
  Serial2.end();
  int bestPin = -1;
  uint32_t best = 0;
  String line = "[SCAN]";
  for (int p : cand) {
    pinMode(p, INPUT_PULLUP);
    delay(2);
    uint32_t edges = 0;
    int last = digitalRead(p);
    uint32_t t0 = micros();
    while (micros() - t0 < 30000) {
      int v = digitalRead(p);
      if (v != last) {
        edges++;
        last = v;
      }
    }
    if (edges > 50) line += " G" + String(p) + "=" + String(edges);
    if (edges > best) {
      best = edges;
      bestPin = p;
    }
  }
  Serial.println(line);
  if (best < 2000) {
    Serial.println("[SCAN] no pin shows a camera data stream");
    return -1;
  }
  Serial.printf("[SCAN] camera data found on G%d (%u edges in 30 ms) -> locking to it\n", bestPin, best);
  return bestPin;
}

static void finishFrame() {
  if (frameBuf[0] != 0xFF || frameBuf[1] != 0xD8 || frameBuf[need - 2] != 0xFF || frameBuf[need - 1] != 0xD9) {
    framesBad++;
    return;
  }
  lastGoodMs = millis();
  framesIn++;
  if (!(client && client.connected())) return;

  // If another whole frame is already waiting, this one is stale: skip it to stay low-latency.
  if (Serial2.available() > (int)need) {
    framesDropped++;
    return;
  }
  const uint8_t hdr[8] = {0xA5, 0x5A, 0xC3, 0x3C, lenBytes[0], lenBytes[1], lenBytes[2], lenBytes[3]};
  client.write(hdr, sizeof(hdr));
  client.write(frameBuf, need);
  framesSent++;
}

// Process a block of received bytes entirely from memory.
static void feed(const uint8_t *p, size_t n) {
  size_t i = 0;
  while (i < n) {
    switch (st) {
      case S_SYNC:
        win = (win << 8) | p[i++];
        if (win == MAGIC) {
          st = S_LEN;
          lenGot = 0;
        }
        break;
      case S_LEN:
        lenBytes[lenGot++] = p[i++];
        if (lenGot == 4) {
          need = lenBytes[0] | (lenBytes[1] << 8) | (lenBytes[2] << 16) | ((uint32_t)lenBytes[3] << 24);
          if (need < 4 || need > MAX_FRAME) {
            framesBad++;
            resetParser();
          } else {
            got = 0;
            st = S_BODY;
          }
        }
        break;
      case S_BODY: {
        size_t c = min((size_t)(n - i), (size_t)(need - got));
        memcpy(frameBuf + got, p + i, c);
        got += c;
        i += c;
        if (got == need) {
          finishFrame();
          resetParser();
        }
        break;
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("--- ESP32 WiFi bridge ---");

  startCamUart(RX_PIN);

  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), IPAddress(255, 255, 255, 0));
  WiFi.softAP(AP_SSID, AP_PASS, AP_CHANNEL, 0, 2);
  WiFi.setSleep(false);
  server.begin();
  server.setNoDelay(true);

  Serial.printf("WiFi '%s' (password %s), connect to %s:%d\n", AP_SSID, AP_PASS, WiFi.softAPIP().toString().c_str(), TCP_PORT);
}

// Type "b460800" + Enter in a serial terminal (or send it from a script) to change the camera link speed on the fly.
static void handleCommands() {
  static char cmd[16];
  static int n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      cmd[n] = 0;
      if (n > 1 && cmd[0] == 'b') {
        uint32_t baud = strtoul(cmd + 1, nullptr, 10);
        if (baud >= 9600 && baud <= 3000000) {
          curBaud = baud;
          startCamUart(rxPin);
          Serial.printf("[CMD] camera link now %u baud\n", baud);
        }
      } else if (n > 1 && cmd[0] == 'p') {
        int pin = atoi(cmd + 1);
        if (pin >= 0 && pin <= 39) {
          startCamUart(pin);
          Serial.printf("[CMD] listening on G%d\n", pin);
        }
      }
      n = 0;
    } else if (n < (int)sizeof(cmd) - 1) {
      cmd[n++] = c;
    }
  }
}

void loop() {
  handleCommands();

  // No valid frame for a while: look for the pin that really carries the camera stream and lock to it.
  if (millis() - lastGoodMs > 6000 && millis() - lastScanMs > 6000) {
    lastScanMs = millis();
    int p = scanPins();
    startCamUart(p >= 0 ? p : rxPin);
  }
  if (!client || !client.connected()) {
    WiFiClient c = server.available();
    if (c) {
      client = c;
      client.setNoDelay(true);
      Serial.println("viewer connected");
    }
  }

  int avail = Serial2.available();
  if (avail > 0) {
    size_t n = Serial2.read(rxTmp, min((size_t)avail, sizeof(rxTmp)));
    if (n > 0) {
      lastByteMs = millis();
      rawBytes += n;
      for (size_t k = 0; k < n && snapN < (int)sizeof(snap); k++) snap[snapN++] = rxTmp[k];
      feed(rxTmp, n);
    }
  } else {
    // Drop a half-received frame if the camera went quiet
    if (st != S_SYNC && millis() - lastByteMs > 300) {
      framesBad++;
      resetParser();
    }
    delay(1);  // let WiFi and other tasks run
  }

  if (millis() - lastStats > 2000) {
    Serial.printf("[BRIDGE] raw=%u in=%u sent=%u dropped=%u bad=%u viewer=%s clients_on_ap=%d heap=%u\n", rawBytes, framesIn, framesSent, framesDropped, framesBad,
                  (client && client.connected()) ? "yes" : "no", WiFi.softAPgetStationNum(), ESP.getFreeHeap());
    Serial.printf("[UART] errors none/data_break/parity/fifo_ovf/frame/buf_ovf = %u/%u/%u/%u/%u/%u  first bytes:", uartErr[0], uartErr[1], uartErr[2],
                  uartErr[3], uartErr[4], uartErr[5]);
    for (int k = 0; k < snapN; k++) Serial.printf(" %02X", snap[k]);
    Serial.println();
    snapN = 0;
    for (int k = 0; k < 8; k++) uartErr[k] = 0;
    framesIn = framesSent = framesDropped = framesBad = rawBytes = 0;
    lastStats = millis();
  }
}
