#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>

// ===========================
// Select camera model in board_config.h
// ===========================
#include "board_config.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid = "SEAL123";
const char *password = "12345678";

// ===========================
// Optional: Static IP configuration
// Set USE_STATIC_IP to true if you want a fixed IP (e.g. 192.168.137.30 for camera_test.py)
// ===========================
#define USE_STATIC_IP true  // Windows hotspot subnet is 192.168.137.x
IPAddress static_ip(192, 168, 137, 30);
IPAddress gateway(192, 168, 137, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress primaryDNS(8, 8, 8, 8);

void startCameraServer();
void setupLedFlash();

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
  Serial.println("--- Booting ESP32-CAM ---");

  if (psramFound()) {
    Serial.printf("PSRAM detected! Size: %d bytes\n", ESP.getPsramSize());
  } else {
    Serial.println("WARNING: PSRAM not detected or disabled! Falling back to internal DRAM.");
  }

  camera_config_t config = {};
  memset(&config, 0, sizeof(camera_config_t));
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  config.jpeg_buffer_size = 0;
  config.sccb_i2c_port = -1;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    // Limit the frame size when PSRAM is not available to prevent DMA malloc failure
    config.frame_size = FRAMESIZE_VGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
    config.jpeg_quality = 12;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }
  Serial.println("Camera init succeeded!");

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }
  // drop down frame size for higher initial frame rate
  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_QVGA);  // 320x240. VGA gave ~1 FPS and quality 10 gave ~7 FPS on this WiFi link
    s->set_quality(s, 18);                // 18 gave ~45 FPS; lower number = better quality but bigger frames
    s->set_aec2(s, 1);                   // better auto-exposure in dim light
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED FLash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif

  if (USE_STATIC_IP) {
    if (!WiFi.config(static_ip, gateway, subnet, primaryDNS)) {
      Serial.println("Static IP configuration failed, using DHCP");
    }
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, strlen(password) ? password : nullptr);
  WiFi.setSleep(false);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
  Serial.print("Stream URL: 'http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream'");
}

extern volatile uint32_t diag_frames, diag_bytes, diag_capture_us, diag_send_us, diag_max_send_us, diag_capture_fail, diag_send_fail;

void loop() {
  // Diagnostics: one line per second so you can see where the bottleneck is
  delay(1000);

  uint32_t frames = diag_frames, bytes = diag_bytes;
  uint32_t cap = diag_capture_us, snd = diag_send_us, maxsnd = diag_max_send_us;
  uint32_t capFail = diag_capture_fail, sndFail = diag_send_fail;
  diag_frames = diag_bytes = diag_capture_us = diag_send_us = diag_max_send_us = 0;

  bool up = WiFi.status() == WL_CONNECTED;
  Serial.printf("[DIAG] wifi=%s rssi=%ddBm ch=%d | fps=%u size=%uB/frame throughput=%.0fkbps | capture=%.1fms send=%.1fms maxsend=%.0fms | fails cap=%u send=%u | heap=%u psram=%u uptime=%lus\n",
                up ? "UP" : "DOWN", up ? WiFi.RSSI() : 0, up ? WiFi.channel() : 0,
                frames, frames ? bytes / frames : 0, bytes * 8.0 / 1000.0,
                frames ? cap / 1000.0 / frames : 0.0, frames ? snd / 1000.0 / frames : 0.0, maxsnd / 1000.0,
                capFail, sndFail, ESP.getFreeHeap(), ESP.getFreePsram(), millis() / 1000);

  if (!up) {
    Serial.println("[DIAG] WiFi lost, reconnecting...");
    WiFi.reconnect();
  }
}
