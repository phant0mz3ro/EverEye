/*
 * ESP32-CAM Security System — Step 1: MJPEG Streaming
 * Board: AI-Thinker ESP32-CAM
 *
 * Confirms camera init + WiFi + streaming work before adding SD/PIR/face logic.
 * View stream at: http://<device-ip>/stream
 */

#include "esp_camera.h"
#include <WiFi.h>

// ===================
// WiFi credentials
// ===================
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASSWORD";

// ===================
// AI-Thinker pin map
// ===================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#include "app_httpd.h"  // streaming server logic lives here (next tool call)

void startCameraServer(); // defined in app_httpd.h

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Use higher resolution/quality only if PSRAM is present — AI-Thinker has PSRAM.
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;    // 640x480 — SVGA looks nicer but adds real latency over wifi
    config.jpeg_quality = 12;             // lower number = higher quality, more bandwidth/latency
    config.fb_count = 2;                  // 2 buffers lets grab_mode below skip stale frames
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST; // KEY FIX: always serve the newest frame, drop backlog
  } else {
    config.frame_size = FRAMESIZE_QVGA;   // no PSRAM = very limited buffer headroom, go small
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // Optional: flip/mirror if your module is mounted upside down
  sensor_t* s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // avoid WiFi power-save causing stream stutter — biggest single latency win after grab_mode
  WiFi.setTxPower(WIFI_POWER_19_5dBm); // max tx power, helps if router isn't right next to the board
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();

  Serial.print("Camera ready. Stream at: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
}

void loop() {
  delay(10000); // server runs on its own task; nothing needed here yet
}
