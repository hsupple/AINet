/*
  AINet ESP32 INMP441 → :1111

  Pins: SD 22, L/R 23, SCK 19, WS 21
  Fill WiFi + PC IP in config.h (or config.local.h), then flash.
*/

#include <WiFi.h>
#include "config.h"

#if __has_include(<ESP_I2S.h>)
#include <ESP_I2S.h>
#define AINET_I2S_NEW 1
#else
#include <driver/i2s.h>
#define AINET_I2S_NEW 0
#endif

static const int PIN_SD = 22;
static const int PIN_LR = 23;
static const int PIN_SCK = 19;
static const int PIN_WS = 21;
static const uint32_t SAMPLE_RATE = 16000;
static const int PING_MS = 4000;

#if AINET_I2S_NEW
I2SClass gMic;
#endif

static bool wifiReady() {
  return WiFi.status() == WL_CONNECTED;
}

static void connectWifi() {
  if (wifiReady()) {
    return;
  }
  Serial.printf("WiFi connecting to %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (!wifiReady() && millis() - start < 20000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  if (wifiReady()) {
    Serial.printf("WiFi %s  rssi %d\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else {
    Serial.println("WiFi failed");
  }
}

static bool pingPong() {
  WiFiClient c;
  c.setTimeout(3000);
  if (!c.connect(AINET_HOST, AINET_PORT)) {
    Serial.printf("ping connect fail %s:%u\n", AINET_HOST, (unsigned)AINET_PORT);
    return false;
  }
  c.printf(
      "GET /api/esp32/ping HTTP/1.1\r\nHost: %s:%u\r\nConnection: close\r\n\r\n",
      AINET_HOST,
      (unsigned)AINET_PORT);
  String body;
  bool headersDone = false;
  unsigned long start = millis();
  while (millis() - start < 3000 && (c.connected() || c.available())) {
    if (!c.available()) {
      delay(5);
      continue;
    }
    String line = c.readStringUntil('\n');
    if (!headersDone) {
      if (line == "\r" || line.length() <= 1) {
        headersDone = true;
      }
      continue;
    }
    body += line;
  }
  c.stop();
  bool ok = body.indexOf("pong") >= 0;
  Serial.println(ok ? "pong" : "ping: no pong");
  return ok;
}

static bool openAudio(WiFiClient &c) {
  c.setTimeout(8000);
  if (!c.connect(AINET_HOST, AINET_PORT)) {
    Serial.println("audio connect fail");
    return false;
  }
  c.printf(
      "POST /api/esp32/audio HTTP/1.1\r\n"
      "Host: %s:%u\r\n"
      "Transfer-Encoding: chunked\r\n"
      "Content-Type: application/octet-stream\r\n"
      "X-Sample-Rate: %u\r\n"
      "X-Bits: 16\r\n"
      "X-Channels: 1\r\n"
      "Connection: close\r\n"
      "\r\n",
      AINET_HOST,
      (unsigned)AINET_PORT,
      (unsigned)SAMPLE_RATE);
  return c.connected();
}

static bool sendChunk(WiFiClient &c, const uint8_t *data, size_t n) {
  if (!c.connected() || n == 0) {
    return false;
  }
  c.printf("%x\r\n", (unsigned)n);
  if (c.write(data, n) != n) {
    return false;
  }
  c.print("\r\n");
  return c.connected();
}

static bool micBegin() {
  pinMode(PIN_LR, OUTPUT);
  digitalWrite(PIN_LR, LOW);
#if AINET_I2S_NEW
  gMic.setPins(PIN_SCK, PIN_WS, -1, PIN_SD);
  if (!gMic.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("I2S begin failed");
    return false;
  }
  return true;
#else
  i2s_config_t cfg = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
#if defined(I2S_COMM_FORMAT_STAND_I2S)
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#else
      .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S),
#endif
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};
  i2s_pin_config_t pins = {};
  pins.bck_io_num = PIN_SCK;
  pins.ws_io_num = PIN_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_SD;
  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) {
    Serial.println("I2S install failed");
    return false;
  }
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) {
    Serial.println("I2S pins failed");
    return false;
  }
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
#endif
}

static size_t micReadPcm16(int16_t *out, size_t maxSamples) {
#if AINET_I2S_NEW
  int32_t raw[256];
  size_t want = sizeof(raw);
  if (maxSamples < 128) {
    want = maxSamples * 8;
  }
  size_t n = gMic.readBytes(reinterpret_cast<char *>(raw), want);
  size_t frames = n / 8;
  if (frames > maxSamples) {
    frames = maxSamples;
  }
  for (size_t i = 0; i < frames; i++) {
    int32_t v = raw[i * 2] >> 11;
    if (v > 32767) {
      v = 32767;
    }
    if (v < -32768) {
      v = -32768;
    }
    out[i] = static_cast<int16_t>(v);
  }
  return frames;
#else
  int32_t raw[256];
  size_t n = 0;
  i2s_read(I2S_NUM_0, raw, sizeof(raw), &n, portMAX_DELAY);
  size_t samples = n / 4;
  if (samples > maxSamples) {
    samples = maxSamples;
  }
  for (size_t i = 0; i < samples; i++) {
    int32_t v = raw[i] >> 11;
    if (v > 32767) {
      v = 32767;
    }
    if (v < -32768) {
      v = -32768;
    }
    out[i] = static_cast<int16_t>(v);
  }
  return samples;
#endif
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("AINet INMP441");
  if (String(WIFI_SSID) == "SET_ME" || String(AINET_HOST) == "SET_ME") {
    Serial.println("Edit config.h: WIFI_SSID, WIFI_PASS, AINET_HOST");
  }
  if (!micBegin()) {
    while (true) {
      delay(1000);
    }
  }
  connectWifi();
}

void loop() {
  if (!wifiReady()) {
    connectWifi();
    delay(500);
    return;
  }

  if (!pingPong()) {
    delay(PING_MS);
    return;
  }

  WiFiClient audio;
  if (!openAudio(audio)) {
    delay(500);
    return;
  }
  Serial.println("audio stream open");

  int16_t pcm[256];
  while (audio.connected()) {
    size_t n = micReadPcm16(pcm, 256);
    if (n == 0) {
      delay(1);
      continue;
    }
    if (!sendChunk(audio, reinterpret_cast<uint8_t *>(pcm), n * 2)) {
      break;
    }
  }
  audio.stop();
  Serial.println("audio stream closed");
  delay(250);
}
