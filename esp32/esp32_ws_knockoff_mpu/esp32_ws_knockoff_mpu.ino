// esp32_ws_knockoff_mpu.ino
#include <WiFi.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <ArduinoWebsockets.h>

using namespace websockets;

// ===== CONFIG =====
const char* ssid = "ZTE_2.4G_BFS";
const char* password = "uMHUQaQP";
const char* ws_server = "ws://192.168.1.27:8765/esp";  // connect to /esp

// ===== MPU (clone-safe) =====
#define MPU 0x68

uint8_t safeRead8(uint8_t reg) {
  Wire.beginTransmission(MPU);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, (uint8_t)2);
  uint8_t v = Wire.read();
  if (Wire.available()) Wire.read();
  return v;
}

int16_t safeRead16(uint8_t reg) {
  Wire.beginTransmission(MPU);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, (uint8_t)6);
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  return (int16_t)((hi << 8) | lo);
}

WebsocketsClient client;

void init_mpu() {
  Wire.begin(21, 22);
  Wire.setClock(100000);
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission();
  delay(100);
}

void setup() {
  Serial.begin(115200);
  init_mpu();

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi: " + WiFi.localIP().toString());

  // connect websocket
  connectWS();
}

void connectWS() {
  Serial.println("Connecting to WS...");
  client.onMessage([](WebsocketsMessage message) {
    Serial.print("WS msg: ");
    Serial.println(message.data());
  });

  client.onEvent([](WebsocketsEvent event, String data) {
    if (event == WebsocketsEvent::ConnectionOpened) {
      Serial.println("WS connected");
    } else if (event == WebsocketsEvent::ConnectionClosed) {
      Serial.println("WS disconnected");
    } else if (event == WebsocketsEvent::GotPing) {
      Serial.println("Ping");
    } else if (event == WebsocketsEvent::GotPong) {
      Serial.println("Pong");
    }
  });

  // try to connect; returns true/false
  bool ok = client.connect(ws_server);
  if (!ok) {
    Serial.println("WS connect failed");
  } else {
    Serial.println("WS connect OK");
  }
}

unsigned long lastSendMillis = 0;
const unsigned long SEND_INTERVAL = 50; // ms (~20Hz)
unsigned long lastReconnectAttempt = 0;
const unsigned long RECONNECT_DELAY = 2000; // ms

void loop() {
  // keep the websocket library processing incoming frames / heartbeats
  client.poll();

  // if not connected: attempt reconnect occasionally
  if (!client.available()) {
    unsigned long now = millis();
    if (WiFi.status() == WL_CONNECTED) {
      if (now - lastReconnectAttempt > RECONNECT_DELAY) {
        lastReconnectAttempt = now;
        Serial.println("Attempting WS reconnect...");
        connectWS();
      }
    } else {
      // if wifi dropped, try to reconnect wifi
      Serial.println("WiFi lost, reconnecting...");
      WiFi.reconnect();
    }
  }

  // send at fixed rate
  if (millis() - lastSendMillis >= SEND_INTERVAL) {
    lastSendMillis = millis();

    // Read IMU (clone-safe)
    int16_t ax = safeRead16(0x3B);
    int16_t ay = safeRead16(0x3D);
    int16_t az = safeRead16(0x3F);
    int16_t gx = safeRead16(0x43);
    int16_t gy = safeRead16(0x45);
    int16_t gz = safeRead16(0x47);

    // Build JSON object
    StaticJsonDocument<256> doc;
    doc["timestamp_ms"] = millis();
    doc["ax"] = ax;
    doc["ay"] = ay;
    doc["az"] = az;
    doc["gx"] = gx;
    doc["gy"] = gy;
    doc["gz"] = gz;

    String out;
    serializeJson(doc, out);

    bool sent = false;
    if (client.available()) {
      // send text frame (String ensures text)
      sent = client.send(out);
    }

    if (!sent) {
      // fallback: print to serial for debugging
      Serial.print("NOT SENT: ");
      Serial.println(out);
    } else {
      Serial.print("SENT: ");
      Serial.println(out);
    }
  }

  // small poll allow other network tasks
  client.poll();
}
