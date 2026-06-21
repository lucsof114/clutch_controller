// icm_bitbang_test — SPLIT SCK/MOSI: two banks of 3 IMUs, each bank its own SCK/MOSI.
// MISO individual per IMU. Bit-banged SPI mode 3. Halves the stub load on each
// clock/data line and isolates the two groups.
//
// Bank A (IMU0-2): SCK 13, MOSI 11.   Bank B (IMU3-5): SCK 23, MOSI 12.

#include <Arduino.h>

struct Imu { uint8_t cs; uint8_t miso; uint8_t intp; uint8_t sck; uint8_t mosi; };
// CS, MISO, INT1, SCK, MOSI
Imu IMUS[] = {
  {2, 19, 8,  13, 11}, {3, 18, 9,  13, 11}, {4, 17, 10, 13, 11},   // bank A: SCK13 MOSI11
  {5, 16, 20, 23, 12}, {6, 15, 21, 23, 12}, {7, 14, 22, 23, 12},   // bank B: SCK23 MOSI12
  {35, 36, 37, 33, 34},                                            // bank C: SCK33 MOSI34
};
const uint8_t N = sizeof(IMUS) / sizeof(IMUS[0]);

const uint8_t WHO_AM_I=0x75, ICM_WHOAMI=0x42, DEVICE_CONFIG=0x11, PWR_MGMT0=0x4E,
              GYRO_CONFIG0=0x4F, ACCEL_CONFIG0=0x50, ACCEL_DATA=0x1F, REG_BANK_SEL=0x76;

bool present[8]; uint32_t cnt[8];

inline void half() { delayNanoseconds(1500); }   // ~330 kHz

// SPI mode 3 (SCK idle HIGH). Set MOSI, pulse SCK low->high, sample MISO at END of high phase.
uint8_t bbByte(uint8_t out, uint8_t miso, uint8_t sck, uint8_t mosi) {
  uint8_t in = 0;
  for (int i = 7; i >= 0; i--) {
    digitalWriteFast(mosi, (out >> i) & 1);
    digitalWriteFast(sck, LOW);  half();
    digitalWriteFast(sck, HIGH); half();
    in = (in << 1) | (digitalReadFast(miso) & 1);
  }
  return in;
}

void wr(Imu& m, uint8_t reg, uint8_t val) {
  digitalWriteFast(m.cs, LOW); bbByte(reg & 0x7F, m.miso, m.sck, m.mosi); bbByte(val, m.miso, m.sck, m.mosi); digitalWriteFast(m.cs, HIGH);
}
uint8_t rd(Imu& m, uint8_t reg) {
  digitalWriteFast(m.cs, LOW); bbByte(reg | 0x80, m.miso, m.sck, m.mosi); uint8_t v = bbByte(0, m.miso, m.sck, m.mosi); digitalWriteFast(m.cs, HIGH); return v;
}
void rdBurst(Imu& m, uint8_t reg, uint8_t* b, int n) {
  digitalWriteFast(m.cs, LOW); bbByte(reg | 0x80, m.miso, m.sck, m.mosi); for (int i = 0; i < n; i++) b[i] = bbByte(0, m.miso, m.sck, m.mosi); digitalWriteFast(m.cs, HIGH);
}

bool cfg(Imu& m) {
  wr(m, REG_BANK_SEL, 0);
  wr(m, DEVICE_CONFIG, 0x01); delay(50);
  uint8_t who = 0; int good = 0;
  for (int i = 0; i < 40 && good < 2; i++) { who = rd(m, WHO_AM_I); good = (who == ICM_WHOAMI) ? good + 1 : 0; delay(3); }
  Serial.print("# CS"); Serial.print(m.cs); Serial.print(" MISO"); Serial.print(m.miso);
  Serial.print(" SCK"); Serial.print(m.sck); Serial.print(" WHOAMI=0x"); Serial.println(who, HEX);
  if (who != ICM_WHOAMI) return false;
  wr(m, GYRO_CONFIG0, 0x0F); wr(m, ACCEL_CONFIG0, 0x0F); wr(m, PWR_MGMT0, 0x0F); delay(45);
  return true;
}

void setup() {
  Serial.begin(2000000);
  uint32_t t0 = millis(); while (!Serial && millis() - t0 < 2000) {}
  // init every SCK (idle high) and MOSI used, plus CS high, MISO pullup
  for (uint8_t i = 0; i < N; i++) {
    pinMode(IMUS[i].sck, OUTPUT);  digitalWriteFast(IMUS[i].sck, HIGH);
    pinMode(IMUS[i].mosi, OUTPUT);
    pinMode(IMUS[i].cs, OUTPUT);   digitalWriteFast(IMUS[i].cs, HIGH);
    pinMode(IMUS[i].miso, INPUT_PULLUP);
  }
  delay(10);
  uint8_t p = 0;
  for (uint8_t i = 0; i < N; i++) { present[i] = cfg(IMUS[i]); if (present[i]) p++; }
  Serial.print("# split-bus ready: "); Serial.print(p); Serial.print("/"); Serial.print(N); Serial.println(" present");
}

void loop() {
  static uint32_t last = 0;
  for (uint8_t i = 0; i < N; i++) { if (present[i]) { uint8_t b[6]; rdBurst(IMUS[i], ACCEL_DATA, b, 6); cnt[i]++; } }
  if (millis() - last >= 1000) {
    last = millis();
    for (uint8_t i = 0; i < N; i++) {
      Serial.print("IMU"); Serial.print(i); Serial.print("(CS"); Serial.print(IMUS[i].cs); Serial.print("): ");
      if (!present[i]) { Serial.print("ABSENT   "); continue; }
      uint8_t b[6]; rdBurst(IMUS[i], ACCEL_DATA, b, 6);
      int16_t ax=(b[0]<<8)|b[1], ay=(b[2]<<8)|b[3], az=(b[4]<<8)|b[5];
      float mag = sqrtf((ax/2048.0f)*(ax/2048.0f)+(ay/2048.0f)*(ay/2048.0f)+(az/2048.0f)*(az/2048.0f));
      Serial.print(cnt[i]); Serial.print(" rd/s |a|="); Serial.print(mag, 2); Serial.print("g   ");
      cnt[i] = 0;
    }
    Serial.println();
  }
}
