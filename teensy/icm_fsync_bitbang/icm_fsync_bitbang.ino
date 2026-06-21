// icm_fsync_bitbang — FSYNC check on the split-bus, individual-MISO, bit-bang setup.
// Teensy self-generates a 30 Hz sync on pin 32 (wired to every IMU's FSYNC/INT2).
// Each IMU is configured with FSYNC + FIFO; an FSYNC-tagged sample shows up as a
// FIFO packet with header 0x6C (vs 0x68 for a normal ODR-timestamped packet).
// Reports per-IMU: total packets/s and FSYNC-tagged/s (expect ~30/s).
//
// Bank A (IMU0-2): SCK 13, MOSI 11.  Bank B (IMU3-5): SCK 23, MOSI 12.

#include <Arduino.h>
#include <IntervalTimer.h>

struct Imu { uint8_t cs; uint8_t miso; uint8_t intp; uint8_t sck; uint8_t mosi; };
Imu IMUS[] = {
  {2, 19, 8,  13, 11}, {3, 18, 9,  13, 11}, {4, 17, 10, 13, 11},   // bank A
  {5, 16, 20, 23, 12}, {6, 15, 21, 23, 12}, {7, 14, 22, 23, 12},   // bank B
  {35, 36, 37, 33, 34},                                            // bank C
};
const uint8_t N = sizeof(IMUS) / sizeof(IMUS[0]);

// registers
const uint8_t WHO_AM_I=0x75, ICM_WHOAMI=0x42, DEVICE_CONFIG=0x11, REG_BANK_SEL=0x76,
  INT_CONFIG1=0x64, PWR_MGMT0=0x4E, GYRO_CONFIG0=0x4F, ACCEL_CONFIG0=0x50,
  INTF_CONFIG0=0x4C, TMST_CONFIG=0x54, INTF_CONFIG5=0x7B, FSYNC_CONFIG=0x62,
  FIFO_CONFIG=0x16, FIFO_CONFIG1=0x5F, SIGNAL_PATH_RESET=0x4B,
  FIFO_COUNTH=0x2E, FIFO_DATA=0x30;

const uint8_t PIN_SYNC = 32;
IntervalTimer syncTimer; volatile bool syncLvl = false;
void syncToggle() { syncLvl = !syncLvl; digitalWriteFast(PIN_SYNC, syncLvl); }  // 60 Hz toggle -> 30 Hz

bool present[8]; uint32_t pkt[8], tag[8];

inline void half() { delayNanoseconds(1500); }

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
void wr(Imu& m, uint8_t r, uint8_t v) { digitalWriteFast(m.cs,LOW); bbByte(r&0x7F,m.miso,m.sck,m.mosi); bbByte(v,m.miso,m.sck,m.mosi); digitalWriteFast(m.cs,HIGH); }
uint8_t rd(Imu& m, uint8_t r) { digitalWriteFast(m.cs,LOW); bbByte(r|0x80,m.miso,m.sck,m.mosi); uint8_t v=bbByte(0,m.miso,m.sck,m.mosi); digitalWriteFast(m.cs,HIGH); return v; }
void rdBurst(Imu& m, uint8_t r, uint8_t* b, int n) { digitalWriteFast(m.cs,LOW); bbByte(r|0x80,m.miso,m.sck,m.mosi); for(int i=0;i<n;i++) b[i]=bbByte(0,m.miso,m.sck,m.mosi); digitalWriteFast(m.cs,HIGH); }

bool cfg(Imu& m) {
  wr(m, REG_BANK_SEL, 0);
  wr(m, DEVICE_CONFIG, 0x01); delay(50);
  uint8_t who=0; int good=0;
  for (int i=0;i<40 && good<2;i++){ who=rd(m,WHO_AM_I); good=(who==ICM_WHOAMI)?good+1:0; delay(3); }
  Serial.print("# CS"); Serial.print(m.cs); Serial.print(" WHOAMI=0x"); Serial.println(who,HEX);
  if (who != ICM_WHOAMI) return false;
  wr(m, INT_CONFIG1, 0x00);
  wr(m, GYRO_CONFIG0, 0x0F); wr(m, ACCEL_CONFIG0, 0x0F); wr(m, PWR_MGMT0, 0x0F); delay(45);
  wr(m, INTF_CONFIG0, 0x30);          // count in bytes, big-endian
  wr(m, TMST_CONFIG,  0x23);          // TMST + FSYNC enabled (reset default — do NOT force low)
  wr(m, REG_BANK_SEL, 1);
  wr(m, INTF_CONFIG5, 0x02);          // PIN9 = FSYNC
  wr(m, REG_BANK_SEL, 0);
  wr(m, FSYNC_CONFIG, 0x10);          // FSYNC_UI_SEL=1, rising
  wr(m, FIFO_CONFIG1, 0x0F);          // accel+gyro+temp+tmst_fsync -> 16-byte packets
  wr(m, FIFO_CONFIG,  0x00); delay(2);
  wr(m, FIFO_CONFIG,  0x40);          // stream-to-FIFO
  wr(m, SIGNAL_PATH_RESET, 0x02); delay(2);   // FIFO flush
  return true;
}

void drain(uint8_t i) {
  Imu& m = IMUS[i];
  uint8_t c[2]; rdBurst(m, FIFO_COUNTH, c, 2);
  uint16_t cnt = (c[0] << 8) | c[1];
  if (cnt < 16) return;
  if (cnt > 256) cnt = 256;            // cap per pass (this is a check, not a logger)
  uint16_t whole = (cnt / 16) * 16;
  uint8_t buf[256];
  rdBurst(m, FIFO_DATA, buf, whole);
  for (uint16_t o = 0; o + 16 <= whole; o += 16) {
    uint8_t h = buf[o];
    if ((h & 0xF8) != 0x68) return;    // bad header -> stop (desync); flush next pass
    pkt[i]++;
    if (h & 0x04) tag[i]++;            // 0x6C = FSYNC-tagged
  }
}

void setup() {
  Serial.begin(2000000);
  uint32_t t0=millis(); while(!Serial && millis()-t0<2000){}
  for (uint8_t i=0;i<N;i++){
    pinMode(IMUS[i].sck,OUTPUT); digitalWriteFast(IMUS[i].sck,HIGH);
    pinMode(IMUS[i].mosi,OUTPUT);
    pinMode(IMUS[i].cs,OUTPUT); digitalWriteFast(IMUS[i].cs,HIGH);
    pinMode(IMUS[i].miso,INPUT_PULLUP);
  }
  pinMode(PIN_SYNC, OUTPUT); digitalWriteFast(PIN_SYNC, LOW);
  delay(10);
  uint8_t p=0;
  for (uint8_t i=0;i<N;i++){ present[i]=cfg(IMUS[i]); if(present[i]) p++; }
  Serial.print("# fsync check ready: "); Serial.print(p); Serial.print("/"); Serial.print(N);
  Serial.println(" present — 30 Hz sync on pin 32");
  syncTimer.begin(syncToggle, 1000000.0/60.0);   // 60 Hz toggle -> 30 Hz square
}

void loop() {
  static uint32_t last=0;
  for (uint8_t i=0;i<N;i++) if (present[i]) drain(i);
  if (millis()-last >= 1000) {
    last=millis();
    for (uint8_t i=0;i<N;i++){
      Serial.print("IMU"); Serial.print(i); Serial.print("(CS"); Serial.print(IMUS[i].cs); Serial.print("): ");
      if (!present[i]) { Serial.print("ABSENT   "); continue; }
      Serial.print(pkt[i]); Serial.print(" pkt/s  fsync="); Serial.print(tag[i]); Serial.print("/s   ");
      pkt[i]=0; tag[i]=0;
    }
    Serial.println();
  }
}
