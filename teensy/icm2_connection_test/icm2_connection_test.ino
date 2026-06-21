#include <Arduino.h>
#include <SPI.h>

// Focused electrical/connection test for the first two ICM-42605 IMUs.
// Shared bus: SCLK=13, MOSI/SDA=11, MISO/AD0=12.
// Unique selects: IMU0 CS=2, IMU1 CS=3. INT1 is not required by this test.

static constexpr uint8_t CS_PINS[] = {2, 3};
static constexpr uint8_t ALL_CS_PINS[] = {2, 3, 4, 5, 6, 7};
// These are INT1 pins in the intended layout, but earlier bench wiring used
// pin 8 as a CS. This connection-only sketch does not enable IMU interrupts,
// so drive them HIGH to conclusively deselect any alternate-CS wiring.
static constexpr uint8_t POSSIBLE_ALT_CS_PINS[] = {8, 9, 10, 14, 15, 16};
static constexpr uint8_t NUM_IMUS = sizeof(CS_PINS) / sizeof(CS_PINS[0]);

static constexpr uint8_t REG_DEVICE_CONFIG = 0x11;
static constexpr uint8_t REG_TEMP_DATA1    = 0x1D;
static constexpr uint8_t REG_PWR_MGMT0     = 0x4E;
static constexpr uint8_t REG_GYRO_CONFIG0  = 0x4F;
static constexpr uint8_t REG_ACCEL_CONFIG0 = 0x50;
static constexpr uint8_t REG_WHO_AM_I      = 0x75;
static constexpr uint8_t REG_BANK_SEL      = 0x76;
static constexpr uint8_t WHO_AM_I_EXPECTED = 0x42;

static SPISettings ICM_SPI(1000000, MSBFIRST, SPI_MODE3);

struct ImuStats {
  uint32_t transactions;
  uint32_t whoErrors;
  uint32_t allZeroFrames;
  uint32_t allOneFrames;
  uint8_t lastWho;
  bool configured;
};

ImuStats stats[NUM_IMUS] = {};

static int16_t be16(const uint8_t *p) {
  return (int16_t)(((uint16_t)p[0] << 8) | p[1]);
}

void deselectAll() {
  for (uint8_t pin : ALL_CS_PINS) digitalWriteFast(pin, HIGH);
}

void writeReg(uint8_t cs, uint8_t reg, uint8_t value) {
  SPI.beginTransaction(ICM_SPI);
  deselectAll();
  digitalWriteFast(cs, LOW);
  SPI.transfer(reg & 0x7F);
  SPI.transfer(value);
  digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
}

uint8_t readReg(uint8_t cs, uint8_t reg) {
  SPI.beginTransaction(ICM_SPI);
  deselectAll();
  digitalWriteFast(cs, LOW);
  SPI.transfer(reg | 0x80);
  uint8_t value = SPI.transfer(0xFF);
  digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
  return value;
}

void readBurst(uint8_t cs, uint8_t reg, uint8_t *dst, size_t count) {
  SPI.beginTransaction(ICM_SPI);
  deselectAll();
  digitalWriteFast(cs, LOW);
  SPI.transfer(reg | 0x80);
  for (size_t i = 0; i < count; ++i) dst[i] = SPI.transfer(0xFF);
  digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
}

bool configureImu(uint8_t index) {
  const uint8_t cs = CS_PINS[index];
  writeReg(cs, REG_BANK_SEL, 0);
  writeReg(cs, REG_DEVICE_CONFIG, 0x01);
  delay(20);

  uint8_t good = 0;
  for (uint8_t attempt = 0; attempt < 20 && good < 3; ++attempt) {
    writeReg(cs, REG_BANK_SEL, 0);
    stats[index].lastWho = readReg(cs, REG_WHO_AM_I);
    good = stats[index].lastWho == WHO_AM_I_EXPECTED ? good + 1 : 0;
    delay(2);
  }
  if (good < 3) return false;

  writeReg(cs, REG_GYRO_CONFIG0, 0x0F);   // +/-2000 dps, 500 Hz
  writeReg(cs, REG_ACCEL_CONFIG0, 0x0F);  // +/-16 g, 500 Hz
  writeReg(cs, REG_PWR_MGMT0, 0x0F);      // accel and gyro low-noise
  delay(50);
  return true;
}

void testImu(uint8_t index, int16_t *values) {
  const uint8_t cs = CS_PINS[index];
  uint8_t raw[14];

  stats[index].lastWho = readReg(cs, REG_WHO_AM_I);
  stats[index].transactions++;
  if (stats[index].lastWho != WHO_AM_I_EXPECTED) stats[index].whoErrors++;

  readBurst(cs, REG_TEMP_DATA1, raw, sizeof(raw));
  bool allZero = true;
  bool allOne = true;
  for (uint8_t value : raw) {
    if (value != 0x00) allZero = false;
    if (value != 0xFF) allOne = false;
  }
  if (allZero) stats[index].allZeroFrames++;
  if (allOne) stats[index].allOneFrames++;

  // TEMP occupies bytes 0..1, then accel XYZ and gyro XYZ.
  for (uint8_t axis = 0; axis < 6; ++axis) values[axis] = be16(&raw[2 + axis * 2]);
}

void printReport() {
  Serial.println("\nidx cs configured who  transactions who_err zero_frame ff_frame   ax      ay      az      gx      gy      gz");
  for (uint8_t i = 0; i < NUM_IMUS; ++i) {
    int16_t v[6];
    testImu(i, v);
    Serial.printf(" %u   %u     %-3s     %02X   %10lu %8lu %10lu %8lu %7d %7d %7d %7d %7d %7d\n",
                  i, CS_PINS[i], stats[i].configured ? "yes" : "NO",
                  stats[i].lastWho, stats[i].transactions, stats[i].whoErrors,
                  stats[i].allZeroFrames, stats[i].allOneFrames,
                  v[0], v[1], v[2], v[3], v[4], v[5]);
  }
}

void setup() {
  Serial.begin(2000000);
  uint32_t started = millis();
  while (!Serial && millis() - started < 3000) {}

  // Every possible CS is driven high before SPI starts. This prevents any
  // additional connected IMU from driving MISO during the two-device test.
  for (uint8_t pin : ALL_CS_PINS) {
    pinMode(pin, OUTPUT);
    digitalWriteFast(pin, HIGH);
  }
  for (uint8_t pin : POSSIBLE_ALT_CS_PINS) {
    pinMode(pin, OUTPUT);
    digitalWriteFast(pin, HIGH);
  }
  SPI.begin();
  delay(50);

  Serial.println("# ICM-42605 two-IMU connection test");
  Serial.println("# shared SCLK=13 MOSI/SDA=11 MISO/AD0=12; IMU0 CS=2; IMU1 CS=3");
  Serial.println("# SPI mode 3 at 1 MHz; expected WHO_AM_I=0x42");
  for (uint8_t i = 0; i < NUM_IMUS; ++i) {
    stats[i].configured = configureImu(i);
    Serial.printf("# IMU%u CS=%u initialization: %s (last WHO_AM_I=0x%02X)\n",
                  i, CS_PINS[i], stats[i].configured ? "PASS" : "FAIL", stats[i].lastWho);
  }
}

void loop() {
  // Alternate devices rapidly. Shared-MISO/CS contention generally appears as
  // intermittent WHO_AM_I corruption here even if a one-shot scan succeeds.
  int16_t ignored[6];
  for (uint16_t pass = 0; pass < 500; ++pass) {
    testImu(0, ignored);
    testImu(1, ignored);
  }
  printReport();
  delay(1000);
}
