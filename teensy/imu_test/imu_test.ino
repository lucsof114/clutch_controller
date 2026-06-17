#include <Arduino.h>
#include <Wire.h>

// =============================================================================
// 6-IMU bring-up / diagnostic sketch (Teensy 4.1)
//
// Standalone test for the multi-IMU logger architecture. NO SD card involved.
// It exercises the exact timing path the logger uses (per-IMU ISR timestamp
// queue + round-robin blocking I2C reads) and reports health, so you can verify
// wiring and timing fidelity before trusting the real logger.
//
// Three phases, printed to serial @ 2 Mbaud:
//   1. CONNECTIVITY   - per-bus address scan + WHO_AM_I for each configured IMU
//   2. DATA SANITY    - configure + read one sample each; scale & sanity-check
//   3. TIMING         - run all IMUs under full DATA_READY interrupt load for
//                       TEST_SECONDS and report per-IMU rate, dt jitter, I2C
//                       read time, dropped timestamps and ISR queue overflows.
//
// Send 'r' over serial to re-run the timing phase.
//
// Wiring under test (6-IMU table; AD0 on the Wire-bus pair is swapped):
//   Wire  (SDA18/SCL19):  0x69 INT2,  0x68 INT3
//   Wire1 (SDA17/SCL16):  0x68 INT4,  0x69 INT5
//   Wire2 (SDA25/SCL24):  0x68 INT6,  0x69 INT7
// =============================================================================

#ifndef F_CPU_ACTUAL
#define F_CPU_ACTUAL F_CPU
#endif
#ifndef ARM_DEMCR
#define ARM_DEMCR (*(volatile uint32_t *)0xE000EDFC)
#endif
#ifndef ARM_DEMCR_TRCENA
#define ARM_DEMCR_TRCENA (1UL << 24)
#endif
#ifndef ARM_DWT_CTRL
#define ARM_DWT_CTRL (*(volatile uint32_t *)0xE0001000)
#endif
#ifndef ARM_DWT_CYCCNT
#define ARM_DWT_CYCCNT (*(volatile uint32_t *)0xE0001004)
#endif
#ifndef ARM_DWT_CTRL_CYCCNTENA
#define ARM_DWT_CTRL_CYCCNTENA (1UL << 0)
#endif

// ---- MPU-6050 registers ----
static constexpr uint8_t REG_SMPLRT_DIV   = 0x19;
static constexpr uint8_t REG_CONFIG       = 0x1A;
static constexpr uint8_t REG_GYRO_CONFIG  = 0x1B;
static constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
static constexpr uint8_t REG_FIFO_EN      = 0x23;
static constexpr uint8_t REG_INT_PIN_CFG  = 0x37;
static constexpr uint8_t REG_INT_ENABLE   = 0x38;
static constexpr uint8_t REG_INT_STATUS   = 0x3A;
static constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
static constexpr uint8_t REG_USER_CTRL    = 0x6A;
static constexpr uint8_t REG_PWR_MGMT_1   = 0x6B;
static constexpr uint8_t REG_PWR_MGMT_2   = 0x6C;
static constexpr uint8_t REG_WHO_AM_I     = 0x75;

static constexpr uint8_t SMPLRT_DIV   = 19;    // 8000/(1+19) = 400 Hz
static constexpr uint8_t GYRO_CONFIG  = 0x18;  // ±2000 dps, 16.4 LSB/(dps)
static constexpr uint8_t ACCEL_CONFIG = 0x08;  // ±4 g,      8192 LSB/g

static constexpr float ACCEL_LSB_PER_G   = 8192.0f;
static constexpr float GYRO_LSB_PER_DPS  = 16.4f;
static constexpr uint16_t SAMPLE_RATE_HZ = 8000 / (1 + SMPLRT_DIV);   // 400
static constexpr uint32_t TEST_SECONDS   = 10;

// ---- IMU configuration table (full 6-IMU layout) ----
struct ImuConfig { TwoWire *bus; uint8_t addr; uint8_t intPin; };

static const ImuConfig IMUS[] = {
  {&Wire,  0x69, 2}, {&Wire,  0x68, 3},   // Wire:  SDA18/SCL19  (AD0 swapped vs the others)
  {&Wire1, 0x68, 4}, {&Wire1, 0x69, 5},   // Wire1: SDA17/SCL16
  {&Wire2, 0x68, 6}, {&Wire2, 0x69, 7},   // Wire2: SDA25/SCL24
};

static constexpr uint8_t MAX_IMUS = 6;
static constexpr uint8_t NUM_IMUS = sizeof(IMUS) / sizeof(IMUS[0]);
static_assert(NUM_IMUS <= MAX_IMUS, "increase MAX_IMUS / ISR_TABLE");

// ---- Per-IMU timestamp queue (single producer ISR / single consumer loop) ----
static constexpr uint16_t TS_QUEUE_SIZE = 256;
static constexpr uint16_t TS_QUEUE_MASK = TS_QUEUE_SIZE - 1;

struct ImuSample { int16_t ax, ay, az, temp, gx, gy, gz; };

struct ImuState {
  volatile uint32_t tsQueue[TS_QUEUE_SIZE];
  volatile uint16_t tsHead;
  volatile uint16_t tsTail;
  volatile uint32_t isrCount;
  volatile uint32_t isrOverflowCount;
  bool     present;     // responded to WHO_AM_I
  bool     havePrev;
  uint32_t prevCyc;
};
ImuState imuState[MAX_IMUS];

// ---- Per-IMU statistics gathered during the timing phase ----
struct ImuStats {
  uint32_t samples;        // timestamps consumed + read
  uint32_t readErrors;
  uint32_t droppedTs;      // cumulative stale timestamps dropped
  uint32_t dtCount;        // consecutive samples (dropped==0) used for dt stats
  uint32_t dtMinCyc, dtMaxCyc;
  uint64_t dtSumCyc, dtSumSqCyc;
  uint32_t rdMinCyc, rdMaxCyc;
  uint64_t rdSumCyc;
};
ImuStats stats[MAX_IMUS];

void enableCycleCounter() {
  ARM_DEMCR |= ARM_DEMCR_TRCENA;
  ARM_DWT_CYCCNT = 0;
  ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
}

static inline double cycToUs(uint64_t c) {
  return (double)c * 1000000.0 / (double)F_CPU_ACTUAL;
}

// ---- Tiny per-IMU ISR (timestamp only), templated like the real logger ----
template <uint8_t I>
void imuISR() {
  uint32_t t = ARM_DWT_CYCCNT;
  ImuState &st = imuState[I];
  uint16_t head = st.tsHead;
  uint16_t nextHead = (head + 1) & TS_QUEUE_MASK;
  if (nextHead != st.tsTail) {
    st.tsQueue[head] = t;
    st.tsHead = nextHead;
  } else {
    st.isrOverflowCount++;
  }
  st.isrCount++;
}

typedef void (*IsrFn)();
static const IsrFn ISR_TABLE[MAX_IMUS] = {
  imuISR<0>, imuISR<1>, imuISR<2>, imuISR<3>, imuISR<4>, imuISR<5>,
};

// No noInterrupts(): SPSC queue, atomic 16-bit volatile access on Cortex-M7.
bool popNewestTimestamp(ImuState &st, uint32_t &t_cyc, uint16_t &dropped) {
  uint16_t head = st.tsHead;
  uint16_t tail = st.tsTail;
  if (tail == head) return false;
  dropped = 0;
  while (tail != head) {
    t_cyc = st.tsQueue[tail];
    tail = (tail + 1) & TS_QUEUE_MASK;
    if (tail != head) dropped++;
  }
  st.tsTail = tail;
  return true;
}

// ---- I2C helpers (parameterized by bus + addr) ----
bool writeReg(TwoWire &bus, uint8_t addr, uint8_t reg, uint8_t value) {
  bus.beginTransmission(addr);
  bus.write(reg);
  bus.write(value);
  return bus.endTransmission(true) == 0;
}

bool readReg(TwoWire &bus, uint8_t addr, uint8_t reg, uint8_t &value) {
  bus.beginTransmission(addr);
  bus.write(reg);
  if (bus.endTransmission(false) != 0) return false;
  if (bus.requestFrom(addr, (uint8_t)1, (uint8_t)true) != 1) return false;
  value = bus.read();
  return true;
}

bool probe(TwoWire &bus, uint8_t addr) {
  bus.beginTransmission(addr);
  return bus.endTransmission(true) == 0;
}

static inline int16_t readWord(TwoWire &bus) {
  uint8_t hi = bus.read();
  uint8_t lo = bus.read();
  return (int16_t)((hi << 8) | lo);
}

bool readImu(TwoWire &bus, uint8_t addr, ImuSample &s) {
  bus.beginTransmission(addr);
  bus.write(REG_ACCEL_XOUT_H);
  if (bus.endTransmission(false) != 0) return false;
  if (bus.requestFrom(addr, (uint8_t)14, (uint8_t)true) != 14) return false;
  s.ax = readWord(bus); s.ay = readWord(bus); s.az = readWord(bus);
  s.temp = readWord(bus);
  s.gx = readWord(bus); s.gy = readWord(bus); s.gz = readWord(bus);
  return true;
}

bool configureMpu6050(TwoWire &bus, uint8_t addr) {
  bool ok = true;
  ok &= writeReg(bus, addr, REG_INT_ENABLE, 0x00);
  ok &= writeReg(bus, addr, REG_PWR_MGMT_1, 0x80);  delay(100);
  ok &= writeReg(bus, addr, REG_PWR_MGMT_1, 0x01);  delay(10);
  ok &= writeReg(bus, addr, REG_PWR_MGMT_2, 0x00);
  ok &= writeReg(bus, addr, REG_FIFO_EN, 0x00);
  ok &= writeReg(bus, addr, REG_USER_CTRL, 0x00);
  ok &= writeReg(bus, addr, REG_CONFIG, 0x00);
  ok &= writeReg(bus, addr, REG_SMPLRT_DIV, SMPLRT_DIV);
  ok &= writeReg(bus, addr, REG_GYRO_CONFIG, GYRO_CONFIG);
  ok &= writeReg(bus, addr, REG_ACCEL_CONFIG, ACCEL_CONFIG);
  ok &= writeReg(bus, addr, REG_INT_PIN_CFG, 0x00);   // pulsed, auto-clear
  uint8_t dummy = 0;
  readReg(bus, addr, REG_INT_STATUS, dummy);
  return ok;
}

static uint8_t busIndex(TwoWire *b) {
  if (b == &Wire)  return 0;
  if (b == &Wire1) return 1;
  if (b == &Wire2) return 2;
  return 255;
}
static const char *busName(TwoWire *b) {
  switch (busIndex(b)) {
    case 0: return "Wire ";
    case 1: return "Wire1";
    case 2: return "Wire2";
    default: return "?????";
  }
}

void initBuses() {
  TwoWire *seen[3] = {nullptr, nullptr, nullptr};
  uint8_t nSeen = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    TwoWire *b = IMUS[i].bus;
    bool already = false;
    for (uint8_t j = 0; j < nSeen; j++) if (seen[j] == b) { already = true; break; }
    if (already) continue;
    b->begin();
    b->setClock(400000);
    if (nSeen < 3) seen[nSeen++] = b;
  }
}

// =============================================================================
// Phase 1: connectivity
// =============================================================================
void phaseConnectivity() {
  Serial.println();
  Serial.println("===== PHASE 1: CONNECTIVITY =====");

  // Per-bus presence scan at the two MPU addresses (catches AD0 / bus mistakes).
  TwoWire *scanned[3] = {nullptr, nullptr, nullptr};
  uint8_t nScanned = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    TwoWire *b = IMUS[i].bus;
    bool done = false;
    for (uint8_t j = 0; j < nScanned; j++) if (scanned[j] == b) { done = true; break; }
    if (done) continue;
    Serial.print("  bus "); Serial.print(busName(b)); Serial.print(": ");
    bool any = false;
    for (uint8_t a = 0x68; a <= 0x69; a++) {
      if (probe(*b, a)) {
        Serial.print("0x"); Serial.print(a, HEX); Serial.print(" ");
        any = true;
      }
    }
    if (!any) Serial.print("(no devices - check SDA/SCL/power/pull-ups)");
    Serial.println();
    if (nScanned < 3) scanned[nScanned++] = b;
  }

  Serial.println("  idx  bus    addr  intPin  WHO_AM_I  result");
  uint8_t present = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    TwoWire &bus = *IMUS[i].bus;
    uint8_t addr = IMUS[i].addr;
    uint8_t who = 0;
    bool ok = readReg(bus, addr, REG_WHO_AM_I, who);
    imuState[i].present = (ok && who == 0x68);

    char line[80];
    snprintf(line, sizeof(line), "  %2u   %s  0x%02X    %3u     0x%02X      %s",
             i, busName(IMUS[i].bus), addr, IMUS[i].intPin,
             ok ? who : 0,
             imuState[i].present ? "OK" :
             (!ok ? "NO RESPONSE" : "BAD WHO_AM_I"));
    Serial.println(line);
    if (imuState[i].present) present++;
  }
  Serial.print("  -> "); Serial.print(present); Serial.print("/");
  Serial.print(NUM_IMUS); Serial.println(" IMUs responding.");
}

// =============================================================================
// Phase 2: configure + single-sample data sanity
// =============================================================================
void phaseDataSanity() {
  Serial.println();
  Serial.println("===== PHASE 2: DATA SANITY (sensors should be still) =====");
  Serial.println("  idx  |a|(g)   az(g)   gmag(dps)  temp(C)  result");

  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (!imuState[i].present) {
      Serial.print("  "); Serial.print(i); Serial.println("    -- skipped (not present) --");
      continue;
    }
    TwoWire &bus = *IMUS[i].bus;
    uint8_t addr = IMUS[i].addr;

    if (!configureMpu6050(bus, addr)) {
      Serial.print("  "); Serial.print(i); Serial.println("    CONFIG FAILED");
      imuState[i].present = false;
      continue;
    }
    delay(5);
    ImuSample s;
    if (!readImu(bus, addr, s)) {
      Serial.print("  "); Serial.print(i); Serial.println("    READ FAILED");
      continue;
    }

    float gx = s.ax / ACCEL_LSB_PER_G, gy = s.ay / ACCEL_LSB_PER_G, gz = s.az / ACCEL_LSB_PER_G;
    float amag = sqrtf(gx*gx + gy*gy + gz*gz);
    float wx = s.gx / GYRO_LSB_PER_DPS, wy = s.gy / GYRO_LSB_PER_DPS, wz = s.gz / GYRO_LSB_PER_DPS;
    float gmag = sqrtf(wx*wx + wy*wy + wz*wz);
    float tc = s.temp / 340.0f + 36.53f;

    bool accelOk = (amag > 0.7f && amag < 1.3f);     // ~1 g at rest
    bool gyroOk  = (gmag < 20.0f);                    // near zero at rest
    bool tempOk  = (tc > 10.0f && tc < 50.0f);
    const char *res = (accelOk && gyroOk && tempOk) ? "OK" :
                      (!accelOk ? "WARN accel!=1g" : (!gyroOk ? "WARN gyro motion?" : "WARN temp"));

    char line[100];
    snprintf(line, sizeof(line), "  %2u   %5.2f   %+5.2f   %7.1f   %6.1f   %s",
             i, amag, gz, gmag, tc, res);
    Serial.println(line);
  }
}

// =============================================================================
// Phase 3: full-load interrupt timing test
// =============================================================================
void resetStats() {
  for (uint8_t i = 0; i < MAX_IMUS; i++) {
    stats[i] = ImuStats{};
    stats[i].dtMinCyc = 0xFFFFFFFF;
    stats[i].rdMinCyc = 0xFFFFFFFF;
    imuState[i].tsHead = 0;
    imuState[i].tsTail = 0;
    imuState[i].isrCount = 0;
    imuState[i].isrOverflowCount = 0;
    imuState[i].havePrev = false;
    imuState[i].prevCyc = 0;
  }
}

void phaseTiming() {
  Serial.println();
  Serial.print("===== PHASE 3: TIMING under full load ("); Serial.print(TEST_SECONDS);
  Serial.println("s) =====");

  resetStats();

  // Attach interrupts + enable DATA_READY only for present IMUs.
  uint8_t active = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (!imuState[i].present) continue;
    pinMode(IMUS[i].intPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(IMUS[i].intPin), ISR_TABLE[i], RISING);
    writeReg(*IMUS[i].bus, IMUS[i].addr, REG_INT_ENABLE, 0x01);
    active++;
  }
#if defined(__IMXRT1062__)
  NVIC_SET_PRIORITY(IRQ_GPIO6789, 16);
#endif

  if (active == 0) { Serial.println("  no present IMUs to test."); return; }

  // Measurement loop: mirrors the logger's serviceImus() exactly, but
  // accumulates statistics instead of writing records. Silent during the run.
  uint32_t loopIters = 0;
  uint32_t startMs = millis();
  while (millis() - startMs < TEST_SECONDS * 1000) {
    loopIters++;
    for (uint8_t i = 0; i < NUM_IMUS; i++) {
      ImuState &st = imuState[i];
      if (!st.present) continue;

      uint32_t tCyc;
      uint16_t dropped = 0;
      if (!popNewestTimestamp(st, tCyc, dropped)) continue;

      ImuStats &S = stats[i];
      uint32_t dtCyc = 0;
      bool haveDt = st.havePrev;
      if (st.havePrev) dtCyc = tCyc - st.prevCyc;
      else st.havePrev = true;
      st.prevCyc = tCyc;
      S.droppedTs += dropped;

      uint32_t r0 = ARM_DWT_CYCCNT;
      ImuSample s;
      bool ok = readImu(*IMUS[i].bus, IMUS[i].addr, s);
      uint32_t rd = ARM_DWT_CYCCNT - r0;

      S.samples++;
      if (!ok) { S.readErrors++; continue; }

      // I2C read-time stats.
      if (rd < S.rdMinCyc) S.rdMinCyc = rd;
      if (rd > S.rdMaxCyc) S.rdMaxCyc = rd;
      S.rdSumCyc += rd;

      // dt stats only for consecutive samples (no dropped) -> true jitter.
      if (haveDt && dropped == 0) {
        if (dtCyc < S.dtMinCyc) S.dtMinCyc = dtCyc;
        if (dtCyc > S.dtMaxCyc) S.dtMaxCyc = dtCyc;
        S.dtSumCyc += dtCyc;
        S.dtSumSqCyc += (uint64_t)dtCyc * (uint64_t)dtCyc;
        S.dtCount++;
      }
    }
  }
  uint32_t elapsedMs = millis() - startMs;

  // Stop interrupts.
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (!imuState[i].present) continue;
    writeReg(*IMUS[i].bus, IMUS[i].addr, REG_INT_ENABLE, 0x00);
    detachInterrupt(digitalPinToInterrupt(IMUS[i].intPin));
  }

  // ---- Report ----
  double expectedUs = 1000000.0 / SAMPLE_RATE_HZ;
  Serial.print("  loop iterations: "); Serial.print(loopIters);
  Serial.print("  ("); Serial.print((double)loopIters * 1000.0 / elapsedMs, 0);
  Serial.println(" i/s)");
  Serial.print("  expected per-IMU rate: "); Serial.print(SAMPLE_RATE_HZ);
  Serial.print(" Hz, period "); Serial.print(expectedUs, 1); Serial.println(" us");
  Serial.println();
  Serial.println("  idx  bus    Hz     dt_mean  dt_jit(sd)  dt_p2p   i2c_rd   drop  ovf  rderr  isr");

  uint32_t totSamples = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (!imuState[i].present) continue;
    ImuStats &S = stats[i];
    totSamples += S.samples;

    double hz = (double)S.samples * 1000.0 / elapsedMs;
    double dtMean = S.dtCount ? cycToUs(S.dtSumCyc / S.dtCount) : 0;
    double dtP2p  = S.dtCount ? cycToUs(S.dtMaxCyc - S.dtMinCyc) : 0;
    double rdMean = S.samples ? cycToUs(S.rdSumCyc / (S.samples - S.readErrors > 0 ? S.samples - S.readErrors : 1)) : 0;

    // std dev of dt in us
    double sd = 0;
    if (S.dtCount > 1) {
      double n = S.dtCount;
      double mean = (double)S.dtSumCyc / n;
      double var = (double)S.dtSumSqCyc / n - mean * mean;
      if (var < 0) var = 0;
      sd = cycToUs((uint64_t)sqrt(var));
    }

    char line[140];
    snprintf(line, sizeof(line),
      "  %2u   %s  %5.1f  %7.1f  %9.2f  %7.2f  %6.1f  %4lu  %3lu  %4lu  %3s",
      i, busName(IMUS[i].bus), hz, dtMean, sd, dtP2p, rdMean,
      (unsigned long)S.droppedTs, (unsigned long)imuState[i].isrOverflowCount,
      (unsigned long)S.readErrors,
      imuState[i].isrCount == 0 ? "NO!" : "ok");
    Serial.println(line);
  }

  Serial.print("  total throughput: ");
  Serial.print((double)totSamples * 1000.0 / elapsedMs, 0);
  Serial.print(" samples/s  (target ");
  Serial.print((uint32_t)active * SAMPLE_RATE_HZ);
  Serial.println(")");

  // ---- Verdicts ----
  Serial.println();
  Serial.println("  diagnosis:");
  bool clean = true;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (!imuState[i].present) continue;
    ImuStats &S = stats[i];
    double hz = (double)S.samples * 1000.0 / elapsedMs;
    if (imuState[i].isrCount == 0) {
      Serial.print("   IMU "); Serial.print(i);
      Serial.print(": NO INTERRUPTS - check INT wiring to pin ");
      Serial.println(IMUS[i].intPin);
      clean = false;
    } else if (hz < SAMPLE_RATE_HZ * 0.95) {
      Serial.print("   IMU "); Serial.print(i);
      Serial.print(": low rate ("); Serial.print(hz, 0);
      Serial.println(" Hz) - loop not keeping up or sensor not at 400Hz");
      clean = false;
    }
    if (S.droppedTs > S.samples / 50) {  // >2% dropped
      Serial.print("   IMU "); Serial.print(i);
      Serial.print(": "); Serial.print(S.droppedTs);
      Serial.println(" dropped timestamps - read loop falling behind (I2C/SD budget)");
      clean = false;
    }
    if (imuState[i].isrOverflowCount > 0) {
      Serial.print("   IMU "); Serial.print(i);
      Serial.println(": ISR queue overflow - consumer starved (should never happen here)");
      clean = false;
    }
    if (S.readErrors > 0) {
      Serial.print("   IMU "); Serial.print(i);
      Serial.print(": "); Serial.print(S.readErrors);
      Serial.println(" I2C read errors - flaky bus (pull-ups / wiring / clock)");
      clean = false;
    }
  }
  if (clean) Serial.println("   all present IMUs: full rate, low jitter, no drops. PASS.");
}

void runAll() {
  phaseConnectivity();
  phaseDataSanity();
  phaseTiming();
  Serial.println();
  Serial.println("done. send 'a' to re-run all phases, 'r' for timing only.");
}

void setup() {
  Serial.begin(2000000);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}   // wait briefly for the monitor
  delay(200);

  Serial.println();
  Serial.println("############ 6-IMU DIAGNOSTIC ############");
  Serial.print("F_CPU_ACTUAL = "); Serial.print(F_CPU_ACTUAL / 1000000); Serial.println(" MHz");

  enableCycleCounter();
  initBuses();
  runAll();
}

void loop() {
  if (Serial.available()) {
    int c = Serial.read();
    if (c == 'a' || c == 'A') runAll();              // re-run connectivity + sanity + timing
    else if (c == 'r' || c == 'R') phaseTiming();    // timing only
  }
}
