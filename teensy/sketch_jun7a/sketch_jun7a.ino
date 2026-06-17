#include <Arduino.h>
#include <Wire.h>
#include <SdFat.h>
#include <RingBuf.h>   // SdFat's logging FIFO (not pulled in by SdFat.h)

// =============================================================================
// Multi-IMU SD logger for Teensy 4.1
//
// Reads up to MAX_IMUS MPU-6050s at 400 Hz each, hardware-timestamped in a tiny
// per-IMU ISR (ARM_DWT_CYCCNT), and streams fixed-size binary records to the
// built-in micro-SD slot via an in-RAM FIFO (SdFat RingBuf) drained in loop().
//
// Design goal: timestamp fidelity must not degrade as IMUs are added. That holds
// because (a) each IMU's ISR is O(1) and only records a cycle count, (b) the
// loop-side consumer never disables interrupts, and (c) SD write latency is
// absorbed by the RingBuf, never blocking the ISRs.
//
// Wiring (per IMU):
//   MPU VCC -> 3.3V, GND -> GND
//   MPU SDA/SCL -> the IMU's I2C bus (see IMUS[] table below)
//   MPU AD0 -> 3.3V for addr 0x69, GND for addr 0x68
//   MPU INT -> the IMU's int pin (see IMUS[] table)
//
// Teensy 4.1 I2C buses:  Wire = SDA18/SCL19,  Wire1 = SDA17/SCL16,  Wire2 = SDA25/SCL24
//
// Serial: a ~1 Hz status line (not per-sample). Send 's' to stop & close the file.
// =============================================================================

// ---- Fallbacks in case these are not already defined by the Teensy core. ----
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

// Sensor configuration constants (also recorded in the file header for decoding).
static constexpr uint8_t SMPLRT_DIV  = 19;    // 8000/(1+19) = 400 Hz
static constexpr uint8_t GYRO_CONFIG  = 0x18; // ±2000 deg/s, 16.4 LSB/(deg/s)
static constexpr uint8_t ACCEL_CONFIG = 0x08; // ±4 g, 8192 LSB/g

// =============================================================================
// IMU configuration table
//
// Add an entry per physically wired IMU. The system scales to MAX_IMUS; the
// timestamp path is identical for every IMU, so fidelity is preserved as the
// count grows. Up to 2 IMUs per I2C bus (addr 0x68 and 0x69).
//
// Default below = the single IMU currently wired (0x69 on Wire, INT pin 2).
// Example full 6-IMU layout (uncomment / adapt as buses are populated):
//   {&Wire,  0x68, 2}, {&Wire,  0x69, 3},
//   {&Wire1, 0x68, 4}, {&Wire1, 0x69, 5},
//   {&Wire2, 0x68, 6}, {&Wire2, 0x69, 7},
// =============================================================================
struct ImuConfig {
  TwoWire *bus;
  uint8_t  addr;
  uint8_t  intPin;
};

static const ImuConfig IMUS[] = {
  {&Wire, 0x69, 2},
};

static constexpr uint8_t MAX_IMUS = 6;
static constexpr uint8_t NUM_IMUS = sizeof(IMUS) / sizeof(IMUS[0]);
static_assert(NUM_IMUS <= MAX_IMUS, "increase MAX_IMUS / ISR_TABLE for this many IMUs");

// =============================================================================
// Per-IMU timestamp queue + state
//
// Single-producer (its ISR) / single-consumer (loop) queue of CPU cycle counts.
// TS_QUEUE_SIZE must be a power of 2.
// =============================================================================
static constexpr uint16_t TS_QUEUE_SIZE = 256;
static constexpr uint16_t TS_QUEUE_MASK = TS_QUEUE_SIZE - 1;

struct ImuSample {
  int16_t ax, ay, az, temp, gx, gy, gz;
};

struct ImuState {
  // Timestamp ring buffer (ISR writes head, loop writes tail).
  volatile uint32_t tsQueue[TS_QUEUE_SIZE];
  volatile uint16_t tsHead;
  volatile uint16_t tsTail;
  volatile uint32_t isrCount;
  volatile uint32_t isrOverflowCount;

  // Loop-side delta tracking.
  bool     havePrev;
  uint32_t prevCyc;

  // Status / health counters.
  bool     active;
  uint32_t sampleCount;        // records successfully queued for SD
  uint32_t droppedTimestamps;  // cumulative stale timestamps dropped
  uint32_t i2cErrors;
  uint32_t lastReportCount;    // for per-interval rate
};

ImuState imuState[MAX_IMUS];

// ---- Cycle counter helpers ----
void enableCycleCounter() {
  ARM_DEMCR |= ARM_DEMCR_TRCENA;
  ARM_DWT_CYCCNT = 0;
  ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
}

static inline uint32_t cyclesToUs32(uint32_t cycles) {
  return (uint32_t)(((uint64_t)cycles * 1000000ULL) / (uint64_t)F_CPU_ACTUAL);
}

// =============================================================================
// Tiny per-IMU ISR: timestamp only.
//
// Templated so each IMU gets its own zero-overhead ISR writing into its own
// queue. ISR_TABLE[i] is registered for the i-th configured IMU.
// =============================================================================
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

// =============================================================================
// Pop the newest pending timestamp for one IMU.
//
// No noInterrupts(): this is single-producer/single-consumer and 16-bit aligned
// volatile reads/writes are atomic on the Cortex-M7, so we never mask interrupts
// here. That keeps every other IMU's ISR jitter-free while we consume.
//
// If the loop fell behind, drop older timestamps so we pair fresh register data
// with the newest timestamp, and report how many were dropped.
// =============================================================================
bool popNewestTimestamp(ImuState &st, uint32_t &t_cyc, uint16_t &dropped) {
  uint16_t head = st.tsHead;   // atomic snapshot
  uint16_t tail = st.tsTail;

  if (tail == head) {
    return false;
  }

  dropped = 0;
  while (tail != head) {
    t_cyc = st.tsQueue[tail];
    tail = (tail + 1) & TS_QUEUE_MASK;
    if (tail != head) {
      dropped++;
    }
  }

  st.tsTail = tail;   // atomic publish
  return true;
}

// =============================================================================
// I2C helpers (parameterized by bus + address so any IMU on any bus works).
// =============================================================================
bool writeReg(TwoWire &bus, uint8_t addr, uint8_t reg, uint8_t value) {
  bus.beginTransmission(addr);
  bus.write(reg);
  bus.write(value);
  return bus.endTransmission(true) == 0;
}

bool readReg(TwoWire &bus, uint8_t addr, uint8_t reg, uint8_t &value) {
  bus.beginTransmission(addr);
  bus.write(reg);
  if (bus.endTransmission(false) != 0) {
    return false;
  }
  if (bus.requestFrom(addr, (uint8_t)1, (uint8_t)true) != 1) {
    return false;
  }
  value = bus.read();
  return true;
}

static inline int16_t readWordFromWire(TwoWire &bus) {
  uint8_t hi = bus.read();
  uint8_t lo = bus.read();
  return (int16_t)((hi << 8) | lo);
}

bool readImu(TwoWire &bus, uint8_t addr, ImuSample &s) {
  bus.beginTransmission(addr);
  bus.write(REG_ACCEL_XOUT_H);
  if (bus.endTransmission(false) != 0) {
    return false;
  }
  if (bus.requestFrom(addr, (uint8_t)14, (uint8_t)true) != 14) {
    return false;
  }
  s.ax   = readWordFromWire(bus);
  s.ay   = readWordFromWire(bus);
  s.az   = readWordFromWire(bus);
  s.temp = readWordFromWire(bus);
  s.gx   = readWordFromWire(bus);
  s.gy   = readWordFromWire(bus);
  s.gz   = readWordFromWire(bus);
  return true;
}

bool configureMpu6050For400Hz(TwoWire &bus, uint8_t addr) {
  bool ok = true;

  ok &= writeReg(bus, addr, REG_INT_ENABLE, 0x00);   // disable interrupts during config
  ok &= writeReg(bus, addr, REG_PWR_MGMT_1, 0x80);   // reset
  delay(100);
  ok &= writeReg(bus, addr, REG_PWR_MGMT_1, 0x01);   // wake, PLL w/ X gyro clock
  delay(10);
  ok &= writeReg(bus, addr, REG_PWR_MGMT_2, 0x00);   // all axes on
  ok &= writeReg(bus, addr, REG_FIFO_EN, 0x00);      // FIFO off
  ok &= writeReg(bus, addr, REG_USER_CTRL, 0x00);    // internal I2C master off
  ok &= writeReg(bus, addr, REG_CONFIG, 0x00);       // DLPF off, gyro base 8 kHz
  ok &= writeReg(bus, addr, REG_SMPLRT_DIV, SMPLRT_DIV);   // -> 400 Hz
  ok &= writeReg(bus, addr, REG_GYRO_CONFIG, GYRO_CONFIG); // ±2000 dps
  ok &= writeReg(bus, addr, REG_ACCEL_CONFIG, ACCEL_CONFIG); // ±4 g

  // PULSED interrupt (active-high, push-pull, ~50us, auto-clear). Do NOT use
  // latched (0x30): latching couples re-arming to I2C reads and wrecks timing.
  ok &= writeReg(bus, addr, REG_INT_PIN_CFG, 0x00);

  uint8_t dummy = 0;                                 // clear stale INT status
  readReg(bus, addr, REG_INT_STATUS, dummy);

  return ok;
}

// =============================================================================
// Binary on-SD format
// =============================================================================
static constexpr char     FILE_MAGIC[4] = {'I', 'M', 'U', 'L'};
static constexpr uint16_t FILE_VERSION  = 1;
static constexpr uint16_t HEADER_BLOCK  = 512;   // header padded to one sector

struct __attribute__((packed)) ImuRecord {
  uint8_t  imu_id;
  uint8_t  flags;          // bit0 = I2C read error
  uint16_t dropped_ts;     // stale timestamps dropped for this sample
  uint32_t isr_overflows;  // cumulative for this IMU
  uint32_t t_cyc;          // DWT cycle count at interrupt
  uint32_t dt_cyc;         // delta from previous sample (this IMU)
  int16_t  ax, ay, az;
  int16_t  gx, gy, gz;
  int16_t  temp;
  uint16_t i2c_read_us;    // duration of the register read
};
static_assert(sizeof(ImuRecord) == 32, "ImuRecord must be 32 bytes (16 per 512B sector)");

struct __attribute__((packed)) FileHeader {
  char     magic[4];
  uint16_t version;
  uint16_t record_size;
  uint32_t f_cpu;
  uint16_t sample_rate_hz;
  uint8_t  num_imus;
  uint8_t  gyro_config;
  uint8_t  accel_config;
  uint8_t  smplrt_div;
};

struct __attribute__((packed)) ImuHeaderEntry {
  uint8_t bus_index;   // 0=Wire, 1=Wire1, 2=Wire2
  uint8_t addr;
  uint8_t int_pin;
  uint8_t reserved;
};

static uint8_t busIndex(TwoWire *b) {
  if (b == &Wire)  return 0;
  if (b == &Wire1) return 1;
  if (b == &Wire2) return 2;
  return 255;
}

// =============================================================================
// SD + RingBuf (FIFO drained to disk in loop())
// =============================================================================
#define SD_CONFIG SdioConfig(FIFO_SDIO)
static constexpr size_t   RING_BUF_CAPACITY = 131072;            // 128 KB RAM FIFO
static constexpr uint64_t LOG_PREALLOC_BYTES = 256ULL * 1024 * 1024;

SdFs   sd;
FsFile logFile;
RingBuf<FsFile, RING_BUF_CAPACITY> rb;

volatile bool logging = false;
uint32_t sdOverrun = 0;      // records dropped because the FIFO was full
size_t   rbMaxUsed = 0;      // RingBuf high-water mark
bool     sdWriteError = false;

// =============================================================================
// Setup helpers
// =============================================================================
void resetImuState() {
  for (uint8_t i = 0; i < MAX_IMUS; i++) {
    ImuState &st = imuState[i];
    st.tsHead = 0;
    st.tsTail = 0;
    st.isrCount = 0;
    st.isrOverflowCount = 0;
    st.havePrev = false;
    st.prevCyc = 0;
    st.active = false;
    st.sampleCount = 0;
    st.droppedTimestamps = 0;
    st.i2cErrors = 0;
    st.lastReportCount = 0;
  }
}

// Bring up each unique I2C bus exactly once.
void initBuses() {
  TwoWire *seen[3] = {nullptr, nullptr, nullptr};
  uint8_t  nSeen = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    TwoWire *b = IMUS[i].bus;
    bool already = false;
    for (uint8_t j = 0; j < nSeen; j++) {
      if (seen[j] == b) { already = true; break; }
    }
    if (already) continue;
    b->begin();
    b->setClock(400000);   // MPU-6050 fast-mode max
    if (nSeen < 3) seen[nSeen++] = b;
  }
}

bool openLogFile() {
  if (!sd.begin(SD_CONFIG)) {
    Serial.println("# SD begin() failed. Check card.");
    return false;
  }

  // Find the next free LOGNNNN.BIN.
  char name[16];
  uint16_t idx = 0;
  do {
    snprintf(name, sizeof(name), "LOG%04u.BIN", idx++);
  } while (sd.exists(name) && idx < 10000);

  if (!logFile.open(name, O_RDWR | O_CREAT | O_TRUNC)) {
    Serial.print("# open failed: ");
    Serial.println(name);
    return false;
  }
  if (!logFile.preAllocate(LOG_PREALLOC_BYTES)) {
    Serial.println("# preAllocate failed (continuing unallocated).");
  }

  // Build and write a 512-byte header block so all subsequent writes stay
  // sector-aligned for fast SDIO throughput.
  uint8_t block[HEADER_BLOCK];
  memset(block, 0, sizeof(block));

  FileHeader h;
  memcpy(h.magic, FILE_MAGIC, 4);
  h.version        = FILE_VERSION;
  h.record_size    = sizeof(ImuRecord);
  h.f_cpu          = F_CPU_ACTUAL;
  h.sample_rate_hz = 8000 / (1 + SMPLRT_DIV);
  h.num_imus       = NUM_IMUS;
  h.gyro_config    = GYRO_CONFIG;
  h.accel_config   = ACCEL_CONFIG;
  h.smplrt_div     = SMPLRT_DIV;
  memcpy(block, &h, sizeof(h));

  size_t off = sizeof(h);
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    ImuHeaderEntry e;
    e.bus_index = busIndex(IMUS[i].bus);
    e.addr      = IMUS[i].addr;
    e.int_pin   = IMUS[i].intPin;
    e.reserved  = 0;
    memcpy(block + off, &e, sizeof(e));
    off += sizeof(e);
  }

  if (logFile.write(block, sizeof(block)) != (int)sizeof(block)) {
    Serial.println("# header write failed.");
    return false;
  }

  rb.begin(&logFile);
  Serial.print("# logging to ");
  Serial.println(name);
  return true;
}

void closeLogFile() {
  // Flush whatever remains in the FIFO, then truncate to real size and close.
  rb.sync();
  while (rb.bytesUsed() > 0) {
    if (rb.writeOut(rb.bytesUsed()) == 0) break;
  }
  logFile.truncate();
  logFile.flush();
  logFile.close();
  logging = false;
  Serial.println("# log closed.");
}

// =============================================================================
void setup() {
  Serial.begin(2000000);
  delay(1000);

  enableCycleCounter();
  resetImuState();
  initBuses();

  // Bring up each configured IMU; skip (but don't halt on) any that fail, so a
  // partially-wired multi-IMU rig still logs the working sensors.
  uint8_t activeCount = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    TwoWire &bus = *IMUS[i].bus;
    uint8_t  addr = IMUS[i].addr;

    uint8_t whoami = 0;
    if (!readReg(bus, addr, REG_WHO_AM_I, whoami) || whoami != 0x68) {
      Serial.print("# IMU ");
      Serial.print(i);
      Serial.print(" (bus ");
      Serial.print(busIndex(IMUS[i].bus));
      Serial.print(", addr 0x");
      Serial.print(addr, HEX);
      Serial.print("): bad WHO_AM_I=0x");
      Serial.print(whoami, HEX);
      Serial.println(" -> disabled.");
      continue;
    }
    if (!configureMpu6050For400Hz(bus, addr)) {
      Serial.print("# IMU ");
      Serial.print(i);
      Serial.println(": config failed -> disabled.");
      continue;
    }

    pinMode(IMUS[i].intPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(IMUS[i].intPin), ISR_TABLE[i], RISING);
    imuState[i].active = true;
    activeCount++;
  }

  if (activeCount == 0) {
    Serial.println("# No IMUs initialized. Halting.");
    while (1) delay(100);
  }

#if defined(__IMXRT1062__)
  // Elevate the shared GPIO IRQ so timestamp ISRs preempt most other interrupts.
  NVIC_SET_PRIORITY(IRQ_GPIO6789, 16);
#endif

  if (!openLogFile()) {
    Serial.println("# Failed to open log file. Halting.");
    while (1) delay(100);
  }

  // Enable DATA_READY only after the file is ready, so we don't drop early data.
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (imuState[i].active) {
      writeReg(*IMUS[i].bus, IMUS[i].addr, REG_INT_ENABLE, 0x01);
    }
  }

  logging = true;
  Serial.print("# active IMUs: ");
  Serial.println(activeCount);
  Serial.println("# status: ST,uptime_ms,[imu:id,n,hz,dropped]*,isr_ovf,rb_max,sd_overrun,bytes,werr");
}

// =============================================================================
// Round-robin over active IMUs: timestamp -> register read -> binary record.
// =============================================================================
void serviceImus() {
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    ImuState &st = imuState[i];
    if (!st.active) continue;

    uint32_t tCyc = 0;
    uint16_t dropped = 0;
    if (!popNewestTimestamp(st, tCyc, dropped)) continue;

    uint32_t dtCyc = 0;
    if (st.havePrev) {
      dtCyc = tCyc - st.prevCyc;   // wrap-safe unsigned subtraction
    } else {
      st.havePrev = true;
    }
    st.prevCyc = tCyc;
    st.droppedTimestamps += dropped;

    ImuSample s;
    uint32_t r0 = ARM_DWT_CYCCNT;
    bool ok = readImu(*IMUS[i].bus, IMUS[i].addr, s);
    uint32_t r1 = ARM_DWT_CYCCNT;

    ImuRecord rec;
    rec.imu_id        = i;
    rec.flags         = ok ? 0 : 0x01;
    rec.dropped_ts    = dropped;
    rec.isr_overflows = st.isrOverflowCount;
    rec.t_cyc         = tCyc;
    rec.dt_cyc        = dtCyc;
    if (ok) {
      rec.ax = s.ax; rec.ay = s.ay; rec.az = s.az;
      rec.gx = s.gx; rec.gy = s.gy; rec.gz = s.gz;
      rec.temp = s.temp;
    } else {
      rec.ax = rec.ay = rec.az = 0;
      rec.gx = rec.gy = rec.gz = 0;
      rec.temp = 0;
      st.i2cErrors++;
    }
    rec.i2c_read_us = (uint16_t)cyclesToUs32(r1 - r0);

    // Push to the FIFO atomically (never write a partial record).
    if (rb.bytesFree() >= sizeof(rec)) {
      rb.write((const uint8_t *)&rec, sizeof(rec));
      st.sampleCount++;
    } else {
      sdOverrun++;   // SD couldn't keep up; this sample is lost (recorded as a count)
    }
  }
}

// =============================================================================
// Drain the FIFO to SD in sector-sized chunks when the card isn't busy.
// =============================================================================
void flushSd() {
  size_t used = rb.bytesUsed();
  if (used > rbMaxUsed) rbMaxUsed = used;

  if (used >= 512 && !logFile.isBusy()) {
    if (rb.writeOut(512) != 512) {
      sdWriteError = true;
    }
  }
  if (rb.getWriteError()) {
    sdWriteError = true;
  }
}

// =============================================================================
// ~1 Hz health line (replaces per-sample serial).
// =============================================================================
void statusReport() {
  static uint32_t lastMs = 0;
  uint32_t now = millis();
  uint32_t dt = now - lastMs;
  if (dt < 1000) return;
  lastMs = now;

  Serial.print("ST,");
  Serial.print(now);

  uint32_t isrOvfTotal = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    ImuState &st = imuState[i];
    if (!st.active) continue;
    uint32_t n = st.sampleCount;
    uint32_t hz = ((n - st.lastReportCount) * 1000) / dt;
    st.lastReportCount = n;
    isrOvfTotal += st.isrOverflowCount;

    Serial.print(",");
    Serial.print(i);
    Serial.print(":");
    Serial.print(n);
    Serial.print(":");
    Serial.print(hz);
    Serial.print(":");
    Serial.print(st.droppedTimestamps);
  }

  Serial.print(",");
  Serial.print(isrOvfTotal);
  Serial.print(",");
  Serial.print(rbMaxUsed);
  Serial.print(",");
  Serial.print(sdOverrun);
  Serial.print(",");
  Serial.print((uint32_t)logFile.curPosition());
  Serial.print(",");
  Serial.println(sdWriteError ? 1 : 0);

  rbMaxUsed = 0;   // high-water mark is per-interval

  // Durability checkpoint every ~10 s. flush() forces a directory sync that
  // briefly blocks the loop (may cost one dropped sample), so we do it sparingly
  // rather than every status line. preAllocate() lets the decoder recover data
  // written since the last flush even after an unclean shutdown.
  static uint8_t flushDivider = 0;
  if (++flushDivider >= 10) {
    flushDivider = 0;
    logFile.flush();
  }
}

void handleSerialCommand() {
  if (!Serial.available()) return;
  int c = Serial.read();
  if (c == 's' || c == 'S') {
    closeLogFile();
    Serial.println("# stopped. reset Teensy to start a new log.");
    while (1) delay(100);
  }
}

void loop() {
  if (!logging) return;
  serviceImus();
  flushSd();
  statusReport();
  handleSerialCommand();
}
