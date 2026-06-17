#include <Arduino.h>
#include <Wire.h>

// =============================================================================
// 6-IMU USB logger for Teensy 4.1
//
// Same architecture and binary record format as the SD logger, but the sink is
// the USB serial port instead of the micro-SD card (use this until the SD card
// arrives). The connected host runs imu_usb_recorder.py to save the stream to a
// .bin that is byte-identical to what the SD logger would have written.
//
// Timing is unaffected: records go into a RAM FIFO (non-blocking; drop+count if
// full), and the FIFO is drained to USB only up to Serial.availableForWrite(),
// so a slow/stalled host can never block the ISR timestamps or the I2C reads.
//
// Protocol (host-driven):
//   host sends 'g'  -> device resets, pushes 512-byte header, streams records
//   host sends 's'  -> device stops streaming, returns to idle
// The 512-byte header begins with magic "IMUL" so the host can lock onto the
// stream regardless of any boot text.
//
// Wiring (6-IMU table; AD0 on the Wire-bus pair is swapped):
//   Wire  (SDA18/SCL19):  0x69 INT2,  0x68 INT3
//   Wire1 (SDA17/SCL16):  0x68 INT4,  0x69 INT5
//   Wire2 (SDA25/SCL24):  0x68 INT6,  0x69 INT7
//
// Camera sync: Arduino 30 Hz trigger (5V) -> resistor divider -> Teensy pin 26.
//   Each rising edge is hardware-timestamped on the same clock and emitted as a
//   0xF0 record. Sync is never dropped (held in its own queue until written),
//   so every camera frame trigger is logged on the IMU timebase.
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
static constexpr uint8_t GYRO_CONFIG  = 0x18;  // ±2000 dps
static constexpr uint8_t ACCEL_CONFIG = 0x08;  // ±4 g

// ---- IMU configuration table ----
struct ImuConfig { TwoWire *bus; uint8_t addr; uint8_t intPin; };

static const ImuConfig IMUS[] = {
  {&Wire,  0x69, 2}, {&Wire,  0x68, 3},   // Wire:  SDA18/SCL19  (AD0 swapped)
  {&Wire1, 0x68, 4}, {&Wire1, 0x69, 5},   // Wire1: SDA17/SCL16
  {&Wire2, 0x68, 6}, {&Wire2, 0x69, 7},   // Wire2: SDA25/SCL24
};

static constexpr uint8_t MAX_IMUS = 6;
static constexpr uint8_t NUM_IMUS = sizeof(IMUS) / sizeof(IMUS[0]);
static_assert(NUM_IMUS <= MAX_IMUS, "increase MAX_IMUS / ISR_TABLE");

// ---- Per-IMU timestamp queue ----
static constexpr uint16_t TS_QUEUE_SIZE = 256;
static constexpr uint16_t TS_QUEUE_MASK = TS_QUEUE_SIZE - 1;

struct ImuSample { int16_t ax, ay, az, temp, gx, gy, gz; };

struct ImuState {
  volatile uint32_t tsQueue[TS_QUEUE_SIZE];
  volatile uint16_t tsHead;
  volatile uint16_t tsTail;
  volatile uint32_t isrCount;
  volatile uint32_t isrOverflowCount;
  bool     present;
  bool     havePrev;
  uint32_t prevCyc;
};
ImuState imuState[MAX_IMUS];

// ---- Camera sync timestamp queue (never drop-stale: every edge is kept) ----
// 512 entries @ 30 Hz ≈ 17 s of headroom to ride out any USB stall.
static constexpr uint16_t SYNC_QUEUE_SIZE = 512;
static constexpr uint16_t SYNC_QUEUE_MASK = SYNC_QUEUE_SIZE - 1;
volatile uint32_t syncQueue[SYNC_QUEUE_SIZE];
volatile uint16_t syncHead = 0;
volatile uint16_t syncTail = 0;
volatile uint32_t syncIsrOverflow = 0;   // queue full in ISR (should never happen at 30 Hz)
uint32_t syncSeq = 0;                     // monotonic edge counter (committed on write)
uint32_t syncPrevCyc = 0;
bool     syncHavePrev = false;

void enableCycleCounter() {
  ARM_DEMCR |= ARM_DEMCR_TRCENA;
  ARM_DWT_CYCCNT = 0;
  ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
}

static inline uint32_t cyclesToUs32(uint32_t cycles) {
  return (uint32_t)(((uint64_t)cycles * 1000000ULL) / (uint64_t)F_CPU_ACTUAL);
}

// ---- Per-IMU ISR (timestamp only), templated ----
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

// Camera-sync rising-edge ISR: capture the cycle counter FIRST, then enqueue.
// Same elevated GPIO priority as the IMU ISRs, so it preempts the I²C reads.
void syncISR() {
  uint32_t t = ARM_DWT_CYCCNT;
  uint16_t head = syncHead;
  uint16_t next = (head + 1) & SYNC_QUEUE_MASK;
  if (next != syncTail) {
    syncQueue[head] = t;
    syncHead = next;
  } else {
    syncIsrOverflow++;   // 512-deep queue full — not expected at 30 Hz
  }
}

// SPSC, no noInterrupts(): atomic 16-bit volatile access on Cortex-M7.
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

// ---- I2C helpers ----
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

// ---- Binary format (identical to the SD logger) ----
static constexpr char     FILE_MAGIC[4] = {'I', 'M', 'U', 'L'};
static constexpr uint16_t FILE_VERSION  = 2;       // v2 adds 0xF0 camera-sync records
static constexpr uint16_t HEADER_BLOCK  = 512;
static constexpr uint8_t  STATUS_IMU_ID = 0xFF;   // sentinel: host strips these
static constexpr uint8_t  SYNC_IMU_ID   = 0xF0;   // camera-sync rising-edge event

// ---- Camera sync input (Arduino 30 Hz trigger via a 5V->3.0V divider on pin 26) ----
// A 0xF0 record is emitted per rising edge. For sync records the repurposed fields
// are: isr_overflows = sync sequence number, dt_cyc = cycles since previous edge,
// t_cyc = edge time on the shared clock, dropped_ts = sync ISR queue overflows (=0).
static constexpr uint8_t  SYNC_PIN = 26;

struct __attribute__((packed)) ImuRecord {
  uint8_t  imu_id;
  uint8_t  flags;          // bit0 = I2C read error
  uint16_t dropped_ts;
  uint32_t isr_overflows;
  uint32_t t_cyc;
  uint32_t dt_cyc;
  int16_t  ax, ay, az;
  int16_t  gx, gy, gz;
  int16_t  temp;
  uint16_t i2c_read_us;
};
static_assert(sizeof(ImuRecord) == 32, "ImuRecord must be 32 bytes");

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
  uint8_t bus_index;
  uint8_t addr;
  uint8_t int_pin;
  uint8_t reserved;
};

// =============================================================================
// RAM FIFO drained to USB (the only thing that differs from the SD logger)
// =============================================================================
static constexpr size_t FIFO_SIZE = 1u << 17;   // 128 KB, power of 2
static constexpr size_t FIFO_MASK = FIFO_SIZE - 1;
uint8_t  fifo[FIFO_SIZE];
size_t   fifoHead = 0;
size_t   fifoTail = 0;

static inline size_t fifoUsed() { return (fifoHead - fifoTail) & FIFO_MASK; }
static inline size_t fifoFree() { return FIFO_SIZE - 1 - fifoUsed(); }

bool fifoPush(const uint8_t *data, size_t n) {
  if (fifoFree() < n) return false;
  for (size_t i = 0; i < n; i++) {
    fifo[fifoHead] = data[i];
    fifoHead = (fifoHead + 1) & FIFO_MASK;
  }
  return true;
}

// Drain one contiguous chunk, bounded by what USB can take without blocking.
void drainUsb() {
  size_t used = fifoUsed();
  if (used == 0) return;
  int avail = Serial.availableForWrite();
  if (avail <= 0) return;
  size_t n = used;
  if ((size_t)avail < n) n = (size_t)avail;
  size_t contig = FIFO_SIZE - fifoTail;
  if (n > contig) n = contig;
  Serial.write(&fifo[fifoTail], n);   // <= availableForWrite, never blocks
  fifoTail = (fifoTail + n) & FIFO_MASK;
}

// ---- Streaming state ----
bool     streaming = false;
uint32_t usbOverrun = 0;       // records dropped because the FIFO was full
uint32_t totalRecords = 0;

void pushHeader() {
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
  fifoPush(block, sizeof(block));   // 512 B always fits in a fresh FIFO
}

// Status sentinel record (imu_id 0xFF): carries device-side counters the host
// can't otherwise see (FIFO overrun). Host prints it and strips it from the file.
void pushStatusRecord() {
  ImuRecord r;
  memset(&r, 0, sizeof(r));
  r.imu_id        = STATUS_IMU_ID;
  r.isr_overflows = usbOverrun;     // FIFO-full drops so far
  r.t_cyc         = millis();       // device uptime (ms)
  r.dt_cyc        = totalRecords;   // data records produced so far
  fifoPush((const uint8_t *)&r, sizeof(r));
}

void resetForStream() {
  fifoHead = fifoTail = 0;
  usbOverrun = 0;
  totalRecords = 0;
  syncHead = syncTail = 0;
  syncIsrOverflow = 0;
  syncSeq = 0;
  syncHavePrev = false;
  for (uint8_t i = 0; i < MAX_IMUS; i++) {
    imuState[i].tsHead = 0;
    imuState[i].tsTail = 0;
    imuState[i].isrCount = 0;
    imuState[i].isrOverflowCount = 0;
    imuState[i].havePrev = false;
    imuState[i].prevCyc = 0;
  }
}

void startStreaming() {
  resetForStream();
  pushHeader();
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (imuState[i].present) writeReg(*IMUS[i].bus, IMUS[i].addr, REG_INT_ENABLE, 0x01);
  }
  attachInterrupt(digitalPinToInterrupt(SYNC_PIN), syncISR, RISING);
  streaming = true;
}

void stopStreaming() {
  detachInterrupt(digitalPinToInterrupt(SYNC_PIN));
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (imuState[i].present) writeReg(*IMUS[i].bus, IMUS[i].addr, REG_INT_ENABLE, 0x00);
  }
  streaming = false;
  // Best-effort flush of whatever is still buffered.
  uint32_t t0 = millis();
  while (fifoUsed() > 0 && millis() - t0 < 200) drainUsb();
}

// =============================================================================
// Drain camera-sync edges to the FIFO. Sync is NEVER dropped: if the FIFO is
// full we leave the edge in its queue and retry next loop (its timestamp value
// is already frozen, so waiting costs no accuracy). Drained before serviceImus()
// so sync gets first claim on FIFO space.
void serviceSync() {
  while (syncTail != syncHead) {
    uint32_t tCyc = syncQueue[syncTail];        // peek; commit tail only on success
    uint32_t dtCyc = syncHavePrev ? (tCyc - syncPrevCyc) : 0;

    ImuRecord rec;
    memset(&rec, 0, sizeof(rec));
    rec.imu_id        = SYNC_IMU_ID;
    rec.dropped_ts    = (uint16_t)syncIsrOverflow;
    rec.isr_overflows = syncSeq;
    rec.t_cyc         = tCyc;
    rec.dt_cyc        = dtCyc;

    if (fifoFree() < sizeof(rec)) break;        // FIFO full: keep the edge, retry later
    fifoPush((const uint8_t *)&rec, sizeof(rec));
    syncPrevCyc = tCyc;
    syncHavePrev = true;
    syncSeq++;
    syncTail = (syncTail + 1) & SYNC_QUEUE_MASK;
  }
}

// =============================================================================
void serviceImus() {
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    ImuState &st = imuState[i];
    if (!st.present) continue;

    uint32_t tCyc = 0;
    uint16_t dropped = 0;
    if (!popNewestTimestamp(st, tCyc, dropped)) continue;

    uint32_t dtCyc = 0;
    if (st.havePrev) dtCyc = tCyc - st.prevCyc;
    else st.havePrev = true;
    st.prevCyc = tCyc;

    uint32_t r0 = ARM_DWT_CYCCNT;
    ImuSample s;
    bool ok = readImu(*IMUS[i].bus, IMUS[i].addr, s);
    uint32_t rd = ARM_DWT_CYCCNT - r0;

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
    }
    rec.i2c_read_us = (uint16_t)cyclesToUs32(rd);

    if (fifoPush((const uint8_t *)&rec, sizeof(rec))) totalRecords++;
    else usbOverrun++;   // host/USB couldn't keep up; record dropped (counted)
  }
}

void setup() {
  Serial.begin(2000000);                 // baud ignored on Teensy USB CDC
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}

  enableCycleCounter();

  // Bring up each unique bus once.
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

  // Configure each IMU; attach ISRs; keep DATA_READY off until 'g'.
  uint8_t present = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    uint8_t who = 0;
    bool ok = readReg(*IMUS[i].bus, IMUS[i].addr, REG_WHO_AM_I, who);
    imuState[i].present = (ok && who == 0x68);
    if (!imuState[i].present) continue;
    configureMpu6050(*IMUS[i].bus, IMUS[i].addr);
    pinMode(IMUS[i].intPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(IMUS[i].intPin), ISR_TABLE[i], RISING);
    present++;
  }

  pinMode(SYNC_PIN, INPUT);   // camera-sync input (interrupt attached on 'g')

#if defined(__IMXRT1062__)
  NVIC_SET_PRIORITY(IRQ_GPIO6789, 16);
#endif

  Serial.print("# imu_usb_logger ready: ");
  Serial.print(present);
  Serial.println(" IMUs present. Send 'g' to start, 's' to stop.");
}

void loop() {
  if (Serial.available()) {
    int c = Serial.read();
    if (c == 'g' || c == 'G') { if (!streaming) startStreaming(); }
    else if (c == 's' || c == 'S') { if (streaming) stopStreaming(); }
  }

  if (!streaming) return;

  serviceSync();   // drained first: sync gets priority and is never dropped
  serviceImus();
  drainUsb();

  // ~1 Hz device-side status sentinel (FIFO overrun visibility for the host).
  static uint32_t lastStatusMs = 0;
  uint32_t now = millis();
  if (now - lastStatusMs >= 1000) {
    lastStatusMs = now;
    pushStatusRecord();
  }
}
