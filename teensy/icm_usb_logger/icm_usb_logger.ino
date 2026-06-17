#include <Arduino.h>
#include <SPI.h>

// =============================================================================
// 3x ICM-42605 USB logger with shared-clock time sync (Teensy 4.1)
//
// Same streaming/format as imu_usb_logger, but: SPI instead of I2C, 3 ICM-42605s
// at 500 Hz via direct register reads, and the camera sync timestamped on the
// Teensy cycle counter (same clock as every IMU sample).
//
// Timestamping: each IMU's INT1 data-ready edge -> ISR latches ARM_DWT_CYCCNT
// (per-IMU queue). The camera sync edge -> ISR latches CYCCNT (never-dropped
// queue). Frames and all 3 IMUs land on one timebase by construction.
//
// Pin map:
//   SPI: SCLK=13, MOSI=11, MISO=12   (all 3 IMUs in parallel)
//   IMU0 CS=10 INT1=7 | IMU1 CS=9 INT1=6 | IMU2 CS=8 INT1=5
//   Camera sync (5V -> 10k/15k divider -> 3.0V) = pin 2
//   3.3V to VDD+VDDIO, common GND, decouple at each package.
//
// Host: imu_usb_recorder.py drives 'g'/'s' and saves the .bin (same format).
// ICM scale: accel /2048 g (±16 g), gyro /16.4 dps (±2000 dps).
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

// ---- ICM-42605 registers (Bank 0) ----
static constexpr uint8_t REG_DEVICE_CONFIG = 0x11;
static constexpr uint8_t REG_INT_CONFIG    = 0x14;
static constexpr uint8_t REG_TEMP_DATA1    = 0x1D;   // temp(2)+accel(6)+gyro(6) = 14 bytes
static constexpr uint8_t REG_INT_STATUS    = 0x2D;
static constexpr uint8_t REG_PWR_MGMT0     = 0x4E;
static constexpr uint8_t REG_GYRO_CONFIG0  = 0x4F;
static constexpr uint8_t REG_ACCEL_CONFIG0 = 0x50;
static constexpr uint8_t REG_INT_CONFIG1   = 0x64;
static constexpr uint8_t REG_INT_SOURCE0   = 0x65;
static constexpr uint8_t REG_WHO_AM_I      = 0x75;
static constexpr uint8_t REG_BANK_SEL      = 0x76;
static constexpr uint8_t ICM_WHOAMI        = 0x42;

static SPISettings ICM_SPI(1000000, MSBFIRST, SPI_MODE3);

// ---- IMU table (CS pin, INT1 pin) ----
struct ImuConfig { uint8_t cs; uint8_t intPin; };
static const ImuConfig IMUS[] = {
  {10, 7},   // IMU 0
  { 9, 6},   // IMU 1
  { 8, 5},   // IMU 2
};
static constexpr uint8_t MAX_IMUS = 3;
static constexpr uint8_t NUM_IMUS = sizeof(IMUS) / sizeof(IMUS[0]);
static_assert(NUM_IMUS <= MAX_IMUS, "increase MAX_IMUS / ISR_TABLE");

static constexpr uint8_t PIN_SYNC = 2;   // camera sync (via 5V->3V divider)

// ---- Per-IMU timestamp queue ----
static constexpr uint16_t TS_QUEUE_SIZE = 256;
static constexpr uint16_t TS_QUEUE_MASK = TS_QUEUE_SIZE - 1;

struct ImuState {
  volatile uint32_t tsQueue[TS_QUEUE_SIZE];
  volatile uint16_t tsHead, tsTail;
  volatile uint32_t isrOverflowCount;
  bool     present;
  bool     havePrev;
  uint32_t prevCyc;
};
ImuState imuState[MAX_IMUS];

void enableCycleCounter() {
  ARM_DEMCR |= ARM_DEMCR_TRCENA;
  ARM_DWT_CYCCNT = 0;
  ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
}
static inline uint32_t cyclesToUs32(uint32_t c) {
  return (uint32_t)(((uint64_t)c * 1000000ULL) / (uint64_t)F_CPU_ACTUAL);
}

// ---- Per-IMU INT1 ISR (timestamp only) ----
template <uint8_t I>
void imuISR() {
  uint32_t t = ARM_DWT_CYCCNT;
  ImuState &st = imuState[I];
  uint16_t head = st.tsHead, next = (head + 1) & TS_QUEUE_MASK;
  if (next != st.tsTail) { st.tsQueue[head] = t; st.tsHead = next; }
  else { st.isrOverflowCount++; }
}
typedef void (*IsrFn)();
static const IsrFn ISR_TABLE[MAX_IMUS] = { imuISR<0>, imuISR<1>, imuISR<2> };

bool popNewestTimestamp(ImuState &st, uint32_t &t_cyc, uint16_t &dropped) {
  uint16_t head = st.tsHead, tail = st.tsTail;
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

// ---- Camera sync queue (never drop-stale) ----
static constexpr uint16_t SYNC_QUEUE_SIZE = 512;
static constexpr uint16_t SYNC_QUEUE_MASK = SYNC_QUEUE_SIZE - 1;
volatile uint32_t syncQueue[SYNC_QUEUE_SIZE];
volatile uint16_t syncHead = 0, syncTail = 0;
volatile uint32_t syncIsrOverflow = 0;
uint32_t syncSeq = 0, syncPrevCyc = 0;
bool     syncHavePrev = false;

void syncISR() {
  uint32_t t = ARM_DWT_CYCCNT;
  uint16_t head = syncHead, next = (head + 1) & SYNC_QUEUE_MASK;
  if (next != syncTail) { syncQueue[head] = t; syncHead = next; }
  else { syncIsrOverflow++; }
}

// ---- SPI helpers (manual CS) ----
void icmWrite(uint8_t cs, uint8_t reg, uint8_t val) {
  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(cs, LOW);
  SPI.transfer(reg & 0x7F);
  SPI.transfer(val);
  digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
}
uint8_t icmRead(uint8_t cs, uint8_t reg) {
  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(cs, LOW);
  SPI.transfer(reg | 0x80);
  uint8_t v = SPI.transfer(0x00);
  digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
  return v;
}
void icmReadBurst(uint8_t cs, uint8_t reg, uint8_t *buf, size_t n) {
  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(cs, LOW);
  SPI.transfer(reg | 0x80);
  for (size_t i = 0; i < n; i++) buf[i] = SPI.transfer(0x00);
  digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
}

bool icmConfig(uint8_t cs) {
  icmWrite(cs, REG_BANK_SEL, 0);
  icmWrite(cs, REG_DEVICE_CONFIG, 0x01);   // soft reset
  delay(50);
  (void)icmRead(cs, REG_INT_STATUS);

  uint8_t who = 0; int good = 0;
  for (int i = 0; i < 40 && good < 2; i++) {
    icmWrite(cs, REG_BANK_SEL, 0);
    who = icmRead(cs, REG_WHO_AM_I);
    good = (who == ICM_WHOAMI) ? good + 1 : 0;
    delay(3);
  }
  if (who != ICM_WHOAMI) return false;

  icmWrite(cs, REG_INT_CONFIG,    0x03);   // INT1 active-high, push-pull, pulsed
  icmWrite(cs, REG_INT_CONFIG1,   0x00);   // clear INT_ASYNC_RESET
  icmWrite(cs, REG_GYRO_CONFIG0,  0x0F);   // ±2000 dps @ 500 Hz
  icmWrite(cs, REG_ACCEL_CONFIG0, 0x0F);   // ±16 g  @ 500 Hz
  icmWrite(cs, REG_PWR_MGMT0,     0x0F);   // gyro+accel low-noise on
  delayMicroseconds(300);
  icmWrite(cs, REG_INT_SOURCE0,   0x08);   // UI data-ready -> INT1
  return true;
}

// ---- Binary format (identical to imu_usb_logger) ----
static constexpr char     FILE_MAGIC[4] = {'I', 'M', 'U', 'L'};
static constexpr uint16_t FILE_VERSION  = 3;        // v3 = 3x ICM-42605 over SPI
static constexpr uint16_t HEADER_BLOCK  = 512;
static constexpr uint8_t  STATUS_IMU_ID = 0xFF;
static constexpr uint8_t  SYNC_IMU_ID   = 0xF0;

struct __attribute__((packed)) ImuRecord {
  uint8_t  imu_id; uint8_t flags; uint16_t dropped_ts; uint32_t isr_overflows;
  uint32_t t_cyc;  uint32_t dt_cyc;
  int16_t  ax, ay, az, gx, gy, gz, temp; uint16_t i2c_read_us;
};
static_assert(sizeof(ImuRecord) == 32, "ImuRecord must be 32 bytes");

struct __attribute__((packed)) FileHeader {
  char magic[4]; uint16_t version; uint16_t record_size; uint32_t f_cpu;
  uint16_t sample_rate_hz; uint8_t num_imus; uint8_t gyro_fs; uint8_t accel_fs; uint8_t reserved;
};

// ---- RAM FIFO -> USB ----
static constexpr size_t FIFO_SIZE = 1u << 17;
static constexpr size_t FIFO_MASK = FIFO_SIZE - 1;
uint8_t  fifo[FIFO_SIZE];
size_t   fifoHead = 0, fifoTail = 0;
static inline size_t fifoUsed() { return (fifoHead - fifoTail) & FIFO_MASK; }
static inline size_t fifoFree() { return FIFO_SIZE - 1 - fifoUsed(); }
bool fifoPush(const uint8_t *d, size_t n) {
  if (fifoFree() < n) return false;
  for (size_t i = 0; i < n; i++) { fifo[fifoHead] = d[i]; fifoHead = (fifoHead + 1) & FIFO_MASK; }
  return true;
}
void drainUsb() {
  size_t used = fifoUsed(); if (!used) return;
  int avail = Serial.availableForWrite(); if (avail <= 0) return;
  size_t n = used; if ((size_t)avail < n) n = avail;
  size_t contig = FIFO_SIZE - fifoTail; if (n > contig) n = contig;
  Serial.write(&fifo[fifoTail], n);
  fifoTail = (fifoTail + n) & FIFO_MASK;
}

bool     streaming = false;
uint32_t usbOverrun = 0, totalRecords = 0;
static inline int16_t be16(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }

void pushHeader() {
  uint8_t block[HEADER_BLOCK]; memset(block, 0, sizeof(block));
  FileHeader h; memcpy(h.magic, FILE_MAGIC, 4);
  h.version = FILE_VERSION; h.record_size = sizeof(ImuRecord); h.f_cpu = F_CPU_ACTUAL;
  h.sample_rate_hz = 500; h.num_imus = NUM_IMUS; h.gyro_fs = 0; h.accel_fs = 0; h.reserved = 0;
  memcpy(block, &h, sizeof(h));
  fifoPush(block, sizeof(block));
}
void pushStatusRecord() {
  ImuRecord r; memset(&r, 0, sizeof(r));
  r.imu_id = STATUS_IMU_ID; r.isr_overflows = usbOverrun; r.t_cyc = millis(); r.dt_cyc = totalRecords;
  fifoPush((const uint8_t *)&r, sizeof(r));
}
void resetForStream() {
  fifoHead = fifoTail = 0; usbOverrun = 0; totalRecords = 0;
  syncHead = syncTail = 0; syncIsrOverflow = 0; syncSeq = 0; syncHavePrev = false;
  for (uint8_t i = 0; i < MAX_IMUS; i++) {
    imuState[i].tsHead = imuState[i].tsTail = 0;
    imuState[i].isrOverflowCount = 0; imuState[i].havePrev = false;
  }
}
void startStreaming() {
  resetForStream(); pushHeader();
  for (uint8_t i = 0; i < NUM_IMUS; i++)
    if (imuState[i].present)
      attachInterrupt(digitalPinToInterrupt(IMUS[i].intPin), ISR_TABLE[i], RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_SYNC), syncISR, RISING);
  streaming = true;
}
void stopStreaming() {
  detachInterrupt(digitalPinToInterrupt(PIN_SYNC));
  for (uint8_t i = 0; i < NUM_IMUS; i++)
    if (imuState[i].present) detachInterrupt(digitalPinToInterrupt(IMUS[i].intPin));
  streaming = false;
  uint32_t t0 = millis();
  while (fifoUsed() > 0 && millis() - t0 < 200) drainUsb();
}

// Camera sync: drained first, never dropped.
void serviceSync() {
  while (syncTail != syncHead) {
    uint32_t tCyc = syncQueue[syncTail];
    uint32_t dtCyc = syncHavePrev ? (tCyc - syncPrevCyc) : 0;
    ImuRecord rec; memset(&rec, 0, sizeof(rec));
    rec.imu_id = SYNC_IMU_ID; rec.dropped_ts = (uint16_t)syncIsrOverflow;
    rec.isr_overflows = syncSeq; rec.t_cyc = tCyc; rec.dt_cyc = dtCyc;
    if (fifoFree() < sizeof(rec)) break;
    fifoPush((const uint8_t *)&rec, sizeof(rec));
    syncPrevCyc = tCyc; syncHavePrev = true; syncSeq++;
    syncTail = (syncTail + 1) & SYNC_QUEUE_MASK;
  }
}

void serviceImus() {
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    ImuState &st = imuState[i];
    if (!st.present) continue;
    uint32_t tCyc; uint16_t dropped = 0;
    if (!popNewestTimestamp(st, tCyc, dropped)) continue;

    uint32_t dtCyc = 0;
    if (st.havePrev) dtCyc = tCyc - st.prevCyc; else st.havePrev = true;
    st.prevCyc = tCyc;

    uint8_t b[14];
    uint32_t r0 = ARM_DWT_CYCCNT;
    icmReadBurst(IMUS[i].cs, REG_TEMP_DATA1, b, 14);   // temp(2)+accel(6)+gyro(6)
    uint32_t rd = ARM_DWT_CYCCNT - r0;

    ImuRecord rec;
    rec.imu_id = i; rec.flags = 0; rec.dropped_ts = dropped;
    rec.isr_overflows = st.isrOverflowCount; rec.t_cyc = tCyc; rec.dt_cyc = dtCyc;
    rec.temp = be16(&b[0]);
    rec.ax = be16(&b[2]); rec.ay = be16(&b[4]); rec.az = be16(&b[6]);
    rec.gx = be16(&b[8]); rec.gy = be16(&b[10]); rec.gz = be16(&b[12]);
    rec.i2c_read_us = (uint16_t)cyclesToUs32(rd);

    if (fifoPush((const uint8_t *)&rec, sizeof(rec))) totalRecords++;
    else usbOverrun++;
  }
}

void setup() {
  Serial.begin(2000000);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}
  enableCycleCounter();

  for (uint8_t i = 0; i < NUM_IMUS; i++) { pinMode(IMUS[i].cs, OUTPUT); digitalWriteFast(IMUS[i].cs, HIGH); }
  pinMode(PIN_SYNC, INPUT);
  SPI.begin();

  uint8_t present = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    imuState[i].present = icmConfig(IMUS[i].cs);
    if (imuState[i].present) { pinMode(IMUS[i].intPin, INPUT); present++; }
    else { Serial.print("# IMU "); Serial.print(i); Serial.println(" not responding (WHO_AM_I)"); }
  }
#if defined(__IMXRT1062__)
  NVIC_SET_PRIORITY(IRQ_GPIO6789, 16);
#endif

  Serial.print("# icm_usb_logger ready: "); Serial.print(present);
  Serial.println("/3 ICM-42605 present. Send 'g' to start, 's' to stop.");
}

void loop() {
  if (Serial.available()) {
    int c = Serial.read();
    if (c == 'g' || c == 'G') { if (!streaming) startStreaming(); }
    else if (c == 's' || c == 'S') { if (streaming) stopStreaming(); }
  }
  if (!streaming) return;

  serviceSync();
  serviceImus();
  drainUsb();

  static uint32_t lastStatusMs = 0;
  uint32_t now = millis();
  if (now - lastStatusMs >= 1000) { lastStatusMs = now; pushStatusRecord(); }
}
