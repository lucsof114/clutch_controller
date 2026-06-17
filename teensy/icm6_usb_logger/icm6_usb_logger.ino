#include <Arduino.h>
#include <SPI.h>

// =============================================================================
// 6x ICM-42605 USB logger — shared camera/FSYNC edge timestamping
//
// The Teensy timestamps the shared 30 Hz camera/FSYNC edge with ARM_DWT_CYCCNT
// (logged as 0xF0 sync records). Each IMU receives the SAME edge on its FSYNC pin
// and emits FSYNC-tagged FIFO packets carrying the edge->sample delta. The host
// maps tagged IMU samples into Teensy time:
//
//     T_sample = T_edge[matched] + fsync_delta
//
// Ordinary 500 Hz samples are placed between FSYNC anchors by FIFO order / nominal
// sample index / the ICM TMST field. The FIFO watermark interrupt is ONLY a
// read-ready signal (and latency diagnostic) — never a sample timestamp.
//
// Explicit edge association: each tagged IMU record carries its per-IMU fsync_seq
// (isr_overflows), so the host maps IMU i / fsync_seq m -> sync record m and can
// detect (not silently slip) if an IMU is one edge out of phase.
//
// Pin layout (per IMU, 8 pads): VCC->3.3V GND->GND SCL->13 SDA->11 ADO->12
//   CS,INT1 per IMUS[] | INT2 = FSYNC <- the shared 30 Hz camera edge (USED).
//   shared SPI SCK=13 MOSI=11 MISO=12; same 30 Hz edge -> Teensy pin 17 (T_edge).
//   3.3V to every VDD+VDDIO, common GND, 100nF+2.2uF per IMU.
//
// Key config (from the debug saga): TMST_CONFIG MUST stay 0x23 (forcing it low
// poisons the FIFO to zeros). Host: imu_usb_recorder.py ('g'/'s'). ICM scale:
// accel /2048 g (±16 g), gyro /16.4 dps (±2000 dps).
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
static constexpr uint8_t REG_FIFO_CONFIG    = 0x16;
static constexpr uint8_t REG_INT_STATUS    = 0x2D;
static constexpr uint8_t REG_FIFO_DATA     = 0x30;
static constexpr uint8_t REG_SIGNAL_PATH_RESET = 0x4B;
static constexpr uint8_t REG_INTF_CONFIG0  = 0x4C;
static constexpr uint8_t REG_PWR_MGMT0     = 0x4E;
static constexpr uint8_t REG_GYRO_CONFIG0  = 0x4F;
static constexpr uint8_t REG_ACCEL_CONFIG0 = 0x50;
static constexpr uint8_t REG_TMST_CONFIG   = 0x54;
static constexpr uint8_t REG_FIFO_CONFIG1  = 0x5F;
static constexpr uint8_t REG_FIFO_CONFIG2  = 0x60;
static constexpr uint8_t REG_FIFO_CONFIG3  = 0x61;
static constexpr uint8_t REG_FSYNC_CONFIG  = 0x62;
static constexpr uint8_t REG_INT_CONFIG1   = 0x64;
static constexpr uint8_t REG_INT_SOURCE0   = 0x65;
static constexpr uint8_t REG_INTF_CONFIG5  = 0x7B;   // Bank 1: PIN9_FUNCTION
static constexpr uint8_t REG_WHO_AM_I      = 0x75;
static constexpr uint8_t REG_BANK_SEL      = 0x76;
static constexpr uint8_t ICM_WHOAMI        = 0x42;

static constexpr uint32_t CYC_PER_US = F_CPU_ACTUAL / 1000000;   // 600
static constexpr uint8_t  PKT = 16;                              // FIFO packet size
static constexpr uint16_t FIFO_WM_BYTES = 128;                   // watermark = 8 packets

static SPISettings ICM_SPI(1000000, MSBFIRST, SPI_MODE3);        // raise to 8-12 MHz with caps

// ---- 6-IMU table (CS pin, INT1 pin) ----
struct ImuConfig { uint8_t cs; uint8_t intPin; };
static const ImuConfig IMUS[] = {
  {8, 2}, {3, 9}, {4, 10},   // 3 IMUs wired (CS, INT1) per CS-scan; add more for full 6
};
static constexpr uint8_t MAX_IMUS = 6;
static constexpr uint8_t NUM_IMUS = sizeof(IMUS) / sizeof(IMUS[0]);
static_assert(NUM_IMUS <= MAX_IMUS, "increase MAX_IMUS / ISR_TABLE");
static constexpr uint8_t PIN_SYNC = 32;   // hardware camera-sync edge from the Arduino

// --- BENCH TEST ONLY: Teensy fakes the 30 Hz camera edge on PIN_SYNC_GEN.
// Jumper PIN_SYNC_GEN -> ICM INT2 (FSYNC) AND -> PIN_SYNC. For production,
// set TEST_SYNC_GEN 0 and drive PIN_SYNC + the FSYNC pins from the real camera trigger.
#define TEST_SYNC_GEN 0
static constexpr uint8_t PIN_SYNC_GEN = 5;
#if TEST_SYNC_GEN
IntervalTimer syncGen;
volatile bool syncGenLevel = false;
void syncGenISR() { syncGenLevel = !syncGenLevel; digitalWriteFast(PIN_SYNC_GEN, syncGenLevel); }  // 60 Hz toggle -> 30 Hz
#endif

struct ImuState {
  volatile uint32_t wmCyc;       // CYCCNT at watermark INT — DIAGNOSTIC ONLY (read-ready/latency)
  volatile bool     ready;       // batch waiting to be read
  bool     present;
  uint32_t sampleN;              // running sample index (nominal device coordinate, 2000 us/sample)
  uint32_t fsyncN;               // per-IMU FSYNC-tag counter -> explicit edge association
};
ImuState imuState[MAX_IMUS];

void enableCycleCounter() {
  ARM_DEMCR |= ARM_DEMCR_TRCENA;
  ARM_DWT_CYCCNT = 0;
  ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
}

// ---- Per-IMU FIFO-watermark ISR: flag "batch ready to read" (NOT a sample time) ----
template <uint8_t I>
void imuISR() { imuState[I].wmCyc = ARM_DWT_CYCCNT; imuState[I].ready = true; }
typedef void (*IsrFn)();
static const IsrFn ISR_TABLE[MAX_IMUS] = { imuISR<0>, imuISR<1>, imuISR<2>, imuISR<3>, imuISR<4>, imuISR<5> };

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
  digitalWriteFast(cs, LOW); SPI.transfer(reg & 0x7F); SPI.transfer(val); digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
}
uint8_t icmRead(uint8_t cs, uint8_t reg) {
  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(cs, LOW); SPI.transfer(reg | 0x80); uint8_t v = SPI.transfer(0); digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
  return v;
}

bool icmConfig(uint8_t cs) {
  icmWrite(cs, REG_BANK_SEL, 0);
  icmWrite(cs, REG_DEVICE_CONFIG, 0x01); delay(50);
  (void)icmRead(cs, REG_INT_STATUS);

  uint8_t who = 0; int good = 0;
  for (int i = 0; i < 40 && good < 2; i++) {
    icmWrite(cs, REG_BANK_SEL, 0);
    who = icmRead(cs, REG_WHO_AM_I);
    good = (who == ICM_WHOAMI) ? good + 1 : 0; delay(3);
  }
  if (who != ICM_WHOAMI) return false;

  icmWrite(cs, REG_INT_CONFIG,    0x03);   // INT1 active-high, push-pull, pulsed
  icmWrite(cs, REG_INT_CONFIG1,   0x00);   // clear INT_ASYNC_RESET
  icmWrite(cs, REG_GYRO_CONFIG0,  0x0F);   // ±2000 dps @ 500 Hz
  icmWrite(cs, REG_ACCEL_CONFIG0, 0x0F);   // ±16 g  @ 500 Hz
  icmWrite(cs, REG_PWR_MGMT0,     0x0F);   // gyro+accel low-noise on
  delay(45);
  icmWrite(cs, REG_INTF_CONFIG0,  0x30);   // count in BYTES, big-endian count+data
  icmWrite(cs, REG_TMST_CONFIG,   0x23);   // reset default — do NOT force low (poisons FIFO)

  // FSYNC: the shared 30 Hz camera edge drives this pin too -> the IMU tags the
  // coincident sample with the edge->sample delta (Layer 3 = the hard anchor).
  icmWrite(cs, REG_BANK_SEL, 1);
  icmWrite(cs, REG_INTF_CONFIG5, 0x02);    // PIN9_FUNCTION = FSYNC
  icmWrite(cs, REG_BANK_SEL, 0);
  icmWrite(cs, REG_FSYNC_CONFIG, 0x10);    // FSYNC_UI_SEL=001, rising polarity

  icmWrite(cs, REG_FIFO_CONFIG2,  (uint8_t)(FIFO_WM_BYTES & 0xFF));
  icmWrite(cs, REG_FIFO_CONFIG3,  (uint8_t)(FIFO_WM_BYTES >> 8));
  icmWrite(cs, REG_INT_SOURCE0,   0x04);   // FIFO_THS_INT1_EN -> INT1 (read-ready only)
  icmWrite(cs, REG_FIFO_CONFIG1,  0x0F);   // accel+gyro+temp+tmst_fsync
  icmWrite(cs, REG_FIFO_CONFIG,   0x00); delay(2);
  icmWrite(cs, REG_FIFO_CONFIG,   0x40);   // stream-to-FIFO
  icmWrite(cs, REG_SIGNAL_PATH_RESET, 0x02); delay(2);   // FIFO_FLUSH
  return true;
}

// ---- Binary format (identical to imu_usb_logger) ----
static constexpr char     FILE_MAGIC[4] = {'I', 'M', 'U', 'L'};
// v7 record-field meaning by imu_id (the 32-byte struct is reused for all types):
//   IMU (0..5):  t_cyc=nominal device us  dt_cyc=2000  isr_overflows=fsync_seq(when tagged)
//                i2c_read_us=raw ICM TMST (FSYNC delta if FLAG_FSYNC_TAGGED else ODR ts)
//                flags: bit1 tagged | bit2 FIFO_RESET (discontinuity) | bit3 BAD_HEADER (desync)
//   SYNC (0xF0): t_cyc=Teensy T_edge(CYCCNT, unwrap 64-bit)  dt_cyc=edge dt  isr_overflows=edge_seq
//   STATUS(0xFF):t_cyc=millis  dt_cyc=totalRecords  isr_overflows=usbOverrun
// Discontinuity (FLAG_FIFO_RESET/BAD_HEADER): host must NOT interpolate across it and must
// re-acquire the fsync<->edge mapping after (the ordinal can be off by lost tags). dropped_ts/
// dt_cyc carry the pre-break fsync_seq/sampleN; t_cyc carries the watermark CYCCNT (approx time).
static constexpr uint16_t FILE_VERSION  = 7;
static constexpr uint16_t HEADER_BLOCK  = 512;
static constexpr uint8_t  STATUS_IMU_ID = 0xFF;
static constexpr uint8_t  SYNC_IMU_ID   = 0xF0;

static constexpr uint8_t  FLAG_FSYNC_TAGGED = 0x02;
static constexpr uint8_t  FLAG_FIFO_RESET   = 0x04;   // FIFO overflow -> samples/tags lost
static constexpr uint8_t  FLAG_BAD_HEADER   = 0x08;   // FIFO desync -> batch abandoned + flushed
static constexpr uint8_t  FLAG_INVALID      = 0x01;   // STATUS record: recording is invalid (sticky)

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
static constexpr size_t USBFIFO_SIZE = 1u << 17;
static constexpr size_t USBFIFO_MASK = USBFIFO_SIZE - 1;
uint8_t  ufifo[USBFIFO_SIZE];
size_t   ufHead = 0, ufTail = 0;
static inline size_t ufUsed() { return (ufHead - ufTail) & USBFIFO_MASK; }
static inline size_t ufFree() { return USBFIFO_SIZE - 1 - ufUsed(); }
bool ufPush(const uint8_t *d, size_t n) {
  if (ufFree() < n) return false;
  for (size_t i = 0; i < n; i++) { ufifo[ufHead] = d[i]; ufHead = (ufHead + 1) & USBFIFO_MASK; }
  return true;
}
void drainUsb() {
  size_t used = ufUsed(); if (!used) return;
  int avail = Serial.availableForWrite(); if (avail <= 0) return;
  size_t n = used; if ((size_t)avail < n) n = avail;
  size_t contig = USBFIFO_SIZE - ufTail; if (n > contig) n = contig;
  Serial.write(&ufifo[ufTail], n);
  ufTail = (ufTail + n) & USBFIFO_MASK;
}

bool     streaming = false;
bool     recordingValid = true;          // cleared (sticky) on ANY IMU FIFO_RESET/BAD_HEADER
uint32_t usbOverrun = 0, totalRecords = 0;
uint8_t  batch[2048];      // per-IMU FIFO read buffer (one IMU at a time)
static inline int16_t be16(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }

void pushHeader() {
  uint8_t block[HEADER_BLOCK]; memset(block, 0, sizeof(block));
  FileHeader h; memcpy(h.magic, FILE_MAGIC, 4);
  h.version = FILE_VERSION; h.record_size = sizeof(ImuRecord); h.f_cpu = F_CPU_ACTUAL;
  h.sample_rate_hz = 500; h.num_imus = NUM_IMUS; h.gyro_fs = 0; h.accel_fs = 0; h.reserved = 0;
  memcpy(block, &h, sizeof(h));
  if (!ufPush(block, sizeof(block))) usbOverrun++;   // header at stream start (FIFO empty) — guard anyway
}
void pushStatusRecord() {
  ImuRecord r; memset(&r, 0, sizeof(r));
  r.imu_id = STATUS_IMU_ID; r.isr_overflows = usbOverrun; r.t_cyc = millis(); r.dt_cyc = totalRecords;
  r.flags = recordingValid ? 0 : FLAG_INVALID;   // sticky: any discontinuity invalidates the recording
  if (!ufPush((const uint8_t *)&r, sizeof(r))) usbOverrun++;
}
// Mark a timeline break for IMU i (FIFO overflow or header desync). The host must not
// interpolate across it and must re-acquire the fsync<->edge mapping afterward.
void pushDiscontinuity(uint8_t i, uint8_t flag) {
  recordingValid = false;                          // sticky -> whole recording is invalid
  ImuRecord r; memset(&r, 0, sizeof(r));
  r.imu_id = i; r.flags = flag;
  r.dropped_ts = (uint16_t)imuState[i].fsyncN;   // tags seen before the break
  r.isr_overflows = imuState[i].fsyncN;
  r.t_cyc = imuState[i].wmCyc;                    // approx Teensy time of the break (for re-acquire)
  r.dt_cyc = imuState[i].sampleN;                 // samples seen before the break
  if (!ufPush((const uint8_t *)&r, sizeof(r))) usbOverrun++;   // never silently lose a discontinuity
}
void resetForStream() {
  ufHead = ufTail = 0; usbOverrun = 0; totalRecords = 0;
  syncHead = syncTail = 0; syncIsrOverflow = 0; syncSeq = 0; syncHavePrev = false;
  for (uint8_t i = 0; i < MAX_IMUS; i++) { imuState[i].ready = false; imuState[i].sampleN = 0; imuState[i].fsyncN = 0; }
}
void startStreaming() {
  resetForStream(); pushHeader();
  // Sync edges MUST be logged before any IMU FSYNC tag can enter the stream, so the
  // fsync_seq <-> sync-record mapping starts aligned. Attach sync first, then flush
  // each IMU FIFO (clears pre-stream tags) and attach its watermark INT.
  attachInterrupt(digitalPinToInterrupt(PIN_SYNC), syncISR, RISING);
  syncHead = syncTail = 0; syncSeq = 0; syncHavePrev = false;
  for (uint8_t i = 0; i < NUM_IMUS; i++)
    if (imuState[i].present) {
      icmWrite(IMUS[i].cs, REG_SIGNAL_PATH_RESET, 0x02);   // FIFO_FLUSH: first tag = first post-flush edge
      imuState[i].sampleN = 0; imuState[i].fsyncN = 0;
      attachInterrupt(digitalPinToInterrupt(IMUS[i].intPin), ISR_TABLE[i], RISING);
    }
#if TEST_SYNC_GEN
  syncGenLevel = false; digitalWriteFast(PIN_SYNC_GEN, LOW);
  syncGen.begin(syncGenISR, 1000000.0 / 60.0);   // start fake sync AFTER setup -> first edge = seq 0
#endif
  streaming = true;
}
void stopStreaming() {
#if TEST_SYNC_GEN
  syncGen.end();
#endif
  detachInterrupt(digitalPinToInterrupt(PIN_SYNC));
  for (uint8_t i = 0; i < NUM_IMUS; i++)
    if (imuState[i].present) detachInterrupt(digitalPinToInterrupt(IMUS[i].intPin));
  streaming = false;
  uint32_t t0 = millis();
  while (ufUsed() > 0 && millis() - t0 < 200) drainUsb();
}

// Camera sync: drained first, never dropped.
void serviceSync() {
  while (syncTail != syncHead) {
    uint32_t tCyc = syncQueue[syncTail];
    uint32_t dtCyc = syncHavePrev ? (tCyc - syncPrevCyc) : 0;
    ImuRecord rec; memset(&rec, 0, sizeof(rec));
    rec.imu_id = SYNC_IMU_ID; rec.dropped_ts = (uint16_t)syncIsrOverflow;
    rec.isr_overflows = syncSeq; rec.t_cyc = tCyc; rec.dt_cyc = dtCyc;
    if (ufFree() < sizeof(rec)) break;
    ufPush((const uint8_t *)&rec, sizeof(rec));
    syncPrevCyc = tCyc; syncHavePrev = true; syncSeq++;
    syncTail = (syncTail + 1) & SYNC_QUEUE_MASK;
  }
}

// Read one IMU's FIFO batch and log the RAW INGREDIENTS for host alignment:
//   t_cyc        = NOMINAL device coordinate (sampleN * 2000 us/sample — the IMU runs on
//                  its own oscillator; the host corrects it locally via FSYNC anchors)
//   flags bit1   = FSYNC-tagged (this sample coincided with a camera edge)
//   isr_overflows= per-IMU fsync_seq (explicit edge association; valid when tagged)
//   i2c_read_us  = the raw ICM TMST field: FSYNC delta if tagged, else the ODR timestamp
// Host: T_sample = T_edge[fsync_seq] + delta for tagged samples; interpolate the rest
// by device coordinate between adjacent anchors. The watermark CYCCNT is NOT a timestamp.
void drainImuFifo(uint8_t i) {
  uint8_t cs = IMUS[i].cs;

  uint8_t s3[3];
  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(cs, LOW); SPI.transfer(REG_INT_STATUS | 0x80);
  s3[0] = SPI.transfer(0); s3[1] = SPI.transfer(0); s3[2] = SPI.transfer(0);
  digitalWriteFast(cs, HIGH); SPI.endTransaction();

  if (s3[0] & 0x02) {                                  // FIFO_FULL -> samples/tags lost
    pushDiscontinuity(i, FLAG_FIFO_RESET);
    icmWrite(cs, REG_SIGNAL_PATH_RESET, 0x02);
    return;
  }
  uint16_t count = ((uint16_t)s3[1] << 8) | s3[2];
  uint16_t nbytes = (count / PKT) * PKT;
  if (nbytes == 0) return;
  if (nbytes > sizeof(batch)) nbytes = (sizeof(batch) / PKT) * PKT;

  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(cs, LOW); SPI.transfer(REG_FIFO_DATA | 0x80);
  for (uint16_t j = 0; j < nbytes; j++) batch[j] = SPI.transfer(0xFF);
  digitalWriteFast(cs, HIGH); SPI.endTransaction();

  uint16_t N = nbytes / PKT;
  for (uint16_t k = 0; k < N; k++) {
    const uint8_t *p = &batch[k * PKT];
    if ((p[0] & 0xF0) != 0x60) {                          // FIFO desync -> flag + flush, abandon batch
      pushDiscontinuity(i, FLAG_BAD_HEADER);
      icmWrite(cs, REG_SIGNAL_PATH_RESET, 0x02);
      return;
    }
    uint8_t  tsType = (p[0] >> 2) & 0x03;                 // 0b11 = FSYNC-tagged
    uint16_t tsfield = (uint16_t)((p[14] << 8) | p[15]);  // ODR ts, or FSYNC delta if tagged

    uint32_t devUs = (++imuState[i].sampleN) * 2000UL;    // nominal device coordinate
    bool tagged = (tsType == 0x03);

    ImuRecord rec; rec.imu_id = i;
    rec.flags = tagged ? FLAG_FSYNC_TAGGED : 0x00;
    rec.dropped_ts = 0;
    rec.isr_overflows = imuState[i].fsyncN;               // fsync_seq (valid when tagged)
    if (tagged) imuState[i].fsyncN++;
    rec.t_cyc = devUs;                                    // nominal device coordinate (host maps)
    rec.dt_cyc = 2000;
    rec.ax = be16(&p[1]); rec.ay = be16(&p[3]); rec.az = be16(&p[5]);
    rec.gx = be16(&p[7]); rec.gy = be16(&p[9]); rec.gz = be16(&p[11]);
    rec.temp = (int8_t)p[13];
    rec.i2c_read_us = tsfield;                            // raw ICM TMST: delta if tagged, else ODR ts
    if (ufPush((const uint8_t *)&rec, sizeof(rec))) totalRecords++;
    else usbOverrun++;
  }
}

void serviceImus() {
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    if (!imuState[i].present || !imuState[i].ready) continue;
    imuState[i].ready = false;
    drainImuFifo(i);
  }
}

void setup() {
  Serial.begin(2000000);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}
  enableCycleCounter();

  // Drive every non-SPI pin to a defined level, then bring up SPI exactly ONCE and let
  // it settle before any transaction. Floating inputs next to SCK/MISO on a breadboard,
  // and (critically) re-calling SPI.begin() after pins are configured, both corrupt the
  // first reads — that was the "WHO_AM_I reads 0x44" bug. One clean SPI.begin + settle.
  for (uint8_t p = 0; p <= 23; p++) { if (p == 11 || p == 12 || p == 13) continue; pinMode(p, OUTPUT); digitalWriteFast(p, HIGH); }
  SPI.begin();
  delay(50);

  uint8_t present = 0;
  for (uint8_t i = 0; i < NUM_IMUS; i++) {
    imuState[i].present = icmConfig(IMUS[i].cs);
    if (imuState[i].present) { pinMode(IMUS[i].intPin, INPUT); present++; }
    else { Serial.print("# IMU "); Serial.print(i); Serial.println(" not responding (WHO_AM_I)"); }
  }
  pinMode(PIN_SYNC, INPUT);   // restore the sync pin as an input (Arduino drives the edge)
#if defined(__IMXRT1062__)
  NVIC_SET_PRIORITY(IRQ_GPIO6789, 16);
#endif

  Serial.print("# icm6_usb_logger (FSYNC-anchored) ready: "); Serial.print(present);
  Serial.println("/6 ICM-42605 present. Send 'g' to start, 's' to stop.");
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
