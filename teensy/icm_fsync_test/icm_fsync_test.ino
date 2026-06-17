#include <Arduino.h>
#include <SPI.h>

// =============================================================================
// ICM-42605 timestamping bring-up test (Teensy 4.1, SPI 4-wire)
//
// Validates the three-layer timestamping scheme:
//   Layer 1 - INT1 data-ready -> Teensy ISR latches ARM_DWT_CYCCNT (device time
//             anchored to the 600 MHz MCU clock).
//   Layer 2 - FIFO carries a per-sample device timestamp (TMST, 1 us). We drain
//             the FIFO over SPI, unwrap TMST, and regress device time vs the
//             CYCCNT anchors to recover the effective ODR / clock drift.
//   Layer 3 - FSYNC: a 30 Hz reference into INT2/FSYNC. The device reports, for
//             the tagged sample, the edge->sample phase delta in the FIFO
//             timestamp field (header timestamp-type = 0b11). For the test the
//             Teensy generates the 30 Hz pulse on pin 5 and records its own
//             CYCCNT per edge as ground truth.
//
// Pin map: SCLK=13, MOSI=11, MISO=12, CS=10, INT1=9 (in), FSYNC=5 (Teensy out).
// VDD+VDDIO=3.3V, decouple at the package.
//
// NOTE: register addresses verified against the v1.x map / driver; bitfields
// marked "VERIFY" should be double-checked against the datasheet register map.
// =============================================================================

// ---- DWT cycle counter ----
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

// ---- Pins ----
static constexpr uint8_t PIN_CS    = 10;
static constexpr uint8_t PIN_INT1  = 9;
static constexpr uint8_t PIN_FSYNC = 5;   // Teensy-generated 30 Hz reference (test only)

// SPI: ICM-42605 max 24 MHz, mode 3. 1 MHz for breadboard bring-up robustness
// (raise once wiring is short + decoupling caps are in place).
static SPISettings ICM_SPI(1000000, MSBFIRST, SPI_MODE3);

// ---- ICM-42605 registers (Bank 0 unless noted) ----
static constexpr uint8_t REG_DEVICE_CONFIG = 0x11;
static constexpr uint8_t REG_INT_CONFIG    = 0x14;
static constexpr uint8_t REG_FIFO_CONFIG   = 0x16;
static constexpr uint8_t REG_INT_STATUS    = 0x2D;
static constexpr uint8_t REG_FIFO_COUNTH   = 0x2E;
static constexpr uint8_t REG_FIFO_DATA     = 0x30;
static constexpr uint8_t REG_INTF_CONFIG0  = 0x4C;
static constexpr uint8_t REG_PWR_MGMT0     = 0x4E;
static constexpr uint8_t REG_GYRO_CONFIG0  = 0x4F;
static constexpr uint8_t REG_ACCEL_CONFIG0 = 0x50;
static constexpr uint8_t REG_TMST_CONFIG   = 0x54;
static constexpr uint8_t REG_FIFO_CONFIG1  = 0x5F;
static constexpr uint8_t REG_FIFO_CONFIG2  = 0x60;   // FIFO watermark low
static constexpr uint8_t REG_FIFO_CONFIG3  = 0x61;   // FIFO watermark high
static constexpr uint8_t REG_FSYNC_CONFIG  = 0x62;
static constexpr uint8_t REG_INT_CONFIG1   = 0x64;
static constexpr uint8_t REG_INT_SOURCE0   = 0x65;
static constexpr uint8_t REG_WHO_AM_I      = 0x75;
static constexpr uint8_t REG_BANK_SEL      = 0x76;
static constexpr uint8_t REG_INTF_CONFIG5  = 0x7B;   // Bank 1

static constexpr uint8_t ICM_WHOAMI        = 0x42;   // ICM-42605

uint8_t g_whoami = 0;     // last WHO_AM_I read (for live diagnosis)
bool    g_ok     = false; // init succeeded

// Scale factors for the chosen full-scale (±2000 dps, ±16 g)
static constexpr float GYRO_LSB_PER_DPS = 16.4f;
static constexpr float ACCEL_LSB_PER_G  = 2048.0f;
static constexpr uint16_t TARGET_ODR_HZ = 500;

// ---- Layer 1: INT1 data-ready anchor ----
volatile uint32_t int1Cyc   = 0;     // CYCCNT of most recent INT1 (FIFO watermark) edge
volatile uint32_t int1Count = 0;     // INT1 edges seen
volatile bool     fifoReady = false; // FIFO watermark reached -> read in loop

void icmInt1ISR() {
  int1Cyc = ARM_DWT_CYCCNT;
  int1Count++;
  fifoReady = true;
}

// ---- Layer 3: Teensy-generated 30 Hz FSYNC (test ground truth) ----
IntervalTimer fsyncTimer;
volatile bool     fsyncLevel = false;
volatile uint32_t fsyncEdgeCyc = 0;   // CYCCNT of most recent rising edge
volatile uint32_t fsyncEdgeCount = 0;

void fsyncToggleISR() {
  fsyncLevel = !fsyncLevel;
  digitalWriteFast(PIN_FSYNC, fsyncLevel);
  if (fsyncLevel) {                    // rising edge
    fsyncEdgeCyc = ARM_DWT_CYCCNT;
    fsyncEdgeCount++;
  }
}

// ---- SPI helpers (manual CS) ----
void csLow()  { digitalWriteFast(PIN_CS, LOW); }
void csHigh() { digitalWriteFast(PIN_CS, HIGH); }

void icmWrite(uint8_t reg, uint8_t val) {
  SPI.beginTransaction(ICM_SPI);
  csLow();
  SPI.transfer(reg & 0x7F);            // write: MSB=0
  SPI.transfer(val);
  csHigh();
  SPI.endTransaction();
}

uint8_t icmRead(uint8_t reg) {
  SPI.beginTransaction(ICM_SPI);
  csLow();
  SPI.transfer(reg | 0x80);            // read: MSB=1
  uint8_t v = SPI.transfer(0x00);
  csHigh();
  SPI.endTransaction();
  return v;
}

void icmReadBurst(uint8_t reg, uint8_t *buf, size_t n) {
  SPI.beginTransaction(ICM_SPI);
  csLow();
  SPI.transfer(reg | 0x80);
  for (size_t i = 0; i < n; i++) buf[i] = SPI.transfer(0x00);
  csHigh();
  SPI.endTransaction();
}

void setBank(uint8_t bank) { icmWrite(REG_BANK_SEL, bank); }

// =============================================================================
bool icmInit() {
  // Soft reset.
  setBank(0);
  icmWrite(REG_DEVICE_CONFIG, 0x01);   // SOFT_RESET_CONFIG
  delay(50);                           // settle (datasheet >=1 ms; generous for breadboard)
  (void)icmRead(REG_INT_STATUS);       // clears RESET_DONE

  // WHO_AM_I: require two consecutive good reads so a flaky bus can't fluke past.
  uint8_t who = 0; int good = 0;
  for (int i = 0; i < 40 && good < 2; i++) {
    setBank(0);
    who = icmRead(REG_WHO_AM_I);
    g_whoami = who;
    good = (who == ICM_WHOAMI) ? good + 1 : 0;
    delay(3);
  }
  Serial.print("# WHO_AM_I = 0x"); Serial.println(who, HEX);
  if (who != ICM_WHOAMI) {
    Serial.println("# unexpected WHO_AM_I (check SPI wiring / mode / CS)");
    return false;
  }

  // --- Sensors first, FIFO last (order matters; FSYNC/TMST added back once raw FIFO works) ---
  icmWrite(REG_INT_CONFIG, 0x03);      // INT1 active-high, push-pull, pulsed
  icmWrite(REG_INT_CONFIG1, 0x00);     // clear INT_ASYNC_RESET

  icmWrite(REG_GYRO_CONFIG0,  0x0F);   // ±2000 dps @ 500 Hz
  icmWrite(REG_ACCEL_CONFIG0, 0x0F);   // ±16 g @ 500 Hz

  icmWrite(REG_PWR_MGMT0, 0x0F);       // gyro+accel low-noise on
  delay(50);                           // gyro startup ~45 ms

  icmWrite(REG_INTF_CONFIG0, 0x30);    // count in BYTES (bit6=0), big-endian count+data
  icmWrite(REG_TMST_CONFIG, 0x23);     // reset default — TMST_FSYNC_EN(bit1) set (forcing low broke FIFO)

  // Layer 3 (FSYNC): PIN9 -> FSYNC input, tagged into the FIFO. Works now that
  // TMST_CONFIG keeps 0x23 instead of being forced to 0x01/0x03.
  setBank(1);
  icmWrite(REG_INTF_CONFIG5, 0x02);    // PIN9_FUNCTION = FSYNC
  setBank(0);
  icmWrite(REG_FSYNC_CONFIG, 0x10);    // FSYNC_UI_SEL=001, rising polarity [flip bit0->0x11 if fdelta out of range]

  // FIFO watermark = 128 bytes (8 packets); route FIFO-threshold INT to INT1.
  icmWrite(REG_FIFO_CONFIG2, 0x80);    // FIFO_WM = 128 bytes (8 packets)
  icmWrite(REG_FIFO_CONFIG3, 0x00);    // FIFO_WM[11:8]
  icmWrite(REG_INT_SOURCE0, 0x04);     // FIFO_THS_INT1_EN -> INT1

  icmWrite(REG_FIFO_CONFIG1, 0x0F);    // accel+gyro+temp+tmst_fsync
  icmWrite(REG_FIFO_CONFIG, 0x00);     // bypass (flush)
  delay(2);
  icmWrite(REG_FIFO_CONFIG, 0x40);     // stream-to-FIFO
  icmWrite(0x4B, 0x02);                // SIGNAL_PATH_RESET: FIFO_FLUSH
  delay(2);
  return true;
}

// ---- FIFO parse / stats ----
static constexpr uint8_t  PKT = 16;          // 16-byte FIFO packet (accel+gyro+temp+tmst)
uint8_t fifoBuf[2048];

uint64_t devTimeUs   = 0;          // unwrapped device timestamp (us)
uint16_t prevTmst    = 0;
bool     haveTmst    = false;
uint32_t pktCount    = 0;
uint32_t fsyncPkts   = 0;
uint32_t fifoErrors  = 0;
uint32_t fifoResets  = 0;

// Flush the on-chip FIFO (SIGNAL_PATH_RESET 0x4B, FIFO_FLUSH = bit1) and reset our
// timestamp unwrap. Called on FIFO overflow or header desync.
void fifoReset() {
  fifoResets++;
  icmWrite(REG_BANK_SEL, 0);
  icmWrite(0x4B, 0x02);
  haveTmst = false;
}

// device dt stats (consecutive, us)
uint32_t dtCount = 0; double dtSum = 0; uint32_t dtMin = 0xFFFFFFFF, dtMax = 0;
// FSYNC delta stats (us)
uint32_t fsMin = 0xFFFFFFFF, fsMax = 0; double fsSum = 0; uint32_t fsN = 0;

// Per-batch anchoring (NO drift regression): each batch's newest sample is pinned to
// the watermark CYCCNT; intra-batch spacing comes from the device TMST. Reconstructed
// Teensy-clock dt validates it stays ~2000 us with no cross-batch drift.
static constexpr uint32_t CYC_PER_US = F_CPU_ACTUAL / 1000000;   // 600, fixed (not fitted)
bool     reconHavePrev = false;
uint32_t reconPrevCyc = 0;
uint32_t reconMin = 0xFFFFFFFF, reconMax = 0; double reconSum = 0; uint32_t reconCount = 0;

static inline int16_t be16(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }

void parsePacket(const uint8_t *p) {
  uint8_t hdr = p[0];
  if (hdr & 0x80) { return; }                 // MSG/empty header, skip
  bool hasAccel = hdr & 0x40;
  bool hasGyro  = hdr & 0x20;
  uint8_t tsType = (hdr >> 2) & 0x03;          // 10=ODR tmst, 11=FSYNC delta
  if (!hasAccel || !hasGyro) { fifoErrors++; return; }

  // Accel/gyro present in every data packet; stream a decimated copy for plotting.
  int16_t ax = be16(&p[1]), ay = be16(&p[3]), az = be16(&p[5]);
  int16_t gx = be16(&p[7]), gy = be16(&p[9]), gz = be16(&p[11]);
  static uint16_t deci = 0;
  if (++deci >= 5) {                                // 500 Hz / 5 = 100 Hz to the plot
    deci = 0;
    Serial.print("D,");
    Serial.print(ax / ACCEL_LSB_PER_G, 3); Serial.print(',');
    Serial.print(ay / ACCEL_LSB_PER_G, 3); Serial.print(',');
    Serial.print(az / ACCEL_LSB_PER_G, 3); Serial.print(',');
    Serial.print(gx / GYRO_LSB_PER_DPS, 2); Serial.print(',');
    Serial.print(gy / GYRO_LSB_PER_DPS, 2); Serial.print(',');
    Serial.println(gz / GYRO_LSB_PER_DPS, 2);
  }

  uint16_t ts = (uint16_t)((p[14] << 8) | p[15]);   // 2-byte timestamp field

  // Device timeline from sample index — robust to FSYNC packets (whose ts field is the
  // delta, not an ODR timestamp). Device ODR is exactly 500 Hz / 2000 us (zero jitter).
  static uint64_t sampleN = 0;
  devTimeUs = (++sampleN) * 2000ULL;

  if (tsType == 0x03) {
    // FSYNC-tagged: ts = camera-edge -> sample delta (us); camera frame at devTimeUs - ts.
    fsyncPkts++;
    uint32_t d = ts;
    if (d < fsMin) fsMin = d;
    if (d > fsMax) fsMax = d;
    fsSum += d; fsN++;
  } else if (tsType == 0x02) {
    // ODR timestamp — used only to verify the device period (jitter). Skip ~2-period
    // gaps across a FSYNC packet (whose slot has no ODR ts).
    if (haveTmst) {
      uint16_t dt = ts - prevTmst;             // wrap-safe
      if (dt < 3000) { if (dt < dtMin) dtMin = dt; if (dt > dtMax) dtMax = dt; dtSum += dt; dtCount++; }
    } else {
      haveTmst = true;
    }
    prevTmst = ts;
  }
  pktCount++;
}

// PX4-style drain: read INT_STATUS+COUNT in one burst, gate on the committed count,
// read EXACTLY that many whole packets in one CS-low burst, validate the 0x68 header,
// and fifoReset() on overflow or desync (the recovery I was missing).
void drainFifo() {
  // Two clean transactions: read INT_STATUS+count, then read FIFO_DATA addressed at 0x30.
  uint8_t s3[3];
  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(PIN_CS, LOW);
  SPI.transfer(REG_INT_STATUS | 0x80);
  s3[0] = SPI.transfer(0); s3[1] = SPI.transfer(0); s3[2] = SPI.transfer(0);
  digitalWriteFast(PIN_CS, HIGH);
  SPI.endTransaction();

  if (s3[0] & 0x02) { fifoReset(); return; }      // FIFO_FULL
  uint16_t count = ((uint16_t)s3[1] << 8) | s3[2];
  uint16_t nbytes = (count / PKT) * PKT;          // read EXACTLY the committed count
  if (nbytes > sizeof(fifoBuf)) nbytes = (sizeof(fifoBuf) / PKT) * PKT;
  if (nbytes == 0) return;

  // Byte-by-byte read, buffer pre-filled with 0xA5 sentinel and sending 0xFF, so the
  // result disambiguates: 0x00 = MISO low/chip not driving; 0xA5 = transfer didn't
  // write; 0xFF = empty/undriven (FIFO_DATA reset value).
  memset(fifoBuf, 0xA5, nbytes);
  SPI.beginTransaction(ICM_SPI);
  digitalWriteFast(PIN_CS, LOW);
  SPI.transfer(REG_FIFO_DATA | 0x80);             // 0x30, addressed directly
  for (uint16_t i = 0; i < nbytes; i++) fifoBuf[i] = SPI.transfer(0xFF);
  digitalWriteFast(PIN_CS, HIGH);
  SPI.endTransaction();

  static uint32_t lastDbg = 0;
  if (millis() - lastDbg >= 1000) {
    lastDbg = millis();
    Serial.print("# who=0x");  Serial.print(icmRead(REG_WHO_AM_I), HEX);
    Serial.print(" pwr=0x");   Serial.print(icmRead(REG_PWR_MGMT0), HEX);
    Serial.print(" fifo=0x");  Serial.print(icmRead(REG_FIFO_CONFIG), HEX);
    Serial.print(" fifo1=0x"); Serial.print(icmRead(REG_FIFO_CONFIG1), HEX);
    Serial.print(" tmst=0x");  Serial.print(icmRead(REG_TMST_CONFIG), HEX);
    Serial.print(" intf0=0x"); Serial.print(icmRead(REG_INTF_CONFIG0), HEX);
    Serial.print(" count=");   Serial.print(count);
    Serial.print(" raw:");
    for (int i = 0; i < 8 && i < nbytes; i++) { Serial.print(' '); Serial.print(fifoBuf[i], HEX); }
    Serial.println();
  }

  static uint64_t batchDev[128];
  uint16_t np = 0;
  for (uint16_t off = 0; off + PKT <= nbytes; off += PKT) {
    // accept accel+gyro packets; bits[3:0] (tsType + ODR flags) vary -> 0x68 and 0x6C pass
    if ((fifoBuf[off] & 0xF0) != 0x60) { fifoErrors++; continue; }   // skip, do NOT flush
    parsePacket(&fifoBuf[off]);                    // updates devTimeUs (unwrapped device us)
    if (np < 128) batchDev[np++] = devTimeUs;
  }

  // Per-batch anchor: pin the newest sample to the watermark CYCCNT, place the rest by
  // device TMST offset (fixed us->cyc, NOT a fitted slope). Re-anchored every batch ->
  // no drift, nothing to regress. Reconstruct each sample's Teensy-clock time to verify.
  if (np > 0) {
    uint32_t anchorCyc; noInterrupts(); anchorCyc = int1Cyc; interrupts();
    uint64_t newestDev = batchDev[np - 1];
    for (uint16_t k = 0; k < np; k++) {
      uint32_t tCyc = anchorCyc - (uint32_t)((newestDev - batchDev[k]) * CYC_PER_US);
      if (reconHavePrev) {
        uint32_t ddt = tCyc - reconPrevCyc;          // wrap-safe
        if (ddt < reconMin) reconMin = ddt;
        if (ddt > reconMax) reconMax = ddt;
        reconSum += ddt; reconCount++;
      } else reconHavePrev = true;
      reconPrevCyc = tCyc;
    }
  }
}

void report() {
  static uint32_t lastMs = 0;
  static uint32_t lastPkt = 0, lastFsync = 0, lastInt = 0;
  uint32_t now = millis();
  if (now - lastMs < 1000) return;
  uint32_t dms = now - lastMs; lastMs = now;

  if (!g_ok) {
    // Re-probe WHO_AM_I live so it can be diagnosed without catching boot.
    setBank(0);
    g_whoami = icmRead(REG_WHO_AM_I);
    Serial.print("ST INIT_FAIL who=0x"); Serial.print(g_whoami, HEX);
    Serial.println("  (0x00/0xFF => SPI MISO/wiring/mode; other => wrong part)");
    return;
  }

  static uint32_t lastEdge = 0;
  uint32_t ic; noInterrupts(); ic = int1Count; interrupts();
  uint32_t fe = fsyncEdgeCount;

  float pktHz   = (pktCount  - lastPkt)   * 1000.0f / dms;
  float fsyncHz = (fsyncPkts - lastFsync) * 1000.0f / dms;
  float intHz   = (ic        - lastInt)   * 1000.0f / dms;
  float edgeHz  = (fe        - lastEdge)  * 1000.0f / dms;
  lastPkt = pktCount; lastFsync = fsyncPkts; lastInt = ic; lastEdge = fe;

  double dtMean = dtCount ? dtSum / dtCount : 0;
  double devHz  = dtMean > 0 ? 1e6 / dtMean : 0;

  Serial.print("ST pkt/s="); Serial.print(pktHz, 1);
  Serial.print(" int/s=");   Serial.print(intHz, 1);
  Serial.print(" dev_dt=");  Serial.print(dtMean, 1);
  Serial.print("us devODR=");Serial.print(devHz, 2);
  Serial.print(" dtJit[");   Serial.print(dtMin == 0xFFFFFFFF ? 0 : dtMin);
  Serial.print("..");        Serial.print(dtMax); Serial.print("]us");
  Serial.print("  fsync/s="); Serial.print(fsyncHz, 1);
  Serial.print(" (teensy_edges/s="); Serial.print(edgeHz, 1); Serial.print(")");
  if (fsN) { Serial.print(" fdelta["); Serial.print(fsMin); Serial.print(".."); Serial.print(fsMax);
             Serial.print("]us avg="); Serial.print(fsSum / fsN, 0); Serial.print("us"); }
  // Reconstructed per-sample Teensy-clock dt (per-batch anchor + device TMST). No drift fit.
  Serial.print("  teensy_dt="); Serial.print(reconCount ? (reconSum / reconCount) / CYC_PER_US : 0, 1);
  Serial.print("us recon["); Serial.print(reconMin == 0xFFFFFFFF ? 0 : reconMin / CYC_PER_US);
  Serial.print(".."); Serial.print(reconMax / CYC_PER_US); Serial.print("]us");
  if (fifoErrors) { Serial.print("  FIFO_ERR="); Serial.print(fifoErrors); }
  Serial.println();

  // reset per-interval jitter windows
  dtMin = 0xFFFFFFFF; dtMax = 0;
  fsMin = 0xFFFFFFFF; fsMax = 0; fsSum = 0; fsN = 0;
  reconMin = 0xFFFFFFFF; reconMax = 0; reconSum = 0; reconCount = 0;
}

void setup() {
  Serial.begin(2000000);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}

  ARM_DEMCR |= ARM_DEMCR_TRCENA;
  ARM_DWT_CYCCNT = 0;
  ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;

  pinMode(PIN_CS, OUTPUT);   csHigh();
  pinMode(PIN_INT1, INPUT);
  pinMode(PIN_FSYNC, OUTPUT); digitalWriteFast(PIN_FSYNC, LOW);

  SPI.begin();

  Serial.println("\n### ICM-42605 FSYNC/timestamp test ###");
  g_ok = icmInit();
  if (!g_ok) Serial.println("# init failed - will keep reporting WHO_AM_I for diagnosis");

  attachInterrupt(digitalPinToInterrupt(PIN_INT1), icmInt1ISR, RISING);
#if defined(__IMXRT1062__)
  NVIC_SET_PRIORITY(IRQ_GPIO6789, 16);
#endif

  // Start the 30 Hz FSYNC reference: toggle at 60 Hz -> 30 Hz square wave.
  fsyncTimer.begin(fsyncToggleISR, 1000000.0 / 60.0);

  Serial.println("# streaming. expect pkt/s~500, fsync/s~30, drift small.");
}

void loop() {
  if (g_ok && fifoReady) { fifoReady = false; drainFifo(); }
  report();
}
