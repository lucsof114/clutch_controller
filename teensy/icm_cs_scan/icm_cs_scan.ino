// icm_cs_scan — probe WHO_AM_I on every candidate CS pin to locate ICM-42605 chips.
// Shared SPI bus: SCK13, MOSI11, MISO12 (skipped as CS). Prints any pin where
// WHO_AM_I reads 0x42. Useful when IMUs don't respond on their configured CS pins.
#include <SPI.h>

static const uint8_t WHO = 0x75;
static const uint8_t WHOAMI = 0x42;
static SPISettings CFG(1000000, MSBFIRST, SPI_MODE3);
// every digital pin except the SPI bus pins (11/12/13)
static const uint8_t PINS[] = {0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,22,23};

uint8_t readWho(uint8_t cs) {
  SPI.beginTransaction(CFG);
  digitalWriteFast(cs, LOW);
  SPI.transfer(WHO | 0x80);
  uint8_t v = SPI.transfer(0x00);
  digitalWriteFast(cs, HIGH);
  SPI.endTransaction();
  return v;
}

void setup() {
  Serial.begin(2000000);
  while (!Serial && millis() < 3000) {}
  for (uint8_t i = 0; i < sizeof(PINS); i++) { pinMode(PINS[i], OUTPUT); digitalWriteFast(PINS[i], HIGH); }
  SPI.begin();
  delay(50);
}

void loop() {
  Serial.println("# CS scan for ICM-42605 (WHO_AM_I should read 0x42):");
  uint8_t found = 0;
  for (uint8_t i = 0; i < sizeof(PINS); i++) {
    uint8_t cs = PINS[i];
    uint8_t v = readWho(cs);
    if (v == WHOAMI) { Serial.print("  CS "); Serial.print(cs); Serial.print(" -> 0x42  *** ICM FOUND ***"); Serial.println(); found++; }
    else if (v != 0x00 && v != 0xFF) { Serial.print("  CS "); Serial.print(cs); Serial.print(" -> 0x"); Serial.println(v, HEX); }
  }
  Serial.print("# total ICM found: "); Serial.println(found);
  Serial.println("# (pins reading 0x00=MISO low/no chip, 0xFF=bus idle/no MISO)");
  delay(2000);
}
