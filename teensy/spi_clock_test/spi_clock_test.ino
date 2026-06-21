// spi_clock_test — drive square waves on EVERY pin the IMU system uses, for
// PicoScope signal-integrity comparison across wire lengths / stub counts.
//
// PWM 1 MHz, 50% on the PWM-capable pins; pins 16 & 32 aren't PWM-capable so
// they're toggled by IntervalTimer (~500 kHz). Scope each vs GND and compare
// edge shape / overshoot / ringing.
//
// NOTE: INT pins (8/9/10/14/15/16) are normally inputs and CS (2-7) selects
// chips, so with IMUs connected expect contention on those — the SPI bus
// (11/12/13) and lightly-loaded pins are the clean comparison.
//
// Serial: send a number (kHz) to change the PWM frequency live (default 1000).

#include <IntervalTimer.h>

// new layout — PWM-capable pins: SCK13, MOSI11, CS 2-7, MISO 14/15/18/19, INT 8/9/10/22
const uint8_t PWM_PINS[] = {2,3,4,5,6,7,8,9,10,11,13,14,15,18,19,22};
// not PWM-capable -> timer toggled: MISO 16/17, INT 20/21, sync 32
const uint8_t TOG_PINS[] = {16, 17, 20, 21, 32};

uint32_t freqHz = 1000000;
IntervalTimer togTimer;
volatile bool togLevel = false;

void toggleAll() {
  togLevel = !togLevel;
  for (uint8_t i = 0; i < sizeof(TOG_PINS); i++) digitalWriteFast(TOG_PINS[i], togLevel);
}

void applyFreq(uint32_t hz) {
  analogWriteResolution(8);
  for (uint8_t i = 0; i < sizeof(PWM_PINS); i++) {
    analogWriteFrequency(PWM_PINS[i], hz);
    analogWrite(PWM_PINS[i], 128);
  }
}

void setup() {
  Serial.begin(2000000);
  for (uint8_t i = 0; i < sizeof(PWM_PINS); i++) pinMode(PWM_PINS[i], OUTPUT);
  for (uint8_t i = 0; i < sizeof(TOG_PINS); i++) pinMode(TOG_PINS[i], OUTPUT);
  applyFreq(freqHz);
  togTimer.begin(toggleAll, 1.0);   // ~500 kHz on pins 16 & 32
}

void loop() {
  if (Serial.available()) {
    long khz = Serial.parseInt();
    if (khz > 0) { freqHz = (uint32_t)khz * 1000UL; applyFreq(freqHz);
      Serial.print("# PWM @ "); Serial.print(freqHz/1000.0,1); Serial.println(" kHz"); }
  }
  static uint32_t last = 0;
  if (millis() - last > 1000) { last = millis();
    Serial.print("# all pins driven: PWM 2-15 @ "); Serial.print(freqHz/1000.0,1);
    Serial.println(" kHz; pins 16/32 ~500 kHz"); }
}
