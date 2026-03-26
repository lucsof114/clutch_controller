// ============================================
// Camera Trigger Controller - Serial Interface
// ============================================

const int OUTPUT_PIN = 9;  // PWM pin (OC1A)

bool syncRunning = false;
float currentFrequency = 0.0;

void setup() {
  Serial.begin(115200);
  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, LOW);
  
  // Initialize Timer1 but keep it stopped
  TCCR1A = 0;
  TCCR1B = 0;
  
  Serial.println("READY");
}

void startSync(float frequency) {
  if (frequency < 1.0 || frequency > 1000.0) {
    Serial.println("ERROR:FREQ_OUT_OF_RANGE");
    return;
  }
  
  // Stop timer during configuration
  TCCR1B = 0;
  TCCR1A = 0;
  TCNT1 = 0;
  
  // Calculate TOP value
  // f = 16MHz / (prescaler * (1 + TOP))
  uint16_t prescaler = 256;
  uint16_t topValue = (uint16_t)((16000000.0 / (prescaler * frequency)) - 1);
  
  // Check if we need a different prescaler for higher frequencies
  if (topValue < 100 && frequency > 100) {
    prescaler = 64;
    topValue = (uint16_t)((16000000.0 / (prescaler * frequency)) - 1);
    TCCR1B = (1 << WGM13) | (1 << WGM12) | (1 << CS11) | (1 << CS10);  // Prescaler 64
  } else {
    TCCR1B = (1 << WGM13) | (1 << WGM12) | (1 << CS12);  // Prescaler 256
  }
  
  ICR1 = topValue;
  OCR1A = topValue / 2;  // 50% duty cycle
  
  // Enable PWM output: Clear OC1A on compare match, set at BOTTOM
  TCCR1A = (1 << COM1A1) | (1 << WGM11);
  
  syncRunning = true;
  currentFrequency = frequency;
  
  Serial.print("OK:STARTED:");
  Serial.println(frequency);
}

void stopSync() {
  // Disable PWM output
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;
  
  // Ensure pin is low
  digitalWrite(OUTPUT_PIN, LOW);
  
  syncRunning = false;
  currentFrequency = 0.0;
  
  Serial.println("OK:STOPPED");
}

void sendStatus() {
  Serial.print("STATUS:");
  Serial.print(syncRunning ? "RUNNING" : "STOPPED");
  Serial.print(":");
  Serial.println(currentFrequency);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("START:")) {
      float freq = command.substring(6).toFloat();
      startSync(freq);
    }
    else if (command == "STOP") {
      stopSync();
    }
    else if (command == "STATUS") {
      sendStatus();
    }
    else if (command == "PING") {
      Serial.println("PONG");
    }
    else {
      Serial.println("ERROR:UNKNOWN_COMMAND");
    }
  }
}