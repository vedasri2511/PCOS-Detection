#define GSR 34  // Make sure this matches the GPIO pin you're using

void setup() {
  Serial.begin(115200);
  delay(1000);  // Give some time for serial to connect
}

void loop() {
  int avgValue = 0;

  for (int i = 0; i < 10; i++) {
    avgValue += analogRead(GSR);
    delay(10);
  }

  int sensorValue = avgValue / 10;
  float voltage = sensorValue * (3.3 / 4095.0);  // Convert ADC to voltage

  Serial.println(voltage);  // JUST print voltage

  delay(2000);
}
