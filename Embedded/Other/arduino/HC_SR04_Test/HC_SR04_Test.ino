// HC-SR04 Diagnostic & Auto-Detect Sketch
// Checks pin states and tests both pin assignments

void setup() {
  Serial.begin(9600);
  delay(500);
  Serial.println("=== HC-SR04 Hardware Diagnostic ===");
}

long measure(int trigPin, int echoPin) {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  digitalWrite(trigPin, LOW);
  delayMicroseconds(4);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  return pulseIn(echoPin, HIGH, 35000); // 35ms timeout
}

void loop() {
  // Test Config A: Pin 2 = TRIG, Pin 3 = ECHO
  long durA = measure(2, 3);
  
  delay(50);
  
  // Test Config B: Pin 3 = TRIG, Pin 2 = ECHO (swapped)
  long durB = measure(3, 2);

  // Check static pin levels
  pinMode(2, INPUT);
  pinMode(3, INPUT);
  int statePin2 = digitalRead(2);
  int statePin3 = digitalRead(3);

  if (durA > 0) {
    float cm = (durA * 0.0343) / 2.0;
    Serial.print("[WORKING! Config A: TRIG=Pin2, ECHO=Pin3] Distance: ");
    Serial.print(cm, 1);
    Serial.println(" cm");
  } else if (durB > 0) {
    float cm = (durB * 0.0343) / 2.0;
    Serial.print("[WORKING! Config B (Swapped): TRIG=Pin3, ECHO=Pin2] Distance: ");
    Serial.print(cm, 1);
    Serial.println(" cm");
  } else {
    Serial.print("[NO ECHO] Pin 2 level: ");
    Serial.print(statePin2 == HIGH ? "HIGH" : "LOW");
    Serial.print(" | Pin 3 level: ");
    Serial.print(statePin3 == HIGH ? "HIGH" : "LOW");
    Serial.println(" -> Check 5V power, GND, or loose jumper wires");
  }

  delay(400);
}
