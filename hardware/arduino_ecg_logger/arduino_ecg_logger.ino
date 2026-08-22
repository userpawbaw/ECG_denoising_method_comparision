/*
 * ECG acquisition logger  (docs/08_acquisition.md, STEP 28)
 *
 * 목적: 고정 샘플링 주파수로 ADC 를 읽어 "t_ms,adc_raw" 형식으로 시리얼에 내보낸다.
 *       PC 측 scripts/log_arduino.py 가 헤더 3줄을 붙여 CSV 로 저장한다.
 *
 * 배선 (AD8232 기준)
 *   AD8232 OUTPUT -> A0
 *   AD8232 LO+    -> D10      (lead-off detection)
 *   AD8232 LO-    -> D11
 *   3.3V / GND
 *
 * 중요
 *   - millis() 기반 고정 주기 샘플링. delay() 를 쓰면 주기가 흔들린다.
 *   - Serial 속도가 부족하면 샘플이 드랍된다. 500 Hz 에서는 115200 이상 필요.
 *     드랍이 나면 PC 측에서 t_ms 간격으로 검출된다 (arduino.py 가 경고를 띄운다).
 *   - lead-off 상태(LO+/LO- HIGH)에서는 -1 을 내보내 나중에 구간을 배제할 수 있게 한다.
 */

const int PIN_ECG   = A0;
const int PIN_LO_P  = 10;
const int PIN_LO_N  = 11;

const unsigned long FS_HZ     = 500;                 // 샘플링 주파수
const unsigned long PERIOD_US = 1000000UL / FS_HZ;   // 2000 us

unsigned long t_next_us = 0;
unsigned long t0_us     = 0;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_LO_P, INPUT);
  pinMode(PIN_LO_N, INPUT);
  // ADC 프리스케일러를 낮춰 변환 시간을 줄인다 (AVR 기준, 16MHz -> 약 16 us/변환)
#if defined(ADCSRA)
  ADCSRA = (ADCSRA & 0xF8) | 0x04;   // prescaler 16
#endif
  t0_us = micros();
  t_next_us = t0_us;
  Serial.println(F("# logger ready"));
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - t_next_us) < 0) return;
  t_next_us += PERIOD_US;

  int v;
  if (digitalRead(PIN_LO_P) == HIGH || digitalRead(PIN_LO_N) == HIGH) {
    v = -1;                              // lead-off: 전극이 떨어짐
  } else {
    v = analogRead(PIN_ECG);
  }
  unsigned long t_ms = (now - t0_us) / 1000UL;
  Serial.print(t_ms);
  Serial.print(',');
  Serial.println(v);
}
