/*
 * raw_plot — A0 의 ADC 원값만 내보낸다. 필터 · 이득 환산 · 브리지 없음.
 *
 * 쓰는 법: 업로드 -> Arduino IDE 「도구 -> 시리얼 플로터」, 속도 115200.
 * 파형만 본다. 우리 프로그램(브리지 · 화면)과 완전히 무관하게 «입력이 들어오는가» 를 확인한다.
 *
 *   AFE OUTPUT -> A0  (기존 로거와 같은 핀)
 *
 * 플로터는 세로축을 자동으로 맞춘다. 신호가 10 칸뿐이면 그만큼 확대되어 보이므로
 * 실제 크기를 보려면 SHOW_RAILS 를 1 로 두어 0 · 1023 기준선을 함께 그린다.
 */

const int PIN_ECG = A0;
const unsigned long PERIOD_US = 4000;   // 250 Hz
#define SHOW_RAILS 0                     // 1: 0 과 1023 선을 같이 그려 실제 크기로 본다

unsigned long next_us;

void setup() {
  Serial.begin(115200);
  next_us = micros();
}

void loop() {
  if ((long)(micros() - next_us) < 0) return;
  next_us += PERIOD_US;
  int v = analogRead(PIN_ECG);
#if SHOW_RAILS
  Serial.print("lo:0,hi:1023,");
#endif
  Serial.print("adc:");
  Serial.println(v);
}
