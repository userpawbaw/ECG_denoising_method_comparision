/*
 * ECG acquisition firmware — 수집(CSV)과 실시간 시연(브리지)을 **한 스케치**로 겸한다.
 *
 *   docs/08_acquisition.md  (STEP 28, 수집)
 *   docs/30_realtime_demo.md 6.2  (R-5, 실시간)
 *
 * 배선 (AD8232 기준)
 *   AD8232 OUTPUT -> A0
 *   AD8232 LO+    -> D10      (lead-off detection)
 *   AD8232 LO-    -> D11
 *   3.3V / GND
 *
 * ---------------------------------------------------------------------------
 * 왜 스케치가 하나인가
 * ---------------------------------------------------------------------------
 * 수집용과 시연용을 따로 두면 **시연 당일에 어느 것이 올라가 있는지 모르게
 * 된다.** 그래서 하나로 두고 PC 가 명령 1 바이트로 모드를 고른다.
 * 아무 명령도 안 보내면 **켜자마자 ASCII 로 흘린다** — 아두이노 IDE 의 시리얼
 * 모니터/플로터가 그대로 동작하고, scripts/log_arduino.py 도 그대로 쓴다.
 *
 * ---------------------------------------------------------------------------
 * 명령 (1 바이트, 아무 때나)
 * ---------------------------------------------------------------------------
 *   'a'  ASCII 모드      "t_ms,adc\n"      (기본. IDE 플로터 호환)
 *   'b'  BINARY 모드     5 바이트 프레임    (아래)
 *   '2'  fs = 250 Hz     <- 실시간 시연용. 모델 학습 fs 와 같아 리샘플이 없다
 *   '5'  fs = 500 Hz     <- 기본. 수집용
 *   '1'  fs = 1000 Hz    <- EXP-F 의 fs 질문용 (BINARY + 250000 baud 필요)
 *   '?'  헤더 한 줄
 *   'r'  카운터 리셋 (시퀀스·드롭·시각)
 *
 * ---------------------------------------------------------------------------
 * BINARY 프레임 — 5 바이트
 * ---------------------------------------------------------------------------
 *   [0xA5][seq u8][val_lo][val_hi][xor]
 *     seq : 0..255 순환. **PC 가 손실 개수를 정확히 셀 수 있게 하는 유일한 수단**이다.
 *     val : 0..1023 ADC, 0xFFFF = lead-off
 *     xor : 앞 4 바이트의 XOR. 0xA5 는 payload 에도 나올 수 있으므로
 *           동기 바이트만으로는 부족하고 검사합이 있어야 재동기가 확실해진다.
 *
 * ASCII 는 1 kHz 를 못 버틴다. 한 줄이 약 12 바이트라 115200 baud(11.5 kB/s)
 * 에서 1 kHz 는 12 kB/s 로 **용량을 넘는다.** BINARY 는 5 kB/s 다.
 *
 * ---------------------------------------------------------------------------
 * 드롭을 숨기지 않는다
 * ---------------------------------------------------------------------------
 * AVR 의 Serial 송신 버퍼는 64 바이트고, 가득 차면 Serial.write 가 **블록한다.**
 * 그대로 두면 샘플 주기가 밀려 fs 가 조용히 흔들린다 — 그러면 R-peak 간격이
 * 틀리고 그것이 잡음처럼 보인다. 그래서 availableForWrite() 로 자리가 없으면
 * **그 샘플을 버리고 센다.** 버린 것은 seq 가 건너뛰므로 PC 가 정확히 안다.
 * 조용히 느려지는 것보다 **큰 소리로 빠뜨리는 편이 낫다.**
 */

const int PIN_ECG   = A0;
const int PIN_LO_P  = 10;
const int PIN_LO_N  = 11;

// 250000 은 16 MHz AVR 에서 오차 0 % 로 떨어진다(UBRR=3, U2X=1). 1 kHz 를 쓸
// 때만 올리면 된다 — PC 쪽 --baud 도 같이 바꿀 것.
const unsigned long SERIAL_BAUD = 115200;

const uint8_t  MODE_ASCII  = 0;
const uint8_t  MODE_BINARY = 1;

uint8_t       g_mode      = MODE_ASCII;
unsigned long g_fs_hz     = 500;
unsigned long g_period_us = 2000;

unsigned long t_next_us = 0;
unsigned long t0_us     = 0;
uint8_t       g_seq     = 0;
unsigned long g_dropped = 0;      // 송신 버퍼가 없어 버린 샘플

void print_header() {
  Serial.print(F("# ecgstream v1 fs="));
  Serial.print(g_fs_hz);
  Serial.print(F(" mode="));
  Serial.print(g_mode == MODE_BINARY ? F("bin") : F("ascii"));
  Serial.print(F(" bits=10 vref=5.00 dropped="));
  Serial.println(g_dropped);
}

void set_fs(unsigned long fs) {
  g_fs_hz     = fs;
  g_period_us = 1000000UL / fs;
  t0_us       = micros();
  t_next_us   = t0_us;
  g_seq       = 0;
  g_dropped   = 0;
}

void handle_command() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    switch (c) {
      case 'a': g_mode = MODE_ASCII;  print_header(); break;
      case 'b': g_mode = MODE_BINARY; break;   // 헤더를 안 찍는다 — 프레임에 섞인다
      case '2': set_fs(250);  break;
      case '5': set_fs(500);  break;
      case '1': set_fs(1000); break;
      case 'r': set_fs(g_fs_hz); break;
      case '?': print_header(); break;
      default:  break;                          // 개행 등은 무시
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(PIN_LO_P, INPUT);
  pinMode(PIN_LO_N, INPUT);
  // ADC 프리스케일러를 낮춰 변환 시간을 줄인다 (AVR 기준, 16MHz -> 약 16 us/변환).
  // 1 kHz 에서 기본 프리스케일러(128, 약 112 us)면 변환이 주기를 먹는다.
#if defined(ADCSRA)
  ADCSRA = (ADCSRA & 0xF8) | 0x04;   // prescaler 16
#endif
  set_fs(500);
  Serial.println(F("# logger ready"));
  print_header();
}

void loop() {
  handle_command();

  unsigned long now = micros();
  if ((long)(now - t_next_us) < 0) return;
  t_next_us += g_period_us;

  uint16_t val;
  if (digitalRead(PIN_LO_P) == HIGH || digitalRead(PIN_LO_N) == HIGH) {
    val = 0xFFFF;                        // lead-off: 전극이 떨어짐
  } else {
    val = (uint16_t)analogRead(PIN_ECG);
  }

  if (g_mode == MODE_BINARY) {
    // 자리가 없으면 **버리고 센다.** 블록하면 fs 가 조용히 흔들린다.
    if (Serial.availableForWrite() < 5) { g_seq++; g_dropped++; return; }
    uint8_t lo = (uint8_t)(val & 0xFF), hi = (uint8_t)(val >> 8);
    uint8_t x  = 0xA5 ^ g_seq ^ lo ^ hi;
    Serial.write((uint8_t)0xA5);
    Serial.write(g_seq);
    Serial.write(lo);
    Serial.write(hi);
    Serial.write(x);
    g_seq++;
  } else {
    // ASCII 는 기존 규격 그대로다 — lead-off 는 -1, 시각은 ms.
    if (Serial.availableForWrite() < 16) { g_seq++; g_dropped++; return; }
    unsigned long t_ms = (now - t0_us) / 1000UL;
    Serial.print(t_ms);
    Serial.print(',');
    Serial.println(val == 0xFFFF ? -1 : (int)val);
    g_seq++;
  }
}
