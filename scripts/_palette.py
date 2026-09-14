"""ui/palette.json 에서 생성됨 — 직접 고치지 말 것 (scripts/build_tokens.py)

`scripts/make_slides.py` 와 `scripts/build_metric_cards.py` 가 이것을 읽는다.
값을 고칠 곳은 `ui/palette.json` 이고, 고친 뒤 `python3 scripts/build_tokens.py`.
"""

# 방법 → 색. **색은 방법에 고정한다** (docs/33).
METHODS = {
    'M01': '#2a78d6',
    'M04': '#eb6834',
    'M08': '#1baf7a',
    'M05': '#e4a824',
    'M_FE': '#e880b4',
    'M02': '#7b53c1',
    'M06': '#184f95',
}

# 계열 → 색. 시연 화면처럼 방법이 많을 때 쓴다.
FAMILIES = {
    'input': '#b0453a',
    'classic': '#c07a1f',
    'deep': '#184f95',
    'oracle': '#2f7d4f',
    'frontend': '#c07a1f',
}

# 역할 색.
ROLES = {
    'clean': '#b8b6ae',
    'noisy': '#52514e',
    'ink': '#0b0b0b',
    'ink_2': '#52514e',
    'surface': '#fcfcfb',
    'bad': '#d03b3b',
}

# 개입의 종류 (구조 / 손실).
KINDS = {
    '구조': '#8a8a8a',
    '손실': '#184f95',
}

# 손실 램프 — ordinal 이다. 순서를 바꾸면 의미가 달라진다.
LOSS = {
    'L1': '#86b6ef',
    'L3': '#3987e5',
    'L6': '#184f95',
}

# ---- 기존 이름 (호출부를 덜 고치려고 그대로 둔다) ----
C = METHODS
KIND = KINDS
CLEAN = ROLES["clean"]
NOISY = ROLES["noisy"]
INK = ROLES["ink"]
INK2 = ROLES["ink_2"]
MUTE = ROLES["ink_2"]
SURFACE = ROLES["surface"]
BAD = ROLES["bad"]
