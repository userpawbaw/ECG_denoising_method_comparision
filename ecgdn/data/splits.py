"""record 단위 split — **단일 진실 원천(SSOT)** (docs/01_design.md 3.2).

window 단위로 나누면 같은 환자의 같은 morphology 가 train/test 양쪽에 들어가
성능이 과대평가된다 (docs/00_review.md 의 leakage 항목).

DS1/DS2 는 de Chazal 등이 제안해 널리 인용되는 **inter-patient** 분할이다.
paced beat 기록(102,104,107,217)은 morphology 가 근본적으로 다르므로 본 실험에서
분리해 보조 분석에만 쓴다.
"""
from __future__ import annotations

__all__ = ["MITDB_ALL", "MITDB_SPLIT", "NOISE_SPLIT_FRAC", "split_of", "check_split",
           "D2_ADAPT_SPLITS", "assert_adaptation_records"]

MITDB_ALL: tuple[str, ...] = tuple(str(r) for r in (
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124,
    200, 201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
    228, 230, 231, 232, 233, 234,
))

# DS1 (22) 에서 4개를 validation 으로 뗀다.
_DS1_TRAIN = (101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
              122, 124, 201, 203, 205, 207, 209, 215)
_DS1_VAL = (208, 220, 223, 230)
_DS2_TEST = (100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
             212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234)
_PACED = (102, 104, 107, 217)

MITDB_SPLIT: dict[str, tuple[str, ...]] = {
    "train": tuple(str(r) for r in _DS1_TRAIN),
    "val": tuple(str(r) for r in _DS1_VAL),
    "test": tuple(str(r) for r in _DS2_TEST),
    "paced": tuple(str(r) for r in _PACED),      # 주 실험에서 제외, 보조 분석용
}

# 잡음 기록(bw/ma/em)도 시간축으로 disjoint 분할 (docs/00_review.md A-7).
# 같은 잡음 파형 구간이 train 과 test 에 함께 들어가면 noise leakage 다.
NOISE_SPLIT_FRAC: dict[str, tuple[float, float]] = {
    "train": (0.00, 0.60),
    "val": (0.60, 0.75),
    "test": (0.75, 1.00),
}


# device adaptation(D2) 규약: fine-tuning 에 쓸 수 있는 clean ECG 는 TRAIN 뿐이다.
# TEST record 의 morphology 를 adaptation 에 쓰면, F-8 과 같은 종류의 leakage 다
# (split 축과 변이 축이 어긋나면 record split 이 leakage 를 막지 못한다).
D2_ADAPT_SPLITS: tuple[str, ...] = ("train",)


def assert_adaptation_records(records) -> None:
    """D2 fine-tuning 에 쓰는 record 가 TRAIN 안에 있는지 강제한다."""
    allowed = {r for k in D2_ADAPT_SPLITS for r in MITDB_SPLIT[k]}
    bad = sorted(set(map(str, records)) - allowed)
    if bad:
        raise AssertionError(
            f"device adaptation 에 TRAIN 이 아닌 record 가 들어왔다: {bad}. "
            f"허용 split = {D2_ADAPT_SPLITS}")


def split_of(record: str) -> str:
    for k, v in MITDB_SPLIT.items():
        if str(record) in v:
            return k
    raise KeyError(f"record {record!r} not in any split")


def check_split() -> None:
    """무결성 검사. import 시 자동 실행된다."""
    seen: set[str] = set()
    for k, v in MITDB_SPLIT.items():
        dup = seen & set(v)
        if dup:
            raise AssertionError(f"record in multiple splits: {sorted(dup)}")
        seen |= set(v)
    missing = set(MITDB_ALL) - seen
    extra = seen - set(MITDB_ALL)
    if missing or extra:
        raise AssertionError(f"split mismatch: missing={sorted(missing)} extra={sorted(extra)}")


check_split()
