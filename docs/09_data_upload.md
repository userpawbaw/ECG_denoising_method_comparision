# 09. PhysioNet 데이터 공유 절차 (STEP 14 보조)

## 왜 이 문서가 필요한가

원격 개발 세션에서는 `physionet.org` 가 조직 egress 정책으로 **차단(403)** 된다.
따라서 다운로드는 로컬에서 하고, 그 결과물을 저장소로 공유해야 실데이터 실험을 이어갈 수 있다.

`github.com` 과 `raw.githubusercontent.com` 은 원격 세션에서 접근 가능하므로,
**저장소에 커밋해서 push 하는 것이 가장 단순한 경로**다.

## 자주 걸리는 함정

`.gitignore` 에 `data/` 규칙이 있어서 `git add data/raw` 를 하면 **git 이 조용히 무시**한다.
파일이 하나도 스테이징되지 않았는데 에러도 안 난다.

→ 지금은 `data/raw/mitdb/`, `data/raw/nstdb/` 에 대해 **예외를 열어 두었으므로**
`git add data/raw` 가 정상 동작한다. (실측 Arduino 데이터는 계속 제외된다 — 개인정보)

## 절차 (로컬에서)

```bash
git pull                                  # .gitignore 예외를 먼저 받아온다

pip install -r requirements.txt
python scripts/download_data.py --db mitdb --db nstdb
#   -> data/raw/mitdb/  (48 records, 약 100 MB)
#   -> data/raw/nstdb/  (bw/ma/em 포함, 약 30 MB)

python scripts/download_data.py --verify-only    # 개수/샘플수 검증 + manifest 기록

git add data/raw
git status --short | head              # 파일이 실제로 스테이징됐는지 반드시 확인
git commit -m "data: MIT-BIH + NSTDB 원본 추가"
git push
```

`git status` 에서 아무것도 안 잡히면 `.gitignore` 예외가 아직 반영되지 않은 것이다.
`git pull` 을 먼저 했는지 확인할 것.

## 확인 사항

| 항목 | 기대값 |
|---|---|
| `data/raw/mitdb/*.hea` | 48 개 |
| `data/raw/nstdb/{bw,ma,em}.hea` | 3 개 존재 |
| 총 용량 | 약 130 MB |

`data/raw/nothing` 같은 자리표시자 파일이 남아 있으면 지워도 된다.

## 데이터가 들어오면 자동으로 달라지는 것

`ecgdn/data/sources.get_source("auto")` 가 `data/raw/mitdb` 를 발견하면
**합성 소스 대신 MIT-BIH 를 쓴다.** 즉 config 를 고칠 필요가 없다.

다만 아래 세 가지는 실데이터 기준으로 **다시 측정해야 한다** (합성 기준 값이 낙관적이다).

```bash
python scripts/measure_metric_floor.py     # 지표 분해능 (docs/03)
python scripts/tune_swt.py                 # SWT 파라미터, TRAIN split 기준 (docs/05)
python scripts/diagnose_sameni.py          # Sameni 자가진단 (docs/06)
```
