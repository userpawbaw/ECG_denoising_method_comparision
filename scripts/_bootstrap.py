"""스크립트 공통 부트스트랩: 저장소 루트를 import path 에 넣고 cwd 를 루트로 고정한다.

cwd 를 고정하지 않으면 results/ 경로가 실행 위치에 따라 달라진다.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
