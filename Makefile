.PHONY: install test lint check-nodata

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -q

# 외부 데이터 없이 실행 가능한 검증 전체
check-nodata:
	python -m pytest tests/ -q
	python scripts/check_synthetic.py
	python scripts/check_noise.py
	python scripts/check_snr_estimator.py
	python scripts/diagnose_sameni.py --data synthetic
