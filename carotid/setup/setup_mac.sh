#!/bin/bash
# LifeStep 경동맥 프로젝트 - Mac 학습 환경 설치
# 사용: 터미널에서  bash setup/setup_mac.sh   (carotid 폴더 안에서 실행)
set -e
cd "$(dirname "$0")/.."
if [ "$(uname)" != "Darwin" ]; then echo "macOS에서 실행하세요"; exit 1; fi
if [ "$(uname -m)" != "arm64" ]; then echo "경고: Apple Silicon(arm64) Python이 아니면 GPU(MPS)를 쓸 수 없습니다"; fi
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10 이상 필요"' || { echo "Python 3.10 이상을 설치하세요 (예: brew install python@3.12)"; exit 1; }
if [ ! -d .venv ]; then $PY -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install segmentation-models-pytorch timm opencv-python pillow numpy pydicom
echo
echo "설치 완료. 다음 명령으로 모델 다운로드와 성능 점검을 하세요:"
echo "  source .venv/bin/activate && python tools/check_env.py"
