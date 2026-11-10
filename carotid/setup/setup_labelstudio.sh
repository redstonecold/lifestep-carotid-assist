#!/bin/bash
# Label Studio 설치 및 실행 (Mac, conda 사용)
# 학습용 .venv(Python 3.14)와 분리된 conda 환경(Python 3.12)에 설치한다.
# 사용: bash setup/setup_labelstudio.sh
set -e
command -v conda >/dev/null || { echo "conda가 없습니다. Anaconda/Miniconda 설치 후 다시 실행하세요"; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | grep -q "^labelstudio " || conda create -n labelstudio python=3.12 -y
conda activate labelstudio
python -c "import label_studio" 2>/dev/null || pip install label-studio
echo "브라우저에서 http://localhost:8080 이 열립니다. 종료는 이 터미널에서 Ctrl+C"
label-studio start
