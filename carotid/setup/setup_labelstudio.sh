#!/bin/bash
# Label Studio 설치 및 실행 (Mac, conda 사용)
# 학습용 .venv(Python 3.14)와 분리된 conda 환경(Python 3.12)에 설치한다.
# 사용: bash setup/setup_labelstudio.sh
set -e
command -v conda >/dev/null || { echo "conda가 없습니다. Anaconda/Miniconda 설치 후 다시 실행하세요"; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | grep -q "^labelstudio " || conda create -n labelstudio python=3.12 -y
# 학습용 .venv가 활성화돼 있어도 그쪽 label-studio(Python 3.14, 실행 불가)를 쓰지 않도록 전체 경로로 실행한다
LS_ENV="$(conda info --base)/envs/labelstudio"
"$LS_ENV/bin/python" -c "import label_studio" 2>/dev/null || "$LS_ENV/bin/python" -m pip install label-studio
# 기본값은 한 번에 100장까지만 업로드된다. batch03(267장)을 한 번에 넣을 수 있게 늘린다
export DATA_UPLOAD_MAX_NUMBER_FILES=1000
echo "브라우저에서 http://localhost:8080 이 열립니다. 종료는 이 터미널에서 Ctrl+C"
"$LS_ENV/bin/label-studio" start
