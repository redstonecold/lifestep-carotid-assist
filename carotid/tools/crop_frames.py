"""전체 프레임 비식별 크롭 + 흑백 변환 (계획서 11절 전처리 1·2·4단계).

- WMV(P001·P002): (73,64,795,569) → 722×505. 화면 상단 기관명·이니셜·일시 제거
- AVI(P003):      (96,36,548,384) → 452×348. 하단 ECG 띠 제외
- 이미 크롭된 파일(크기가 결과 크기와 같음)은 건너뜀 → 재실행 안전
사용: python tools/crop_frames.py [P001 P002 P003]
"""
import os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BOX = {"P001": (73, 64, 795, 569), "P002": (73, 64, 795, 569), "P003": (96, 36, 548, 384)}

def work(args):
    f, box = args
    im = Image.open(f)
    size = (box[2] - box[0], box[3] - box[1])
    if im.size == size and im.mode == "L":
        return 0
    im.convert("L").crop(box).save(f)
    return 1

T0 = time.time()
BUDGET = float(os.environ.get("BUDGET", "1e9"))

if __name__ == "__main__":
    persons = sys.argv[1:] or list(BOX)
    for p in persons:
        files = sorted((ROOT / "data/frames" / p).rglob("*.png"))
        n = 0
        with ProcessPoolExecutor() as ex:
            for k in range(0, len(files), 500):  # 500장씩 처리, 시간 예산 초과 시 중단(재실행하면 이어서)
                n += sum(ex.map(work, [(f, BOX[p]) for f in files[k:k+500]], chunksize=32))
                if time.time() - T0 > BUDGET:
                    print(p, "시간 예산 도달, 재실행하면 이어서 진행", flush=True); break
        print(p, "files", len(files), "cropped", n, flush=True)
        if time.time() - T0 > BUDGET: break
