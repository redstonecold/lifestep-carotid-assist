"""학습한 모델의 검증 예측을 정답과 겹쳐 그린 이미지 (탐색 결과 눈으로 확인용).

- 초록: 정답(사람 라벨), 빨강: 모델 예측. 칸 위에 세트·Dice·면적(mm²)
- 세트 순서(A·B·C·F)로, 같은 세트 안에서는 Dice가 낮은(틀린) 장부터
- 대상자 영상이 담기므로 reports/ 아래 PNG(커밋 제외)로만 저장
사용: python tools/show_predictions.py runs/s1_lr3e-4_dice_focal [--split val] [--cols 4]
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import CarotidDataset
from src.predict import predict_files

TILE_W = 420


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--split", default="val"); ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    run = ROOT / a.run
    ds = CarotidDataset(ROOT, a.split, augment=None)
    s = ds.samples
    masks = predict_files([x["img_path"] for x in s], run / "best.pt", mm_per_px=[x["mm_per_px"] for x in s])
    items = []
    for x, pred in zip(s, masks):
        gt = np.array(Image.open(x["mask_path"])) > 0
        p = pred > 0
        dice = 2 * (p & gt).sum() / (p.sum() + gt.sum()) if gt.any() else float("nan")
        items.append((x, gt, p, dice))
    items.sort(key=lambda t: ("ABCF".index(t[0]["grade"]), t[3] if t[3] == t[3] else 0))

    tiles = []
    for x, gt, p, dice in items:
        img = cv2.cvtColor(np.array(Image.open(x["img_path"]).convert("L")), cv2.COLOR_GRAY2BGR)
        for m, col in ((gt, (0, 220, 0)), (p, (0, 0, 255))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, col, 2)
        img = cv2.resize(img, (TILE_W, round(img.shape[0] * TILE_W / img.shape[1])))
        mm2 = x["mm_per_px"] ** 2
        d = f"Dice {dice:.3f}" if dice == dice else ("OK (empty)" if p.sum() * mm2 < 10 else "FALSE POSITIVE")
        bar = np.zeros((40, TILE_W, 3), np.uint8)
        cv2.putText(bar, f"{x['person_id']} {x['video_id']} #{x['frame_idx']}  [{x['grade']}]  {d}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(bar, f"area  GT {gt.sum() * mm2:.0f} mm2   pred {p.sum() * mm2:.0f} mm2",
                    (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        tiles.append(np.vstack([bar, img]))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], TILE_W, 3), np.uint8)]) for t in tiles]
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
    head = np.zeros((34, grid.shape[1], 3), np.uint8)
    cv2.putText(head, f"{run.name}  ({a.split})   green = label   red = model", (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    out = ROOT / "reports" / f"pred_{run.name}_{a.split}.png"
    cv2.imwrite(str(out), np.vstack([head, grid]))
    print(out)


if __name__ == "__main__":
    main()
