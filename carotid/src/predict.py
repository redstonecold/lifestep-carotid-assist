"""
학습된 모델로 영상 전체 프레임을 추론한다.

사용:
    python -m src.predict --video data/frames/V01 --checkpoint runs/default/best.pt

출력:
    runs/predict/<video_id>/  아래에 각 프레임의 마스크 PNG 저장

실행 위치: carotid/ 폴더
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import _load_gray, _rescale, _normalize, pad_for_inference, TARGET_MM_PER_PX
from src.model import build_model, get_device, load_checkpoint


def predict_folder(
    frame_dir: Path,
    checkpoint: Path,
    mm_per_px: float = TARGET_MM_PER_PX,
    threshold: float = 0.5,
    out_dir: Path | None = None,
) -> list[np.ndarray]:
    """
    frame_dir 안의 PNG 파일 전체에 대해 마스크를 예측한다.

    Returns:
        masks  프레임 순서와 같은 이진 마스크(0/1) 리스트 (원본 해상도)
    """
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = get_device()
    model  = build_model(pretrained=False).to(device)
    meta   = load_checkpoint(model, checkpoint, device=device)
    print(f"[predict] 체크포인트: {checkpoint.name}"
          f"  (epoch={meta.get('epoch','?')}, val_dice={meta.get('val_dice','?')})")
    model.eval()

    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        print(f"[predict] PNG 없음: {frame_dir}")
        return []

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    for i, fpath in enumerate(frames):
        img = _load_gray(fpath)
        img = _rescale(img, mm_per_px, Image.BILINEAR)
        img = _normalize(img)

        padded, orig_h, orig_w = pad_for_inference(img)
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logit = model(tensor)
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
        mask = (prob[:orig_h, :orig_w] > threshold).astype(np.uint8) * 255

        masks.append(mask)

        if out_dir:
            Image.fromarray(mask).save(out_dir / (fpath.stem + "_mask.png"))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames)} 프레임 처리 중...")

    print(f"[predict] 완료: {len(masks)} 프레임")
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True,  help="프레임 폴더 경로 (예: data/frames/V01)")
    ap.add_argument("--checkpoint", required=True,  help="체크포인트 경로 (예: runs/default/best.pt)")
    ap.add_argument("--mm_per_px",  type=float, default=TARGET_MM_PER_PX,
                    help=f"픽셀당 mm (기본: {TARGET_MM_PER_PX})")
    ap.add_argument("--threshold",  type=float, default=0.5,  help="이진화 임계값 (기본: 0.5)")
    ap.add_argument("--out",        default=None, help="마스크 저장 폴더 (없으면 저장 안 함)")
    args = ap.parse_args()

    video_dir  = ROOT / args.video
    ckpt_path  = ROOT / args.checkpoint
    out_dir    = ROOT / args.out if args.out else None

    if not video_dir.exists():
        print(f"폴더 없음: {video_dir}")
        return
    if not ckpt_path.exists():
        print(f"체크포인트 없음: {ckpt_path}")
        return

    predict_folder(video_dir, ckpt_path,
                   mm_per_px=args.mm_per_px,
                   threshold=args.threshold,
                   out_dir=out_dir)


if __name__ == "__main__":
    main()
