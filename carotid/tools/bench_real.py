"""실제 프레임으로 Mac 학습 속도 측정 (라벨 전이라 마스크만 임시 띠 모양).

- 이미지 읽기·재표본화·증강·크롭·정규화는 src/dataset.py 함수를 그대로 사용
- 모델·손실·옵티마이저는 src/train.py 와 동일 (U-Net ResNet34, Dice+BCE, AdamW, 배치 8)
- PYTORCH_ENABLE_MPS_FALLBACK=0 : GPU 미지원 연산이 있으면 CPU로 넘기지 않고 오류로 멈춤
- epoch = 학습 7명 × 50장 = 350장 (BalancedPersonSampler 규칙)
- 데이터 준비(CPU)와 GPU 연산 시간을 나눠 기록, 연속 실행으로 발열 저하 확인
결과: reports/bench_real.txt

사용: python tools/bench_real.py [--epochs 40]
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import argparse, csv, random, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import _load_gray, _rescale, _augment, _random_crop, _normalize, CROP_SIZE
from src.model import build_model
from src.train import combined_loss

N_TRAIN_PERSONS, PER_PERSON, BATCH = 7, 50, 8
lines = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

def band_mask(shape):
    h, w = shape; m = np.zeros(shape, np.float32)
    c = h // 2; m[c - 35:c + 35, :] = 1.0   # 혈관 지름 약 70px 크기의 임시 마스크
    return m

def make_batch(paths, mm):
    xs, ys = [], []
    for p in paths:
        img = _load_gray(p); mask = band_mask(img.shape)
        img = _rescale(img, mm[p], Image.BILINEAR); mask = _rescale(mask, mm[p], Image.NEAREST)
        img, mask = _augment(img, mask)
        img, mask = _random_crop(img, mask, CROP_SIZE)
        xs.append(_normalize(img)); ys.append(mask)
    return (torch.from_numpy(np.stack(xs))[:, None].float(),
            torch.from_numpy(np.stack(ys))[:, None].float())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    assert torch.backends.mps.is_available(), "MPS 사용 불가"
    dev = torch.device("mps")

    vmm = {r["video_id"]: float(r["mm_per_px"]) for r in csv.DictReader(open(ROOT/"manifest/videos.csv", encoding="utf-8"))}
    pool, mm = [], {}
    for vdir in sorted((ROOT/"data/frames").glob("P*/V*")):
        fs = sorted(vdir.glob("frame_*.png"))
        for f in random.Random(0).sample(fs, min(60, len(fs))):
            pool.append(f); mm[f] = vmm[vdir.name]
    log(f"실제 프레임 {len(pool)}장 사용 (AVI·WMV 혼합, 원본 크기 그대로)")

    model = build_model(pretrained=True).to(dev)
    assert next(model.parameters()).device.type == "mps"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = -(-N_TRAIN_PERSONS * PER_PERSON // BATCH)
    log(f"장치: {dev} | CPU 대체 실행: 꺼짐 | epoch당 {N_TRAIN_PERSONS}명×{PER_PERSON}장 = {steps}스텝, 배치 {BATCH}")

    rng = random.Random(1); ep_times = []
    for ep in range(1, a.epochs + 1):
        t_data = t_gpu = 0.0; t_ep = time.time()
        model.train()
        for _ in range(steps):
            t = time.time(); x, y = make_batch(rng.sample(pool, BATCH), mm); t_data += time.time() - t
            t = time.time()
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = combined_loss(model(x), y, "dice_bce"); loss.backward(); opt.step()
            torch.mps.synchronize(); t_gpu += time.time() - t
        ep_s = time.time() - t_ep; ep_times.append(ep_s)
        log(f"epoch {ep:2d}: {ep_s:5.1f}초 (데이터 준비 CPU {t_data:4.1f}초 / GPU 연산 {t_gpu:4.1f}초, 1스텝 {t_gpu/steps:.2f}초) "
            f"GPU 메모리 {torch.mps.driver_allocated_memory()/1e9:.1f}GB")

    # 추론: 실제 크기 전체 프레임
    model.eval(); inf = []
    with torch.no_grad():
        for p in pool[:40]:
            img = _normalize(_rescale(_load_gray(p), mm[p], Image.BILINEAR))
            h, w = img.shape; img = np.pad(img, ((0, -h % 32), (0, -w % 32)))
            x = torch.from_numpy(img)[None, None].float().to(dev)
            torch.mps.synchronize(); t = time.time(); model(x); torch.mps.synchronize(); inf.append(time.time() - t)

    first, last = np.mean(ep_times[1:6]), np.mean(ep_times[-5:])
    log("\n=== 요약 ===")
    log(f"epoch 평균: 초반(2~6) {first:.1f}초 → 후반(마지막 5) {last:.1f}초, 변화 {100*(last/first-1):+.0f}% (발열 저하 지표)")
    log(f"추론: 전체 프레임 1장 {1000*np.median(inf[5:]):.0f}ms")
    for ep_n in (40, 60, 80):
        log(f"1회 학습 {ep_n} epoch: 약 {last*ep_n/60:.0f}분 (검증 시간 제외)")
    log(f"1차 탐색 22회(조기 종료로 평균 50 epoch 가정): 약 {22*last*50/3600:.1f}시간")
    (ROOT/"reports/bench_real.txt").write_text("\n".join(lines), encoding="utf-8")
    print("저장:", ROOT/"reports/bench_real.txt")

if __name__ == "__main__":
    main()
