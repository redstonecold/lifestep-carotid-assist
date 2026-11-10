"""
학습 루프 — 단일 실행 또는 하이퍼파라미터 탐색.

단일 실행 (기본 설정):
    python -m src.train

특정 설정 하나:
    python -m src.train --config lr3e-4_dice_bce

전체 탐색 (중단 후 재실행해도 완료된 세트는 건너뜀):
    python -m src.train --sweep

실행 위치: carotid/ 폴더
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


# ── 손실 함수 ─────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    return 1 - (2 * (pred * target).sum() + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    d = dice_loss(pred, target)
    if loss_type == "dice_bce":
        return d + nn.functional.binary_cross_entropy_with_logits(pred, target)
    if loss_type == "dice_focal":
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        focal = ((1 - torch.exp(-bce)) ** 2 * bce).mean()
        return d + focal
    return d


# ── 평가 지표 ─────────────────────────────────────────────────────────────────

def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum().item()
    total = pred_bin.sum().item() + target.sum().item()
    return (2 * intersection / total) if total > 0 else 1.0


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum().item()
    union = ((pred_bin + target) > 0).float().sum().item()
    return (intersection / union) if union > 0 else 1.0


# ── 단일 설정 학습 ────────────────────────────────────────────────────────────

def train_one_config(cfg: dict, run_dir: Path) -> dict:
    from src.dataset import CarotidDataset, BalancedPersonSampler
    from src.model import build_model, get_device, save_checkpoint

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = get_device()
    print(f"\n{'='*60}")
    print(f"설정: {cfg['name']}  |  device: {device}")
    print(f"lr={cfg['lr']}  loss={cfg['loss']}  batch={cfg['batch_size']}")
    print(f"{'='*60}")

    train_ds = CarotidDataset(ROOT, split="train", augment=True)
    val_ds   = CarotidDataset(ROOT, split="val",   augment=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"[SKIP] 라벨된 프레임 없음 — 학습: {len(train_ds)}, 검증: {len(val_ds)}")
        print("       먼저 Label Studio로 프레임에 마스크를 그린 뒤 label_status='labeled'로 업데이트하세요.")
        return {"name": cfg["name"], "skipped": True, "best_val_dice": None}

    print(f"학습 프레임: {len(train_ds)}  |  검증 프레임: {len(val_ds)}")

    sampler  = BalancedPersonSampler(train_ds, samples_per_person=cfg.get("samples_per_person", 50))
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
    # 검증은 영상마다 크기가 달라 1장씩 (전체 프레임 평가)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    model    = build_model(pretrained=cfg.get("pretrained", True)).to(device)
    freeze_ep = cfg.get("freeze_encoder_epochs", 0)
    if freeze_ep > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print(f"인코더 {freeze_ep} epoch 고정")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["lr"], weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"], eta_min=1e-6,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = run_dir / "best.pt"
    best_dice  = 0.0
    no_improve = 0
    history    = []

    for epoch in range(1, cfg["max_epochs"] + 1):

        # 인코더 고정 해제
        if freeze_ep > 0 and epoch == freeze_ep + 1:
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            for g in optimizer.param_groups:
                g["lr"] = cfg["lr"]
            print(f"  [epoch {epoch}] 인코더 고정 해제")

        # 학습
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(imgs), masks, cfg["loss"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        scheduler.step()

        # 검증
        model.eval()
        val_dice_sum = val_iou_sum = 0.0
        with torch.no_grad():
            for imgs, masks, (h, w) in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                h, w = int(h), int(w)
                preds = model(imgs)[..., :h, :w]   # 패딩 영역 제외
                masks = masks[..., :h, :w]
                val_dice_sum += compute_dice(preds, masks)
                val_iou_sum  += compute_iou(preds,  masks)
        val_dice = val_dice_sum / len(val_dl)
        val_iou  = val_iou_sum  / len(val_dl)

        elapsed = time.time() - t0
        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "val_dice": round(val_dice, 4), "val_iou": round(val_iou, 4),
               "elapsed_s": round(elapsed, 1)}
        history.append(row)
        print(f"  epoch {epoch:3d}/{cfg['max_epochs']}"
              f"  loss={train_loss:.4f}  dice={val_dice:.4f}  iou={val_iou:.4f}"
              f"  ({elapsed:.0f}s)")

        if val_dice > best_dice:
            best_dice  = val_dice
            no_improve = 0
            save_checkpoint(model, best_ckpt,
                            meta={"epoch": epoch, "val_dice": best_dice, "cfg": cfg})
            print(f"    -> 최선 모델 저장 (dice={best_dice:.4f})")
        else:
            no_improve += 1

        if no_improve >= cfg.get("patience", 15):
            print(f"  조기 종료 (patience={cfg['patience']}, epoch={epoch})")
            break

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n완료: best_dice={best_dice:.4f}  ({len(history)} epochs)")
    return {"name": cfg["name"], "best_val_dice": round(best_dice, 4),
            "epochs_run": len(history), "skipped": False}


# ── 탐색 설정 목록 ────────────────────────────────────────────────────────────
# 플랜 10절 1차 탐색 기준 (총 22회).
# 2단계·3단계 설정은 1단계 결과를 보고 최선 lr/loss를 채워 넣는다.

SWEEP_CONFIGS: list[dict] = [
    # 1단계: lr × loss (6회)
    {"name": "lr3e-4_dice_bce",   "lr": 3e-4, "loss": "dice_bce",   "batch_size": 8, "max_epochs": 80, "patience": 15},
    {"name": "lr1e-3_dice_bce",   "lr": 1e-3, "loss": "dice_bce",   "batch_size": 8, "max_epochs": 80, "patience": 15},
    {"name": "lr3e-3_dice_bce",   "lr": 3e-3, "loss": "dice_bce",   "batch_size": 8, "max_epochs": 80, "patience": 15},
    {"name": "lr3e-4_dice_focal", "lr": 3e-4, "loss": "dice_focal", "batch_size": 8, "max_epochs": 80, "patience": 15},
    {"name": "lr1e-3_dice_focal", "lr": 1e-3, "loss": "dice_focal", "batch_size": 8, "max_epochs": 80, "patience": 15},
    {"name": "lr3e-3_dice_focal", "lr": 3e-3, "loss": "dice_focal", "batch_size": 8, "max_epochs": 80, "patience": 15},
    # 3단계: 사전학습 효과 (2회) — lr/loss는 1단계 최선값으로 교체 예정
    {"name": "pretrained_off",    "lr": 1e-3, "loss": "dice_bce",   "batch_size": 8, "max_epochs": 80, "patience": 15, "pretrained": False},
    {"name": "freeze5ep",         "lr": 1e-3, "loss": "dice_bce",   "batch_size": 8, "max_epochs": 80, "patience": 15, "freeze_encoder_epochs": 5},
]

DEFAULT_CONFIG: dict = {
    "name": "default",
    "lr": 1e-3, "loss": "dice_bce", "batch_size": 8,
    "max_epochs": 80, "patience": 15, "pretrained": True,
}


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep",  action="store_true", help="전체 탐색 루프 실행")
    ap.add_argument("--config", default=None,        help="단일 설정 이름 (예: lr1e-3_dice_bce)")
    args = ap.parse_args()

    runs_root   = ROOT / "runs"
    results_csv = runs_root / "sweep_results.csv"
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        configs = SWEEP_CONFIGS
    elif args.config:
        configs = [c for c in SWEEP_CONFIGS if c["name"] == args.config]
        if not configs:
            print(f"설정 '{args.config}' 없음. 사용 가능: {[c['name'] for c in SWEEP_CONFIGS]}")
            return
    else:
        configs = [DEFAULT_CONFIG]

    # 기존 결과 불러오기 (중단 재개)
    all_results: list[dict] = []
    done_names:  set[str]   = set()
    if results_csv.exists():
        for row in csv.DictReader(open(results_csv, encoding="utf-8")):
            done_names.add(row["name"])
            all_results.append(row)

    for cfg in configs:
        if cfg["name"] in done_names:
            print(f"[sweep] SKIP {cfg['name']} (이미 완료)")
            continue
        result = train_one_config(cfg, runs_root / cfg["name"])
        all_results.append(result)
        # 완료 즉시 저장 (중단해도 결과 유지)
        fieldnames = list(all_results[0].keys())
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)

    # 결과 요약
    ranked = sorted(
        [r for r in all_results if r.get("best_val_dice") is not None],
        key=lambda r: -float(r["best_val_dice"]),
    )
    if ranked:
        print("\n=== 탐색 결과 (val_dice 내림차순) ===")
        for r in ranked:
            print(f"  {r['name']:30s}  val_dice={r['best_val_dice']}")


if __name__ == "__main__":
    main()
