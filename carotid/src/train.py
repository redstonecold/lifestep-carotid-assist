"""
학습 루프 — 단일 실행, 하이퍼파라미터 탐색, 교차검증, 학습 곡선 (계획서 11·15절).

구조 (Dataset 2개 · loop 2개):
  train Dataset : 학습 9명, A·B·C·F, 증강, 256 크롭, 사람별 균형 추출
  valid Dataset : 검증 P006, A·B·C·F, 증강 없음, 전체 프레임
  epoch마다 train loop → valid loop(지표 전부) → history.csv 기록 → 최선 체크포인트 저장

단일 실행 (기본 설정):
    python -m src.train

탐색 1차 23회 (7단계. 앞 단계 결과로 다음 단계 설정을 자동 결정):
    caffeinate -i python -m src.train --sweep

교차검증 9:1 순환 9회 / 학습 곡선 25%·50% (탐색으로 고른 설정 사용):
    caffeinate -i python -m src.train --cv
    caffeinate -i python -m src.train --learning-curve

중단 후 다시 실행하면 끝난 설정은 건너뛰고, 학습 중이던 설정은 마지막 epoch부터 이어서 한다.
실행 위치: carotid/ 폴더
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import (CarotidDataset, BalancedPersonSampler, eval_collate,
                         GRADE_WEIGHT, TRAIN_PERSONS, VAL_PERSONS)
from src.metrics import frame_metrics, summarize

WARMUP_EPOCHS = 3
DIVERGE_WINDOW = 10      # 처음 10 epoch 안에
DIVERGE_RISES  = 3       # train loss가 3 epoch 연속 오르면 발산으로 보고 중단

# 결과표·history에 남기는 검증 지표 (metrics.summarize 키)
METRIC_KEYS = ["dice", "iou", "precision", "recall", "hd95", "assd",
               "diam_mae", "bias", "loa_low", "loa_high", "icc", "f_fp_rate", "miss_rate"]


# ── 손실 함수 ─────────────────────────────────────────────────────────────────

def batch_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """배치 전체 픽셀을 합쳐 한 번 계산. F세트 빈 마스크가 섞여도 0/0이 되지 않는다."""
    p = torch.sigmoid(logits).reshape(-1)
    g = target.reshape(-1)
    return 1 - (2 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth)


def pixel_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, kind: str) -> torch.Tensor:
    """세트 가중치를 곱한 BCE 또는 Focal(γ=2). weight는 장마다 하나 (A·C·F 1.0, B 0.7)."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if kind == "dice_focal":
        bce = (1 - torch.exp(-bce)) ** 2 * bce
    w = weight.view(-1, 1, 1, 1).expand_as(bce)
    return (w * bce).sum() / w.sum()


def combined_loss(logits, target, weight, kind: str) -> torch.Tensor:
    return batch_dice_loss(logits, target) + pixel_loss(logits, target, weight, kind)


# ── 평가 (valid loop) ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, loss_kind: str, keep_probs: bool = False):
    """
    검증·시험 데이터 전체를 평가한다.
    Returns: (프레임별 원자료 rows, 요약 지표 dict, valid loss[, 원래 해상도 확률맵 list])

    - 지표: 예측 확률을 원래 해상도로 되돌린 뒤 0.5로 자르고, 원본 마스크와 비교(mm는 영상별 mm/px)
    - valid loss: 0.1mm/px 격자에서 검증셋 전체를 한 배치처럼 계산(batch Dice + 가중 BCE/Focal)
    """
    model.eval()
    rows, probs = [], []
    inter = psum = gsum = 0.0
    wl_sum = w_sum = 0.0
    for img, mask, mask0, meta in loader:
        img, mask = img.to(device), mask.to(device)
        h, w = meta["h"], meta["w"]
        logits = model(img)[..., :h, :w]
        mask = mask[..., :h, :w]
        p = torch.sigmoid(logits)
        inter += float((p * mask).sum()); psum += float(p.sum()); gsum += float(mask.sum())
        bce = nn.functional.binary_cross_entropy_with_logits(logits, mask, reduction="none")
        if loss_kind == "dice_focal":
            bce = (1 - torch.exp(-bce)) ** 2 * bce
        wl_sum += meta["weight"] * float(bce.sum()); w_sum += meta["weight"] * bce.numel()

        prob0 = cv2.resize(p[0, 0].cpu().numpy(), (mask0.shape[1], mask0.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        pred0 = (prob0 > 0.5).astype(np.uint8)
        row = {k: meta[k] for k in ("person_id", "video_id", "frame_idx", "file", "grade")}
        row.update(frame_metrics(pred0, mask0.astype(np.uint8), meta["mm_per_px"], meta["grade"]))
        rows.append(row)
        if keep_probs:
            probs.append(prob0)
    valid_loss = (1 - (2 * inter + 1) / (psum + gsum + 1)) + (wl_sum / max(w_sum, 1e-9))
    out = (rows, summarize(rows), valid_loss)
    return out + (probs,) if keep_probs else out


# ── 단일 설정 학습 ────────────────────────────────────────────────────────────

DEFAULTS = dict(lr=1e-3, loss="dice_bce", batch_size=8, max_epochs=80, patience=15,
                pretrained=True, freeze_encoder_epochs=0, aug="medium", crop=256,
                clahe=False, b_weight=0.7, seed=0, samples_per_person=50, fraction=1.0)


def _seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _lr_lambda(max_epochs: int, lr: float, eta_min: float = 1e-6):
    """짧은 선형 워밍업(3 epoch) 후 코사인 감소."""
    def f(e):
        if e < WARMUP_EPOCHS:
            return (e + 1) / WARMUP_EPOCHS
        t = (e - WARMUP_EPOCHS) / max(1, max_epochs - WARMUP_EPOCHS)
        return eta_min / lr + (1 - eta_min / lr) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def train_one_config(cfg: dict, run_dir: Path, root: Path = ROOT,
                     train_persons: set | None = None, val_persons: set | None = None) -> dict:
    from src.model import build_model, get_device, save_checkpoint

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    cfg = {**DEFAULTS, **cfg}
    _seed_all(cfg["seed"])
    device = get_device()
    gw = {**GRADE_WEIGHT, "B": cfg["b_weight"]}

    train_ds = CarotidDataset(root, "train", augment=cfg["aug"], persons=train_persons or TRAIN_PERSONS,
                              grade_weight=gw, crop_size=cfg["crop"], clahe=cfg["clahe"],
                              fraction=cfg["fraction"], seed=cfg["seed"])
    val_ds   = CarotidDataset(root, "val", augment=None, persons=val_persons or VAL_PERSONS, grade_weight=gw)

    base = {"name": cfg["name"], "group": cfg.get("group", cfg["name"]), "cfg_json": json.dumps(cfg)}
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"[SKIP] {cfg['name']}: 라벨된 프레임 없음 — 학습 {len(train_ds)}, 검증 {len(val_ds)}")
        return {**base, "status": "skipped"}

    data_version = f"{train_ds.data_version()}-{val_ds.data_version()}"
    print(f"\n{'='*60}\n설정: {cfg['name']}  |  device: {device}  |  데이터 {data_version}")
    print(f"lr={cfg['lr']} loss={cfg['loss']} aug={cfg['aug']} crop={cfg['crop']} "
          f"B={cfg['b_weight']} seed={cfg['seed']} | 학습 {len(train_ds)}장 · 검증 {len(val_ds)}장\n{'='*60}")

    sampler  = BalancedPersonSampler(train_ds, samples_per_person=cfg["samples_per_person"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=eval_collate)

    model = build_model(pretrained=cfg["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(cfg["max_epochs"], cfg["lr"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt, best_ckpt = run_dir / "last.pt", run_dir / "best.pt"
    state = {"epoch": 0, "best_dice": -1.0, "best_epoch": 0, "best_summary": {}, "no_improve": 0,
             "history": [], "data_version": data_version}

    # 중단 재개: 같은 데이터일 때만 이어서 한다
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        if ck["state"]["data_version"] == data_version:
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"]); state = ck["state"]
            random.setstate(ck["rng"]["py"]); np.random.set_state(ck["rng"]["np"])
            torch.set_rng_state(ck["rng"]["torch"])
            print(f"  [재개] epoch {state['epoch']}부터 이어서 학습")
        else:
            print(f"  [경고] 라벨이 바뀌었습니다({ck['state']['data_version']} → {data_version}). "
                  f"이어서 하지 않고 처음부터 학습합니다.")

    status = "done"
    for epoch in range(state["epoch"] + 1, cfg["max_epochs"] + 1):
        freeze = epoch <= cfg["freeze_encoder_epochs"]
        for p in model.encoder.parameters():
            p.requires_grad_(not freeze)

        # ① train loop
        model.train()
        t0 = time.time()
        train_loss, n = 0.0, 0
        for imgs, masks, weights in train_dl:
            imgs, masks, weights = imgs.to(device), masks.to(device), weights.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(imgs), masks, weights, cfg["loss"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item(); n += 1
        train_loss /= max(n, 1)
        scheduler.step()

        # ② valid loop
        rows, summ, valid_loss = evaluate(model, val_dl, device, cfg["loss"])

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
               "valid_loss": valid_loss, **{k: summ[k] for k in METRIC_KEYS}, "elapsed_s": time.time() - t0}
        state["history"].append(row)
        print(f"  epoch {epoch:3d}/{cfg['max_epochs']}  loss {train_loss:.4f}/{valid_loss:.4f}"
              f"  dice {summ['dice']:.4f}  mae {summ['diam_mae']:.3f}mm  F오검출 {summ['f_fp_rate']:.2f}"
              f"  ({row['elapsed_s']:.0f}s)")

        dice = summ["dice"] if summ["dice"] == summ["dice"] else -1.0
        if dice > state["best_dice"]:
            state.update(best_dice=dice, best_epoch=epoch, best_summary=summ, no_improve=0)
            save_checkpoint(model, best_ckpt, meta={"epoch": epoch, "val_dice": dice, "cfg": cfg,
                                                     "data_version": data_version})
            _write_csv(run_dir / "val_frames.csv", rows)
        else:
            state["no_improve"] += 1
        state["epoch"] = epoch
        _write_csv(run_dir / "history.csv", state["history"])
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state,
                    "rng": {"py": random.getstate(), "np": np.random.get_state(),
                            "torch": torch.get_rng_state()}}, last_ckpt)

        losses = [h["train_loss"] for h in state["history"]]
        rises = 0
        for a, b in zip(losses[-DIVERGE_RISES - 1:-1], losses[-DIVERGE_RISES:]):
            rises = rises + 1 if b > a else 0
        if not math.isfinite(train_loss) or (epoch <= DIVERGE_WINDOW and len(losses) > DIVERGE_RISES
                                             and rises >= DIVERGE_RISES):
            print(f"  [발산] epoch {epoch}에서 중단 (손실 {train_loss})")
            status = "diverged"
            break
        if state["no_improve"] >= cfg["patience"]:
            print(f"  조기 종료 (patience={cfg['patience']}, epoch={epoch})")
            break

    print(f"완료: best epoch {state['best_epoch']}  dice={state['best_dice']:.4f}  ({status})")
    return {**base, "status": status, "best_epoch": state["best_epoch"], "epochs_run": state["epoch"],
            "data_version": data_version,
            **{k: state["best_summary"].get(k, float("nan")) for k in METRIC_KEYS}}


# ── 설정 선택 규칙 (계획서 11절 `선택 기준`, 사전 고정) ─────────────────────────

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def rank_groups(results: list[dict], groups: set | None = None) -> list[str]:
    """
    설정(그룹, 시드 반복은 평균)을 좋은 순서로 정렬한다.
      1. Dice가 최고보다 0.01 이상 낮으면 탈락
      2. F 오검출률 20% 초과 탈락 (F 프레임이 없으면 통과)
      3. 내경 MAE 오름차순, 0.02mm 이내면 HD95 오름차순, 그래도 같으면 먼저 정의된(단순한) 설정
    탈락한 설정은 뒤에 Dice 순으로 붙인다.
    """
    agg: dict[str, dict] = {}
    order = []
    for r in results:
        if r.get("status") not in ("done", "diverged") or (groups and r["group"] not in groups):
            continue
        g = agg.setdefault(r["group"], {k: [] for k in ("dice", "f_fp_rate", "diam_mae", "hd95")})
        if r["group"] not in order:
            order.append(r["group"])
        for k in g:
            g[k].append(_f(r.get(k)))
    mean = lambda v: float(np.nanmean(v)) if any(x == x for x in v) else float("nan")
    m = {g: {k: mean(v) for k, v in d.items()} for g, d in agg.items()}
    if not m:
        return []
    best_dice = max((v["dice"] for v in m.values() if v["dice"] == v["dice"]), default=float("nan"))
    ok = [g for g in order if m[g]["dice"] >= best_dice - 0.01
          and not (m[g]["f_fp_rate"] > 0.2)]
    out = []
    for g in sorted(ok, key=lambda g: (m[g]["diam_mae"] if m[g]["diam_mae"] == m[g]["diam_mae"] else 1e9,
                                       order.index(g))):
        # MAE 차이 0.02mm 이내인 바로 앞 설정과는 HD95로 다시 비교
        if out and abs(m[g]["diam_mae"] - m[out[-1]]["diam_mae"]) <= 0.02 and m[g]["hd95"] < m[out[-1]]["hd95"]:
            out.insert(len(out) - 1, g)
        else:
            out.append(g)
    rest = sorted((g for g in order if g not in ok), key=lambda g: -(m[g]["dice"] if m[g]["dice"] == m[g]["dice"] else -1))
    return out + rest


def _cfg_of(results: list[dict], group: str) -> dict:
    for r in results:
        if r["group"] == group and r.get("cfg_json"):
            c = json.loads(r["cfg_json"])
            return {k: v for k, v in c.items() if k not in ("name", "group")}
    raise KeyError(group)


# ── 탐색 단계 (1차 23회) ──────────────────────────────────────────────────────

def stage_configs(stage: int, results: list[dict]) -> list[dict]:
    """앞 단계 결과로 이 단계의 설정 목록을 만든다."""
    names = lambda prefix: {r["group"] for r in results if r["group"].startswith(prefix)}
    if stage == 1:   # 학습률 × 손실
        return [{"name": f"s1_lr{lr}_{loss}", "lr": float(lr), "loss": loss}
                for loss in ("dice_bce", "dice_focal") for lr in ("3e-4", "1e-3", "3e-3")]
    if stage == 2:   # 증강 강도 × 입력 크기
        base = _cfg_of(results, rank_groups(results, names("s1"))[0])
        return [{**base, "name": f"s2_{aug}_{crop}", "aug": aug, "crop": crop}
                for aug in ("weak", "medium", "strong") for crop in (256, 384)]
    if stage == 3:   # 사전학습 활용
        base = _cfg_of(results, rank_groups(results, names("s1") | names("s2"))[0])
        return [{**base, "name": f"s3_freeze{k}", "freeze_encoder_epochs": k} for k in (0, 5)]
    if stage == 4:   # 재현성: 상위 2개 × 시드 3회
        top = rank_groups(results, names("s1") | names("s2") | names("s3"))[:2]
        return [{**_cfg_of(results, g), "name": f"{g}_seed{s}", "group": g, "seed": s}
                for g in top for s in (1, 2, 3)]
    final = rank_groups(results)[0]
    base = _cfg_of(results, final)
    if stage == 5:
        return [{**base, "name": "s5_clahe", "clahe": True}]
    if stage == 6:
        return [{**base, "name": "s6_scratch", "pretrained": False, "freeze_encoder_epochs": 0}]
    if stage == 7:
        return [{**base, "name": "s7_bweight1", "b_weight": 1.0}]
    raise ValueError(stage)


# ── 진입점 ────────────────────────────────────────────────────────────────────

def _load_results(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def _run_list(configs, results, results_csv, runs_root, root, overrides,
              persons_of=lambda c: (None, None)):
    done = {r["name"] for r in results if r.get("status") in ("done", "diverged")}
    for cfg in configs:
        cfg = {**cfg, **overrides}
        if cfg["name"] in done:
            print(f"[SKIP] {cfg['name']} (이미 완료)")
            continue
        tp, vp = persons_of(cfg)
        res = train_one_config(cfg, runs_root / cfg["name"], root, tp, vp)
        results[:] = [r for r in results if r["name"] != cfg["name"]] + [res]
        _write_csv(results_csv, results)   # 설정이 끝날 때마다 저장(중단해도 결과 유지)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="1차 탐색 23회(7단계)")
    ap.add_argument("--stages", default="1-7", help="탐색 단계 범위 (예: 1-3)")
    ap.add_argument("--cv", action="store_true", help="9:1 순환 교차검증 9회 (선택된 설정)")
    ap.add_argument("--learning-curve", action="store_true", help="학습 라벨 25%%·50%% (선택된 설정)")
    ap.add_argument("--final", default=None, help="교차검증·학습 곡선에 쓸 설정 그룹 (기본: 탐색 1등)")
    ap.add_argument("--root", default=str(ROOT), help="데이터 루트 (테스트용)")
    ap.add_argument("--runs", default="runs", help="결과 폴더 (root 기준)")
    ap.add_argument("--max-epochs", type=int, default=None, help="빠른 점검용 epoch 상한")
    ap.add_argument("--spp", type=int, default=None, help="빠른 점검용 사람당 epoch 샘플 수")
    args = ap.parse_args()

    root = Path(args.root)
    runs_root = root / args.runs
    runs_root.mkdir(parents=True, exist_ok=True)
    results_csv = runs_root / "sweep_results.csv"
    results = _load_results(results_csv)
    overrides = {k: v for k, v in (("max_epochs", args.max_epochs), ("samples_per_person", args.spp)) if v}

    if args.sweep:
        a, b = (int(x) for x in args.stages.split("-"))
        for st in range(a, b + 1):
            if not any(r.get("status") in ("done", "diverged") for r in results) and st > 1:
                print("[중단] 앞 단계 결과가 없습니다."); return
            print(f"\n##### 탐색 {st}단계 #####")
            _run_list(stage_configs(st, results), results, results_csv, runs_root, root, overrides)
        ranked = rank_groups(results)
        print("\n=== 선택 규칙에 따른 순위 ===")
        for i, g in enumerate(ranked, 1):
            print(f"  {i:2d}. {g}")
        return

    if args.cv or args.learning_curve:
        final = args.final or (rank_groups(results) or [None])[0]
        if not final:
            print("[중단] 탐색 결과가 없습니다. --final 로 설정 그룹을 지정하세요."); return
        base = _cfg_of(results, final)
        if args.cv:
            # 시험 P004 제외 10명. P003은 항상 학습, 나머지 9명이 한 번씩 검증자 → 매회 학습 9 : 검증 1
            pool = sorted((TRAIN_PERSONS | VAL_PERSONS) - {"P003"})
            cfgs = [{**base, "name": f"cv_{final}_{p}", "group": f"cv_{final}", "val_person": p} for p in pool]
            _run_list(cfgs, results, results_csv, runs_root, root, overrides,
                      persons_of=lambda c: ((TRAIN_PERSONS | VAL_PERSONS) - {c["val_person"]}, {c["val_person"]}))
            merged = []
            for c in cfgs:
                f = runs_root / c["name"] / "val_frames.csv"
                if f.exists():
                    merged += list(csv.DictReader(open(f, encoding="utf-8")))
            _write_csv(runs_root / f"cv_{final}_frames.csv", merged)
            print(f"교차검증 프레임 원자료: {runs_root / f'cv_{final}_frames.csv'}")
        if args.learning_curve:
            cfgs = [{**base, "name": f"lc{int(fr*100)}_{final}", "group": f"lc{int(fr*100)}_{final}",
                     "fraction": fr} for fr in (0.25, 0.5)]
            _run_list(cfgs, results, results_csv, runs_root, root, overrides)
        return

    _run_list([{"name": "default"}], results, results_csv, runs_root, root, overrides)


if __name__ == "__main__":
    main()
