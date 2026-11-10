"""
보고서용 표·그림 생성 (계획서 11절 `보고서 산출물`).

검증 결과(학습 중 저장된 프레임 원자료)로 표·그림 만들기:
    python -m src.report --frames runs/<설정>/val_frames.csv --history runs/<설정>/history.csv --out reports/results/val

교차검증 9명 (사람 단위 bootstrap, 사람별 표·상자그림):
    python -m src.report --frames runs/cv_<설정>_frames.csv --by-person --out reports/results/cv

시험 P004 최종 평가 (마지막에 한 번만):
    python -m src.report --evaluate --checkpoint runs/<설정>/best.pt --split test --out reports/results/test

반복 라벨 기준선 (표 6, AI 결과와 같은 프레임끼리 비교):
    python -m src.report --repeat --frames runs/<설정>/val_frames.csv --out reports/results/repeat

프레임 간 내경 변동 (라벨 없는 연속 프레임):
    python -m src.report --jitter --checkpoint runs/<설정>/best.pt --video V20 --start 280 --out reports/results/val

출력: 표는 CSV + Markdown(tables.md), 그림은 PNG.
박동별 변화량 MAE(표 2)는 박동 분할(계획서 12절) 구현 후 추가한다.
실행 위치: carotid/ 폴더
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (summarize, summarize_by_grade, bootstrap_ci, frame_metrics,
                         temporal_jitter, diameter_mm)

# (키, 표시 이름, 단위, 프레임별 값 계산) — 프레임별 값이 있는 지표만 SD·중앙값·IQR을 낸다
FRAME_METRICS = [
    ("dice", "Dice", "", lambda r: r["dice"]),
    ("iou", "IoU", "", lambda r: r["iou"]),
    ("precision", "정밀도", "", lambda r: r["precision"]),
    ("recall", "민감도", "", lambda r: r["recall"]),
    ("hd95", "HD95", "mm", lambda r: r["hd95"]),
    ("assd", "ASSD", "mm", lambda r: r["assd"]),
    ("diam_mae", "내경 MAE", "mm", lambda r: abs(r["diam_ai"] - r["diam_gt"])),
]
SET_METRICS = [
    ("bias", "내경 편향", "mm"), ("loa_low", "일치 한계 하한", "mm"), ("loa_high", "일치 한계 상한", "mm"),
    ("icc", "ICC(2,1)", ""), ("f_fp_rate", "F 오검출률", ""), ("miss_rate", "미검출률", ""),
]
TABLES = {
    "table1_segmentation": ["dice", "iou", "precision", "recall", "hd95", "assd"],
    "table2_measurement": ["diam_mae", "bias", "loa_low", "loa_high", "icc"],
    "table3_detection": ["f_fp_rate", "miss_rate"],
}
STR_KEYS = {"person_id", "video_id", "file", "grade"}


def load_frames(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append({k: (v if k in STR_KEYS else float(v) if v not in ("", "nan") else float("nan"))
                     for k, v in r.items()})
    return rows


def _scope(rows, key):
    if key in ("dice", "iou", "precision", "recall", "hd95", "assd"):
        return [r for r in rows if r["grade"] in ("A", "B", "C")]
    if key == "diam_mae":
        return [r for r in rows if r["grade"] in ("A", "C") and r["diam_ai"] == r["diam_ai"] and r["diam_gt"] == r["diam_gt"]]
    return rows


def metric_table(rows: list[dict], group: str | None, n_boot: int) -> list[dict]:
    """지표마다 값 · SD · 중앙값 [IQR] · 95% CI · n."""
    s = summarize(rows)
    out = []
    names = {k: (n, u) for k, n, u, _ in FRAME_METRICS} | {k: (n, u) for k, n, u in SET_METRICS}
    fn = {k: f for k, _, _, f in FRAME_METRICS}
    for key, (name, unit) in names.items():
        lo, hi = bootstrap_ci(rows, key, group=group, n_boot=n_boot)
        row = {"지표": name, "단위": unit, "값": s[key], "95% CI 하한": lo, "95% CI 상한": hi}
        if key in fn:
            v = np.array([fn[key](r) for r in _scope(rows, key)], float)
            v = v[~np.isnan(v)]
            if len(v):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                row.update({"SD": v.std(ddof=1) if len(v) > 1 else float("nan"),
                            "중앙값": med, "IQR 하한": q1, "IQR 상한": q3, "n(프레임)": len(v)})
        row["key"] = key
        out.append(row)
    return out


def _fmt(v):
    return "" if v is None else (f"{v:.3f}" if isinstance(v, float) and v == v else ("-" if isinstance(v, float) else str(v)))


def _md(title, rows):
    if not rows:
        return f"### {title}\n\n(자료 없음)\n"
    cols = [c for c in rows[0] if c != "key"]
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _write(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r if k != "key"))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


# ── 그림 ─────────────────────────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fig_bland_altman(rows, path):
    plt = _plt()
    m = _scope(rows, "diam_mae")
    if len(m) < 2:
        return
    a = np.array([r["diam_ai"] for r in m]); g = np.array([r["diam_gt"] for r in m])
    s = summarize(rows)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter((a + g) / 2, a - g, s=14, alpha=0.7)
    for y, ls, lab in ((s["bias"], "-", f"편향 {s['bias']:.3f}"), (s["loa_low"], "--", f"하한 {s['loa_low']:.3f}"),
                       (s["loa_high"], "--", f"상한 {s['loa_high']:.3f}")):
        ax.axhline(y, ls=ls, color="k", lw=1); ax.text(ax.get_xlim()[1], y, f" {lab}", va="center", fontsize=8)
    ax.set_xlabel("(AI + 수동) / 2  내경 (mm)"); ax.set_ylabel("AI - 수동 (mm)")
    ax.set_title(f"Bland-Altman (A·C, n={len(m)})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_scatter(rows, path):
    plt = _plt()
    m = _scope(rows, "diam_mae")
    if len(m) < 2:
        return
    a = np.array([r["diam_ai"] for r in m]); g = np.array([r["diam_gt"] for r in m])
    lo, hi = min(a.min(), g.min()) - 0.3, max(a.max(), g.max()) + 0.3
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(g, a, s=14, alpha=0.7); ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("수동 내경 (mm)"); ax.set_ylabel("AI 내경 (mm)"); ax.legend(loc="upper left")
    ax.set_title(f"내경 산점도 (A·C, n={len(m)})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_person_box(rows, path):
    plt = _plt()
    persons = sorted({r["person_id"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, label) in zip(axes, (("dice", "Dice (A·B·C)"), ("diam_mae", "내경 절대오차 (mm, A·C)"))):
        f = dict((k, fn) for k, _, _, fn in FRAME_METRICS)[key]
        data = [[v for v in (f(r) for r in _scope(rows, key) if r["person_id"] == p) if v == v] for p in persons]
        ax.boxplot(data, tick_labels=persons); ax.set_title(label); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_learning_curve(history_csv, path):
    plt = _plt()
    h = list(csv.DictReader(open(history_csv, encoding="utf-8")))
    e = [int(r["epoch"]) for r in h]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(e, [float(r["train_loss"]) for r in h], label="train loss")
    ax1.plot(e, [float(r["valid_loss"]) for r in h], label="valid loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2 = ax1.twinx(); ax2.plot(e, [float(r["dice"]) for r in h], "g-", label="valid Dice"); ax2.set_ylabel("Dice")
    fig.legend(loc="upper right"); ax1.set_title("학습 곡선")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_examples(rows, images, preds, gts, path):
    """그림 5: 세트별 Dice 최고·최저(F는 예측 면적 최소·최대) 프레임에 경계 겹쳐 보기. 초록=정답, 빨강=AI."""
    import cv2
    plt = _plt()
    picks = []
    for g in ("A", "B", "C", "F"):
        idx = [i for i, r in enumerate(rows) if r["grade"] == g]
        if not idx:
            continue
        key = (lambda i: -rows[i]["area_pred_mm2"]) if g == "F" else (lambda i: rows[i]["dice"] if rows[i]["dice"] == rows[i]["dice"] else -1)
        idx.sort(key=key, reverse=True)
        picks += [(g, "좋은 예", idx[0]), (g, "나쁜 예", idx[-1])]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks) // 2, 2, figsize=(10, 3 * len(picks) // 2), squeeze=False)
    for ax, (g, lab, i) in zip(axes.ravel(), picks):
        rgb = np.dstack([images[i]] * 3)
        for m, col in ((gts[i], (0, 1, 0)), (preds[i], (1, 0, 0))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            canvas = np.zeros_like(rgb); cv2.drawContours(canvas, cs, -1, col, 2)
            rgb = np.where(canvas.any(2, keepdims=True), canvas, rgb)
        ax.imshow(rgb); ax.axis("off")
        d = rows[i]["dice"]
        ax.set_title(f"{g} {lab} · {rows[i]['person_id']} {rows[i]['video_id']} #{int(rows[i]['frame_idx'])}"
                     + (f" · Dice {d:.3f}" if d == d else f" · 예측 {rows[i]['area_pred_mm2']:.0f}mm²"), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ── 실행 모드 ────────────────────────────────────────────────────────────────

def build_report(rows, out: Path, by_person: bool, n_boot: int, history: Path | None):
    out.mkdir(parents=True, exist_ok=True)
    group = "person_id" if by_person else None
    full = metric_table(rows, group, n_boot)
    md = [f"# 결과 표 (프레임 {len(rows)}장, 사람 {len({r['person_id'] for r in rows})}명, "
          f"신뢰구간: {'사람' if by_person else '프레임'} 단위 bootstrap {n_boot}회)\n"]
    titles = {"table1_segmentation": "표 1. 분할 성능 (A·B·C)",
              "table2_measurement": "표 2. 측정 일치도 (A·C) — 박동별 변화량 MAE는 박동 분할 구현 후 추가",
              "table3_detection": "표 3. 검출 성능"}
    for name, keys in TABLES.items():
        t = [r for r in full if r["key"] in keys]
        _write(out / f"{name}.csv", t); md.append(_md(titles[name], t))
    # 표 4: 세트별
    t4 = [{"세트": g, **{k: v for k, v in s.items()}} for g, s in summarize_by_grade(rows).items()]
    _write(out / "table4_by_grade.csv", t4); md.append(_md("표 4. 세트별 성능 (n_*는 장수)", t4))
    # 표 5: 사람별 · 영상 그룹별
    t5 = [{"사람": p, **summarize([r for r in rows if r["person_id"] == p])}
          for p in sorted({r["person_id"] for r in rows})]
    if by_person:
        t5.append({"사람": "사람별 평균의 평균",
                   **{k: float(np.nanmean([r[k] for r in t5])) for k in t5[0] if k != "사람"}})
    _write(out / "table5_by_person.csv", t5); md.append(_md("표 5. 사람별 성능", t5))
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    fig_bland_altman(rows, out / "fig1_bland_altman.png")
    fig_scatter(rows, out / "fig2_diameter_scatter.png")
    if by_person:
        fig_person_box(rows, out / "fig3_person_box.png")
    if history:
        fig_learning_curve(history, out / "fig4_learning_curve.png")
    print(f"저장: {out}")


def run_evaluate(checkpoint: Path, split: str):
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import CarotidDataset, eval_collate, _load_gray
    from src.model import build_model, get_device, load_checkpoint
    from src.train import evaluate
    device = get_device()
    model = build_model(pretrained=False).to(device)
    meta = load_checkpoint(model, checkpoint, device=device)
    ds = CarotidDataset(ROOT, split, augment=None)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=eval_collate)
    rows, summ, vloss, probs = evaluate(model, dl, device, meta.get("cfg", {}).get("loss", "dice_bce"), keep_probs=True)
    images = [_load_gray(s["img_path"]) for s in ds.samples]
    gts = [(_load_gray(s["mask_path"]) > 0.5).astype(np.uint8) for s in ds.samples]
    preds = [(p > 0.5).astype(np.uint8) for p in probs]
    return rows, images, preds, gts


def run_repeat(frames_csv: Path | None, out: Path):
    """표 6: 반복 라벨(data/labels_repeat) vs 원 라벨 = 사람-사람 차이. AI 결과가 있으면 같은 프레임으로 비교."""
    from src.dataset import _load_gray
    mm = {r["video_id"]: float(r["mm_per_px"]) for r in csv.DictReader(open(ROOT / "manifest/videos.csv", encoding="utf-8"))}
    grade = {r["file"]: r["grade"] for r in csv.DictReader(open(ROOT / "manifest/frames.csv", encoding="utf-8"))}
    human = []
    for p in sorted((ROOT / "data/labels_repeat").glob("*/*_mask.png")):
        vid, stem = p.parent.name, p.name.replace("_mask.png", "")
        orig = ROOT / "data/labels" / vid / p.name
        f = next((k for k in grade if k.endswith(f"/{vid}/{stem}.png")), None)
        if not orig.exists() or f is None or grade[f] not in ("A", "C"):
            continue
        r1 = (_load_gray(orig) > 0.5).astype(np.uint8); r2 = (_load_gray(p) > 0.5).astype(np.uint8)
        human.append({"person_id": f.split("/")[2], "video_id": vid, "file": f, "grade": grade[f],
                      **frame_metrics(r2, r1, mm[vid], grade[f])})
    if not human:
        print("반복 라벨이 없습니다 (data/labels_repeat/). ls_export_to_masks.py --repeat 로 먼저 변환하세요."); return
    rows = [{"비교": "사람 vs 사람 (반복 라벨)", **summarize(human)}]
    if frames_csv:
        files = {h["file"] for h in human}
        ai = [r for r in load_frames(frames_csv) if r["file"] in files]
        if ai:
            rows.append({"비교": "AI vs 사람 (같은 프레임)", **summarize(ai)})
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "table6_ai_vs_repeat.csv", rows)
    (out / "table6.md").write_text(_md(f"표 6. AI vs 반복 라벨 (프레임 {len(human)}장)", rows), encoding="utf-8")
    print(f"저장: {out / 'table6_ai_vs_repeat.csv'}")


def run_jitter(checkpoint: Path, video: str, start: int, out: Path):
    from src.predict import predict_folder
    mmpx = {r["video_id"]: float(r["mm_per_px"]) for r in csv.DictReader(open(ROOT / "manifest/videos.csv", encoding="utf-8"))}[video]
    frame_dir = next((ROOT / "data/frames").glob(f"*/{video}"))
    masks = predict_folder(frame_dir, checkpoint, mm_per_px=mmpx, start=start)
    diams = [diameter_mm(m > 0, mmpx) for m in masks]
    j = temporal_jitter(diams)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"jitter_{video}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["frame_idx", "diam_ai_mm"])
        w.writerows((start + i, d) for i, d in enumerate(diams))
    (out / f"jitter_{video}.txt").write_text(f"{video} #{start}~ 프레임 {len(diams)}장  J = {j:.4f} mm\n", encoding="utf-8")
    print(f"{video}: 프레임 간 내경 변동 J = {j:.4f} mm  → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, help="프레임 원자료 CSV (val_frames.csv 또는 cv_*_frames.csv)")
    ap.add_argument("--history", type=Path, help="history.csv (그림 4 학습 곡선)")
    ap.add_argument("--by-person", action="store_true", help="교차검증: 사람 단위 bootstrap·사람별 표")
    ap.add_argument("--evaluate", action="store_true", help="체크포인트로 val/test를 다시 평가")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--repeat", action="store_true", help="표 6 반복 라벨 기준선")
    ap.add_argument("--jitter", action="store_true", help="프레임 간 내경 변동")
    ap.add_argument("--video"); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.repeat:
        return run_repeat(a.frames, a.out)
    if a.jitter:
        return run_jitter(a.checkpoint, a.video, a.start, a.out)
    if a.evaluate:
        if a.split == "test":
            print("[주의] 시험 P004 평가는 모든 결정이 끝난 뒤 한 번만 한다(계획서 6절).")
        rows, images, preds, gts = run_evaluate(a.checkpoint, a.split)
        a.out.mkdir(parents=True, exist_ok=True)
        _write(a.out / f"{a.split}_frames.csv", rows)
        fig_examples(rows, images, preds, gts, a.out / "fig5_examples.png")
    else:
        rows = load_frames(a.frames)
    build_report(rows, a.out, a.by_person, a.n_boot, a.history)


if __name__ == "__main__":
    main()
