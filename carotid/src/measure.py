"""
영상 한 개 → 프레임별 내경 → 3단계 신뢰도 판정 → 박동별 요약 (계획서 12절, 2026-11-18 차별점).

신뢰도 판정
  1. 프레임: 칠한 넓이가 없거나(혈관 없음·F), 길쭉하지 않거나(횡단면), 모델 확신도가 낮으면 제외
  2. 시간  : 한 장으로는 못 잡는 탐촉자 이동·흔들림. 내경 변화 속도, 내강 중심의 세로 이동 속도,
             마스크 겹침(IoU)이 생리적 범위를 넘으면 제외(앞뒤 몇 프레임까지 함께).
             속도는 RATE_LAG_S 이상 떨어진 프레임과 비교한다(프레임 간격이 13~50ms로 고르지 않아 바로 앞과 비교하면 부풀려짐).
             가로 위치와 혈관 길이는 혈관이 화면 끝에서 잘리는 위치에 따라 흔들리므로 쓰지 않는다
  3. 박동  : 내경 신호를 박동으로 나누고, 제외 프레임 비율·박동 길이·변화량이 다른 박동과 크게 다르면 `검토 필요`
믿을 수 있는 박동만 요약값(최소·최대 내경, 변화량, 심박수)에 쓴다. 믿을 수 있는 박동이 2개 미만이면 `자동측정 불가`.

박동 구분: ECG 없이 내경 신호의 주기를 쓴다. 탐촉자 압력 변화 같은 몇 초 단위의 느린 변화(기저선)를 뺀 신호에서
최대 내경(수축기) 봉우리를 찾고, 이웃 봉우리 사이의 최소값을 이완기로 본다. 최소·최대 내경 값은 기저선을 빼기 전 신호에서 잰다.
프레임 시각은 frames.csv의 time_s(DICOM FrameTimeVector 누적값)를 쓴다. 간격이 일정하지 않으므로 균일 격자로 보간해 신호 처리한다.

사용:
    python -m src.measure --video V20 --checkpoint runs/<설정>/best.pt [--out runs/measure]
출력: <out>/<video>/ result.json · frames.csv · masks/ · diameter.png
실행 위치: carotid/ 폴더
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import AREA_MIN_MM2, diameter_geometry
from src.predict import predict_files

# 1단계: 프레임
CONF_MIN = 0.75             # 칠한 영역의 평균 확률이 이보다 낮으면 경계를 믿기 어려움
ELONG_MIN = 3.0             # 내강의 긴 축/짧은 축 비. 사람 라벨 A·B·C 442장 중 3 미만은 약 1%, 횡단면 구간은 중앙값 2.6
# 2단계: 시간 (생리적 변화보다 훨씬 빠른 변화 = 탐촉자 이동·흔들림)
RATE_LAG_S = 0.04           # 속도 계산 시 비교할 프레임 간격(이상)
DIAM_RATE_MAX = 8.0         # mm/s. 박동 중 내경 변화는 약 0.5mm/0.15s ≈ 3mm/s
CENTER_RATE_MAX = 10.0      # mm/s. 내강 중심의 세로 이동 속도
IOU_MIN = 0.80              # 연속 프레임 마스크 겹침 (두 마스크가 모두 있는 열 범위 안에서)
FLAG_PAD_S = 0.06           # 걸린 프레임 앞뒤로 이 시간만큼 함께 제외
# 3단계: 박동
GRID_HZ = 100.0
SMOOTH_S = 0.12
BASELINE_S = 1.5            # 기저선(이동 중앙값) 창. 심박 주기(약 1초)보다 길게
MIN_BEAT_S, MAX_BEAT_S = 0.35, 2.0      # 심박 30~170 bpm
BEAT_VALID_MIN = 0.85       # 박동 안의 믿을 수 있는 프레임 비율
BEAT_LEN_TOL = 0.35         # 박동 길이가 중앙값에서 ±35% 넘게 벗어나면 검토
BEAT_AMP_MAD = 3.0          # 변화량이 중앙값에서 MAD×3(최소 0.15mm) 넘게 벗어나면 검토
MIN_RELIABLE_BEATS = 2

REASON_KO = {"no_vessel": "혈관 없음", "not_long": "종단면 아님", "low_conf": "확신도 낮음", "diam_jump": "내경 급변",
             "motion": "탐촉자 이동", "shape": "모양 급변", "beat_valid": "제외 프레임 많음",
             "beat_len": "박동 길이 이상", "beat_amp": "변화량 이상", "edge": "영상 끝 박동"}


def _frame_features(masks, confs, mm):
    feats, prev = [], None
    for m, c in zip(masks, confs):
        b = m > 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
        if n > 2:   # 가장 큰 덩어리만(화면 끝 잡음 조각 제외)
            b = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        area = b.sum() * mm * mm
        f = {"area_mm2": float(area), "conf": c, "diam_mm": float("nan"), "cx_mm": float("nan"),
             "cy_mm": float("nan"), "iou_prev": float("nan"), "elong": float("nan")}
        if area >= AREA_MIN_MM2:
            ys, xs = np.nonzero(b)
            ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64)))
            g = diameter_geometry(b, mm)
            f.update(diam_mm=g["diam_mm"] if g else float("nan"), cx_mm=float(xs.mean() * mm), cy_mm=float(ys.mean() * mm),
                     elong=float(np.sqrt(ev[1] / max(ev[0], 1e-6))))
            if g:   # 화면에 측정 위치를 그리기 위한 값(앱). 소수 1~3자리로 줄여 저장
                f["geo"] = {"c": np.round(g["c"], 1).tolist(), "u": np.round(g["u"], 4).tolist(),
                            "v": np.round(g["v"], 4).tolist(), "lo": round(g["lo"], 1), "hi": round(g["hi"], 1),
                            "med": np.round(g["med"], 1).tolist(), "cal": np.round(g["cal"], 1).tolist()}
            if prev is not None and prev.any():
                cols = b.any(0) & prev.any(0)
                if cols.any():
                    bb, pp = b[:, cols], prev[:, cols]
                    f["iou_prev"] = float((bb & pp).sum() / (bb | pp).sum())
        feats.append(f)
        prev = b
    return feats


def _flag_frames(feats, t):
    """1·2단계 판정. 프레임마다 이유 목록(비어 있으면 믿을 수 있음)."""
    n = len(feats)
    reasons = [[] for _ in range(n)]
    for i, f in enumerate(feats):
        if f["area_mm2"] < AREA_MIN_MM2 or not np.isfinite(f["diam_mm"]):
            reasons[i].append("no_vessel")
        elif f["elong"] < ELONG_MIN:
            reasons[i].append("not_long")
        elif f["conf"] < CONF_MIN:
            reasons[i].append("low_conf")
    hit = np.zeros(n, bool)
    why = [set() for _ in range(n)]
    j = 0
    for i in range(1, n):
        while j + 1 < i and t[i] - t[j + 1] >= RATE_LAG_S:
            j += 1
        a, b = feats[j], feats[i]
        if "no_vessel" in reasons[i] or "no_vessel" in reasons[i - 1] or t[i] - t[j] < RATE_LAG_S:
            continue
        dt = t[i] - t[j]
        if "no_vessel" not in reasons[j]:
            if abs(b["diam_mm"] - a["diam_mm"]) / dt > DIAM_RATE_MAX:
                why[i].add("diam_jump")
            if abs(b["cy_mm"] - a["cy_mm"]) / dt > CENTER_RATE_MAX:
                why[i].add("motion")
        if b["iou_prev"] < IOU_MIN:
            why[i].add("shape")
        hit[i] = bool(why[i])
    for i in np.nonzero(hit)[0]:            # 앞뒤로 넓혀서 제외
        for j in range(n):
            if abs(t[j] - t[i]) <= FLAG_PAD_S:
                for r in why[i]:
                    if r not in reasons[j]:
                        reasons[j].append(r)
    return reasons


def _segments(flags, t, min_len=1):
    """연속으로 걸린 프레임을 구간으로 묶는다."""
    segs, i, n = [], 0, len(flags)
    while i < n:
        if not flags[i]:
            i += 1; continue
        j = i
        rs = set(flags[i])
        while j + 1 < n and flags[j + 1]:
            j += 1; rs |= set(flags[j])
        if j - i + 1 >= min_len:
            segs.append({"start": i, "end": j, "t0": float(t[i]), "t1": float(t[j]),
                         "frame": (i + j) // 2, "reasons": sorted(rs)})
        i = j + 1
    return segs


def _beats(d, ok, t):
    """3단계: 내경 신호(믿을 수 있는 프레임만)를 박동으로 나누고 박동별 판정."""
    if ok.sum() < 10:
        return [], None
    grid = np.arange(t[0], t[-1], 1 / GRID_HZ)
    sig = np.interp(grid, t[ok], d[ok])
    win = max(5, int(SMOOTH_S * GRID_HZ) | 1)
    if len(sig) > win:
        sig = savgol_filter(sig, win, 2)
    k = int(BASELINE_S * GRID_HZ) | 1
    pad = np.pad(sig, k // 2, mode="edge")
    base = np.array([np.median(pad[i:i + k]) for i in range(len(sig))])
    hp = sig - base
    prom = max(0.05, 0.4 * (np.percentile(hp, 90) - np.percentile(hp, 10)))
    peaks, _ = find_peaks(hp, distance=int(MIN_BEAT_S * GRID_HZ), prominence=prom)
    beats = []
    for k in range(len(peaks) - 1):     # 이웃 봉우리 사이 최저점 = 이완기 → 그다음 봉우리까지 한 박동
        a, b = peaks[k], peaks[k + 1]
        lo = a + int(np.argmin(hp[a:b + 1]))
        beats.append({"trough": lo, "peak": b})
    out = []
    for k, bt in enumerate(beats):
        t_start = grid[bt["trough"]]
        t_end = grid[beats[k + 1]["trough"]] if k + 1 < len(beats) else grid[min(len(grid) - 1, bt["peak"] + (bt["peak"] - bt["trough"]))]
        sel = (t >= t_start) & (t <= t_end)
        valid = float(ok[sel].mean()) if sel.any() else 0.0
        out.append({"t0": float(t_start), "t1": float(t_end), "dur_s": float(t_end - t_start),
                    "dmin": float(sig[bt["trough"]]), "dmax": float(sig[bt["peak"]]),
                    "delta": float(sig[bt["peak"]] - sig[bt["trough"]]), "valid_frac": valid,
                    "frame0": int(np.argmin(np.abs(t - t_start))), "frame_peak": int(np.argmin(np.abs(t - grid[bt["peak"]]))),
                    "reasons": [] if k + 1 < len(beats) else ["edge"]})
    if out:
        dur = np.median([b["dur_s"] for b in out if "edge" not in b["reasons"]] or [b["dur_s"] for b in out])
        amps = np.array([b["delta"] for b in out])
        med, mad = np.median(amps), np.median(np.abs(amps - np.median(amps)))
        for b in out:
            if b["valid_frac"] < BEAT_VALID_MIN:
                b["reasons"].append("beat_valid")
            if abs(b["dur_s"] - dur) > BEAT_LEN_TOL * dur or not (MIN_BEAT_S <= b["dur_s"] <= MAX_BEAT_S):
                b["reasons"].append("beat_len")
            if abs(b["delta"] - med) > max(BEAT_AMP_MAD * 1.4826 * mad, 0.15):
                b["reasons"].append("beat_amp")
            b["reliable"] = not b["reasons"]
    return out, {"grid": grid, "sig": sig}


def analyze(frames: list[Path], times: np.ndarray, mm: float, checkpoint: Path, mask_dir: Path | None = None,
            on_progress=None, return_masks: bool = False) -> dict:
    confs: list = []
    masks = predict_files(frames, checkpoint, mm_per_px=mm, confs=confs, out_dir=mask_dir, on_progress=on_progress)
    res = measure_masks(masks, confs, times, mm)
    res["confs"] = confs
    return {**res, **({"_masks": masks} if return_masks else {})}


def measure_masks(masks: list, confs: list, times: np.ndarray, mm: float, edited: set | None = None) -> dict:
    """마스크(모델 결과 또는 사람이 고친 것)에서 측정·판정·박동 요약. 모델을 다시 돌리지 않는다.
    edited: 사람이 경계를 고친 프레임 번호. 확신도 판정에서 빼고(사람 확인) 결과에 표시한다."""
    edited = edited or set()
    confs = [1.0 if i in edited else c for i, c in enumerate(confs)]
    feats = _frame_features(masks, confs, mm)
    for i in edited:
        feats[i]["edited"] = True
    t = np.asarray(times, float)
    flags = _flag_frames(feats, t)
    d = np.array([f["diam_mm"] for f in feats])
    ok = np.array([not r for r in flags]) & np.isfinite(d)
    beats, sig = _beats(d, ok, t)
    good = [b for b in beats if b["reliable"]]
    summary = {"n_frames": len(masks), "frames_ok": int(ok.sum()), "n_beats": len(beats), "n_reliable": len(good),
               "status": "자동측정" if len(good) >= MIN_RELIABLE_BEATS else "자동측정 불가"}
    if len(good) >= MIN_RELIABLE_BEATS:
        summary.update(dmin_mm=float(np.median([b["dmin"] for b in good])),
                       dmax_mm=float(np.median([b["dmax"] for b in good])),
                       delta_mm=float(np.median([b["delta"] for b in good])),
                       hr_bpm=float(60 / np.median([b["dur_s"] for b in good])))
    review = [{**s, "kind": "구간", "reasons_ko": [REASON_KO[r] for r in s["reasons"]]}
              for s in _segments([[r for r in f if r != "no_vessel"] for f in flags], t)]
    review += [{"kind": "박동", "start": b["frame0"], "frame": b["frame_peak"], "t0": b["t0"], "t1": b["t1"],
                "reasons": b["reasons"], "reasons_ko": [REASON_KO[r] for r in b["reasons"]]}
               for b in beats if not b["reliable"] and "edge" not in b["reasons"]]
    novessel = _segments([["no_vessel"] if "no_vessel" in f else [] for f in flags], t)
    return {"summary": summary, "beats": beats, "review": sorted(review, key=lambda r: r["t0"]),
            "no_vessel": novessel, "frames": [{**f, "t": float(ti), "reasons": r} for f, ti, r in zip(feats, t, flags)],
            "signal": {"t": sig["grid"][::5].round(3).tolist(), "d": sig["sig"][::5].round(3).tolist()} if sig else None,
            "params": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}}


def plot(res: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = res["frames"]
    t = np.array([f["t"] for f in fr]); d = np.array([f["diam_mm"] for f in fr])
    bad = np.array([bool(f["reasons"]) for f in fr])
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=120)
    for s in res["review"]:
        if s["kind"] == "구간":
            ax.axvspan(s["t0"], s["t1"], color="#eda100", alpha=0.25, lw=0)
    for s in res["no_vessel"]:
        ax.axvspan(s["t0"], s["t1"], color="#999999", alpha=0.25, lw=0)
    ax.plot(t[~bad], d[~bad], ".", ms=3, color="#2a78d6", label="frame diameter (reliable)")
    ax.plot(t[bad], d[bad], "x", ms=3, color="#e34948", label="excluded frame")
    if res["signal"]:
        ax.plot(res["signal"]["t"], res["signal"]["d"], "-", lw=1, color="#0b0b0b", alpha=0.6, label="smoothed")
    for b in res["beats"]:
        ax.axvline(b["t0"], color="#1baf7a" if b["reliable"] else "#e34948", lw=0.8, alpha=0.7)
    s = res["summary"]
    sub = (f"{s['status']}  beats {s['n_reliable']}/{s['n_beats']} reliable" +
           (f"  |  min {s['dmin_mm']:.2f}  max {s['dmax_mm']:.2f}  Δ {s['delta_mm']:.2f} mm  HR {s['hr_bpm']:.0f}"
            if "dmin_mm" in s else ""))
    ax.set_title(f"{title}\n{sub}".replace("자동측정 불가", "NOT MEASURABLE").replace("자동측정", "MEASURED"), fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("diameter (mm)")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def video_frames(video_id: str):
    rows = [r for r in csv.DictReader(open(ROOT / "manifest/frames.csv", encoding="utf-8")) if r["video_id"] == video_id]
    rows.sort(key=lambda r: int(r["frame_idx"]))
    mm = {r["video_id"]: float(r["mm_per_px"]) for r in csv.DictReader(open(ROOT / "manifest/videos.csv", encoding="utf-8"))}[video_id]
    return [ROOT / r["file"] for r in rows], np.array([float(r["time_s"]) for r in rows]), mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="영상 ID (예: V20)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="runs/measure")
    a = ap.parse_args()
    out = ROOT / a.out / a.video
    out.mkdir(parents=True, exist_ok=True)
    frames, times, mm = video_frames(a.video)
    res = analyze(frames, times, mm, ROOT / a.checkpoint, mask_dir=out / "masks")
    json.dump(res, open(out / "result.json", "w"), ensure_ascii=False, indent=1)
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "t", "area_mm2", "diam_mm", "conf", "iou_prev", "elong", "reasons"])
        for i, r in enumerate(res["frames"]):
            w.writerow([i, r["t"], round(r["area_mm2"], 2), round(r["diam_mm"], 3), round(r["conf"], 3),
                        round(r["iou_prev"], 3), round(r["elong"], 2), "|".join(r["reasons"])])
    plot(res, out / "diameter.png", f"{a.video}  ({len(frames)} frames)")
    s = res["summary"]
    print(json.dumps(s, ensure_ascii=False))
    print(f"검토 필요 {len(res['review'])}곳, 혈관 없음 구간 {len(res['no_vessel'])}곳 → {out}")


if __name__ == "__main__":
    main()
