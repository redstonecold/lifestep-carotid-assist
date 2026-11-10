"""
검증·평가 지표 (계획서 11절 `검증 지표`, 설명 자료 부록 C).

보고 지표 14종
  겹침   Dice · IoU · 정밀도 · 민감도                       (A·B·C)
  경계   HD95 · ASSD (mm)                                 (A·B·C)
  측정   내경 MAE · 편향 · Bland-Altman 일치 한계 · ICC(2,1) (A·C)
         박동별 변화량 MAE                                  (박동 분석 후, report.py)
  검출   F 오검출률 · 미검출률                               (F / A·B·C)
  안정성 프레임 간 내경 변동                                 (라벨 없는 연속 프레임)

결정 지표는 Dice · F 오검출률 · 내경 MAE 3개 (train.py `select_best`).

프레임별 원자료(TP·FP·FN·TN, HD, NSD, 내경 등)는 frame_metrics()가 모두 돌려준다.
보고서에서 뺀 지표(특이도·정확도·면적 유사도·HD·NSD·RMSE·r·AUC)는 이 원자료로 다시 계산할 수 있다.
"""

import math

import cv2
import numpy as np

AREA_MIN_MM2 = 10.0            # 이 넓이 이상 칠하면 "검출" (판단값, 계획서 11절)
NSD_TAUS_MM  = (0.3, 1.0)      # 원자료로만 저장
DIAM_MIN_PX  = 50              # 내경을 잴 최소 내강 픽셀 수
NAN = float("nan")


# ── 프레임 단위 ──────────────────────────────────────────────────────────────

def confusion(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(p.size - tp - fp - fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _boundary(mask: np.ndarray) -> np.ndarray:
    """내강 경계 픽셀. 화면 가장자리는 경계로 치지 않는다(C세트처럼 혈관이 화면 밖으로 이어지는 경우)."""
    m = mask.astype(np.uint8)
    return m & (1 - cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1))


def _dist_to(boundary: np.ndarray) -> np.ndarray:
    """각 픽셀에서 가장 가까운 경계 픽셀까지의 거리(픽셀)."""
    return cv2.distanceTransform((1 - boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, mm: float) -> dict:
    """HD · HD95 · ASSD · NSD (mm). 예측이 비면 영상 대각선 길이를 벌점으로 넣는다(부록 C-0)."""
    bp, bg = _boundary(pred), _boundary(gt)
    out = {}
    if not bg.any():
        return {"hd": NAN, "hd95": NAN, "assd": NAN, **{f"nsd_{t}": NAN for t in NSD_TAUS_MM}}
    if not bp.any():
        diag = math.hypot(*pred.shape) * mm
        return {"hd": diag, "hd95": diag, "assd": diag, **{f"nsd_{t}": 0.0 for t in NSD_TAUS_MM}}
    d_pg = _dist_to(bg)[bp.astype(bool)] * mm     # 예측 경계 → 정답 경계
    d_gp = _dist_to(bp)[bg.astype(bool)] * mm     # 정답 경계 → 예측 경계
    out["hd"]   = float(max(d_pg.max(), d_gp.max()))
    out["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    out["assd"] = float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)))
    for t in NSD_TAUS_MM:
        out[f"nsd_{t}"] = float((np.count_nonzero(d_pg <= t) + np.count_nonzero(d_gp <= t))
                                / (len(d_pg) + len(d_gp)))
    return out


def diameter_mm(mask: np.ndarray, mm: float) -> float:
    """
    내경(mm). AI 마스크와 수동 마스크에 같은 방법을 쓴다(부록 C-0).
      1. 가장 큰 연결 성분만 사용
      2. 주성분 분석으로 혈관 축 방향을 구함(중심선의 직선 근사)
      3. 축을 따라 1픽셀 간격으로 자르고, 각 위치에서 축에 수직인 방향의 내강 폭을 잼
      4. 축 길이의 가운데 50%(중앙 안정 구간)의 폭 중앙값 × mm/px
    """
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return NAN
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < DIAM_MIN_PX:
        return NAN
    ys, xs = np.nonzero(lab == k)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    u, v = vecs[:, 1], vecs[:, 0]                  # 축 방향, 수직 방향
    t = np.round((pts - c) @ u).astype(int)
    q = (pts - c) @ v
    lo, hi = np.percentile(t, 25), np.percentile(t, 75)
    widths = []
    for ti in range(int(lo), int(hi) + 1):
        sel = q[t == ti]
        if len(sel):
            widths.append(sel.max() - sel.min() + 1)
    return float(np.median(widths) * mm) if widths else NAN


def frame_metrics(pred: np.ndarray, gt: np.ndarray, mm: float, grade: str) -> dict:
    """한 프레임의 원자료 + 지표. pred·gt는 원래 해상도의 0/1 마스크."""
    c = confusion(pred, gt)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    row = dict(c)
    row["area_pred_mm2"] = (tp + fp) * mm * mm
    row["area_gt_mm2"]   = (tp + fn) * mm * mm
    row["detected"]      = int(row["area_pred_mm2"] >= AREA_MIN_MM2)
    if grade == "F":
        # 정답이 비어 있거나 일부뿐이라 겹침·경계·내경은 정의하지 않는다. 검출 여부만 본다.
        row.update(dice=NAN, iou=NAN, precision=NAN, recall=NAN, diam_ai=NAN, diam_gt=NAN,
                   **{k: NAN for k in ("hd", "hd95", "assd", *(f"nsd_{t}" for t in NSD_TAUS_MM))})
        return row
    row["dice"]      = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else NAN
    row["iou"]       = tp / (tp + fp + fn) if (tp + fp + fn) else NAN
    row["precision"] = tp / (tp + fp) if (tp + fp) else NAN
    row["recall"]    = tp / (tp + fn) if (tp + fn) else NAN
    row.update(boundary_metrics(pred, gt, mm))
    row["diam_gt"] = diameter_mm(gt, mm) if grade in ("A", "C") else NAN
    row["diam_ai"] = diameter_mm(pred, mm) if grade in ("A", "C") else NAN
    return row


# ── 집계 통계 ────────────────────────────────────────────────────────────────

def _nanmean(x) -> float:
    x = np.asarray([v for v in x if v == v], dtype=float)
    return float(x.mean()) if len(x) else NAN


def bland_altman(ai, gt) -> dict:
    """편향 ± 1.96 SD (Bland & Altman 1986)."""
    e = np.asarray(ai, float) - np.asarray(gt, float)
    e = e[~np.isnan(e)]
    if len(e) < 2:
        return {"bias": NAN, "loa_low": NAN, "loa_high": NAN}
    b, sd = float(e.mean()), float(e.std(ddof=1))
    return {"bias": b, "loa_low": b - 1.96 * sd, "loa_high": b + 1.96 * sd}


def icc_2_1(ai, gt) -> float:
    """ICC(2,1) 이원 무작위, 절대일치, 단일 측정 (Shrout & Fleiss 1979; McGraw & Wong 1996)."""
    x = np.stack([np.asarray(ai, float), np.asarray(gt, float)], 1)
    x = x[~np.isnan(x).any(1)]
    n, k = x.shape
    if n < 3:
        return NAN
    gm = x.mean()
    ss_r = k * ((x.mean(1) - gm) ** 2).sum()
    ss_c = n * ((x.mean(0) - gm) ** 2).sum()
    ss_e = ((x - gm) ** 2).sum() - ss_r - ss_c
    ms_r, ms_c, ms_e = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / den) if den > 0 else NAN


def summarize(rows: list[dict]) -> dict:
    """프레임 원자료 → 보고 지표 14종 중 프레임 라벨로 계산하는 12종 + valid 결정 지표."""
    seg  = [r for r in rows if r["grade"] in ("A", "B", "C")]
    meas = [r for r in rows if r["grade"] in ("A", "C")
            and r["diam_ai"] == r["diam_ai"] and r["diam_gt"] == r["diam_gt"]]
    fset = [r for r in rows if r["grade"] == "F"]
    ai = [r["diam_ai"] for r in meas]
    gt = [r["diam_gt"] for r in meas]
    out = {
        "n_frames": len(rows), "n_seg": len(seg), "n_meas": len(meas), "n_f": len(fset),
        "dice": _nanmean(r["dice"] for r in seg),
        "iou": _nanmean(r["iou"] for r in seg),
        "precision": _nanmean(r["precision"] for r in seg),
        "recall": _nanmean(r["recall"] for r in seg),
        "hd95": _nanmean(r["hd95"] for r in seg),
        "assd": _nanmean(r["assd"] for r in seg),
        "diam_mae": _nanmean(abs(a - g) for a, g in zip(ai, gt)),
        "icc": icc_2_1(ai, gt),
        "f_fp_rate": (sum(r["detected"] for r in fset) / len(fset)) if fset else NAN,
        "miss_rate": (sum(1 - r["detected"] for r in seg) / len(seg)) if seg else NAN,
    }
    out.update(bland_altman(ai, gt))   # bias, loa_low, loa_high
    return out


def summarize_by_grade(rows: list[dict]) -> dict[str, dict]:
    return {g: summarize([r for r in rows if r["grade"] == g]) for g in ("A", "B", "C", "F")}


def bootstrap_ci(rows: list[dict], key: str, group: str | None = None,
                 n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """
    summarize()[key]의 95% percentile bootstrap 신뢰구간.
    group='person_id'면 사람 단위로 재표집한다(교차검증: 같은 사람 프레임끼리의 상관 반영).
    """
    rng = np.random.default_rng(seed)
    if group:
        units: dict[str, list] = {}
        for r in rows:
            units.setdefault(r[group], []).append(r)
        keys = list(units)
        draw = lambda: [r for k in rng.choice(keys, len(keys)) for r in units[k]]
    else:
        draw = lambda: [rows[i] for i in rng.integers(0, len(rows), len(rows))]
    vals = [summarize(draw())[key] for _ in range(n_boot)]
    vals = np.asarray([v for v in vals if v == v])
    if not len(vals):
        return NAN, NAN
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def temporal_jitter(diams) -> float:
    """프레임 간 내경 변동 J (mm): 5프레임 이동평균을 뺀 고주파 성분의 RMS (부록 C-5)."""
    a = np.asarray(diams, float)
    a = a[~np.isnan(a)]
    if len(a) < 5:
        return NAN
    smooth = np.convolve(a, np.ones(5) / 5, mode="valid")
    return float(np.sqrt(np.mean((a[2:-2] - smooth) ** 2)))
