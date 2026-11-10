"""Label Studio 내보내기(JSON) → PNG 마스크 + 세트 반영 (계획서 7·10절).

- 이미지 이름(001_P003_V03_frame_0215.png)으로 manifest/label_<batch>.csv 에서 원래 프레임을 찾음
  (Label Studio가 업로드 시 붙이는 'abcd1234-' 접두어는 자동 제거)
- 다각형(% 좌표) → 원본 크기 마스크(0/255) 저장: data/labels/<video_id>/<frame>_mask.png  (src/dataset.py 규칙)
- 세트 A/B/C/F/X, 제외 사유, 메모를 manifest/frames.csv 에 반영
    A·B·C → 다각형 필수, label_status=labeled
    F     → 다각형 선택(없으면 빈 마스크 저장), label_status=labeled  (검출 불가 학습용)
    X     → 마스크 없음, label_status=excluded_x
- lead_time(초)으로 장당 작업 시간 집계 → reports/labeling_<batch>.txt
- --repeat: 반복 라벨(같은 프레임 재라벨, 표 6 기준선). data/labels_repeat/ 에 저장하고 frames.csv는 바꾸지 않음
사용: python tools/ls_export_to_masks.py <export.json> [--batch batch01] [--repeat]
"""
import argparse, csv, json, re
from pathlib import Path
from statistics import mean, median
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"(\d{3}_P\d{3}_V\d{2}_frame_\d+\.png)$")
GRADES = ("A", "B", "C", "F", "X")
TOTAL_LABELS = 495 + 20   # 계획서 9절 목표 전체 + 반복 라벨

def first(results, name, key):
    for r in results:
        if r.get("from_name") == name and r.get("value", {}).get(key):
            return r["value"][key]
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("export"); ap.add_argument("--batch", default="batch01")
    ap.add_argument("--repeat", action="store_true", help="반복 라벨: data/labels_repeat/ 에 저장, frames.csv 미변경")
    a = ap.parse_args()
    batch = {r["label_image"]: r for r in csv.DictReader(open(ROOT/f"manifest/label_{a.batch}.csv", encoding="utf-8"))}
    frames = list(csv.DictReader(open(ROOT/"manifest/frames.csv", encoding="utf-8")))
    by_file = {r["file"]: r for r in frames}
    label_dir = ROOT/("data/labels_repeat" if a.repeat else "data/labels")
    tasks = json.load(open(a.export, encoding="utf-8"))
    stats = {**{g: 0 for g in GRADES}, "등급 없음": 0, "다각형 없음(A·B·C)": 0, "F 빈 마스크": 0, "매칭 실패": 0}
    times, problems = [], []
    for t in tasks:
        img = t.get("data", {}).get("image", "")
        m = NAME_RE.search(img.split("/")[-1])
        if not m or m.group(1) not in batch:
            stats["매칭 실패"] += 1; problems.append(f"매칭 실패: {img}"); continue
        b = batch[m.group(1)]; fr = by_file[b["file"]]
        anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            continue
        ann = max(anns, key=lambda x: x.get("updated_at") or x.get("created_at") or "")
        res = ann.get("result", [])
        grade = (first(res, "grade", "choices") or [None])[0]
        reason = (first(res, "exclude_reason", "choices") or [""])[0]
        note = (first(res, "note", "text") or [""])[0]
        if ann.get("lead_time"): times.append(float(ann["lead_time"]))
        if grade not in GRADES:
            stats["등급 없음"] += 1; problems.append(f"세트 없음/알 수 없음({grade}): {m.group(1)}"); continue
        stats[grade] += 1
        if not a.repeat:
            fr["grade"], fr["note"] = grade, note
            if grade == "X":   # 다른 세트는 기존 구간 표시(예: 종단면 구간 밖)를 그대로 둔다
                fr["exclude_reason"] = reason or "X"
        if grade == "X":
            if not a.repeat: fr["label_status"] = "excluded_x"
            continue
        polys = [r for r in res if r.get("type") == "polygonlabels"]
        with Image.open(ROOT/b["file"]) as im:
            W, H = im.size
        if polys and (polys[0]["original_width"], polys[0]["original_height"]) != (W, H):
            problems.append(f"크기 불일치 {m.group(1)}: 이미지 {(W, H)}, 라벨 {(polys[0]['original_width'], polys[0]['original_height'])}")
        if not polys and grade != "F":
            stats["다각형 없음(A·B·C)"] += 1; problems.append(f"{grade}인데 다각형 없음: {m.group(1)}")
            if not a.repeat: fr["label_status"] = "needs_fix"
            continue
        if not polys:
            stats["F 빈 마스크"] += 1
        mask = Image.new("L", (W, H), 0); d = ImageDraw.Draw(mask)
        for p in polys:
            pts = [(x * W / 100.0, y * H / 100.0) for x, y in p["value"]["points"]]
            if len(pts) >= 3: d.polygon(pts, fill=255)
        out = label_dir/b["video_id"]/(Path(b["file"]).stem + "_mask.png")
        out.parent.mkdir(parents=True, exist_ok=True); mask.save(out)
        if not a.repeat: fr["label_status"] = "labeled"
    if not a.repeat:
        with open(ROOT/"manifest/frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(frames[0])); w.writeheader(); w.writerows(frames)
    lines = [f"라벨 결과 ({a.batch}{', 반복 라벨' if a.repeat else ''})", *(f"  {k}: {v}" for k, v in stats.items())]
    n_abc = stats["A"] + stats["B"] + stats["C"]
    if n_abc:
        lines.append(f"  C 비율(A·B·C 중): {stats['C'] / n_abc:.0%}"
                     + ("  ← 50% 초과: A가 부족해질 수 있음(계획서 9절 A 45% 이상 목표)" if stats["C"] / n_abc > 0.5 else ""))
    if times:
        per = mean(times)
        lines += [f"장당 작업 시간: 평균 {per:.0f}초, 중앙값 {median(times):.0f}초 (n={len(times)})",
                  f"  → 전체 목표 {TOTAL_LABELS}장(학습·검증·시험 495 + 반복 20) 약 {per*TOTAL_LABELS/3600:.1f}시간"]
    lines += ["확인 필요:"] + [f"  {p}" for p in problems] if problems else ["확인 필요: 없음"]
    out = ROOT/f"reports/labeling_{a.batch}.txt"; out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print("저장:", out)

if __name__ == "__main__":
    main()
