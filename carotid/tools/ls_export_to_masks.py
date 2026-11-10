"""Label Studio 내보내기(JSON) → PNG 마스크 + 등급 반영 (계획서 10절).

- 이미지 이름(001_P003_V03_frame_0215.png)으로 manifest/label_<batch>.csv 에서 원래 프레임을 찾음
  (Label Studio가 업로드 시 붙이는 'abcd1234-' 접두어는 자동 제거)
- 다각형(% 좌표) → 원본 크기 마스크(0/255) 저장: data/labels/<video_id>/<frame>_mask.png  (src/dataset.py 규칙)
- 등급 A/B/C, 제외 사유, 메모를 manifest/frames.csv 에 반영. A·B → label_status=labeled, C → excluded_c
- lead_time(초)으로 장당 작업 시간 집계 → reports/labeling_<batch>.txt
사용: python tools/ls_export_to_masks.py <export.json> [--batch batch01]
"""
import argparse, csv, json, re
from pathlib import Path
from statistics import mean, median
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"(\d{3}_P\d{3}_V\d{2}_frame_\d+\.png)$")

def first(results, name, key):
    for r in results:
        if r.get("from_name") == name and r.get("value", {}).get(key):
            return r["value"][key]
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("export"); ap.add_argument("--batch", default="batch01")
    a = ap.parse_args()
    batch = {r["label_image"]: r for r in csv.DictReader(open(ROOT/f"manifest/label_{a.batch}.csv", encoding="utf-8"))}
    frames = list(csv.DictReader(open(ROOT/"manifest/frames.csv", encoding="utf-8")))
    by_file = {r["file"]: r for r in frames}
    tasks = json.load(open(a.export, encoding="utf-8"))
    stats = {"A": 0, "B": 0, "C": 0, "등급 없음": 0, "다각형 없음(A/B)": 0, "매칭 실패": 0}
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
        if grade is None:
            stats["등급 없음"] += 1; problems.append(f"등급 없음: {m.group(1)}"); continue
        stats[grade] += 1
        fr["grade"], fr["exclude_reason"], fr["note"] = grade, reason if grade == "C" else "", note
        if grade == "C":
            fr["label_status"] = "excluded_c"; continue
        polys = [r for r in res if r.get("type") == "polygonlabels"]
        if not polys:
            stats["다각형 없음(A/B)"] += 1; problems.append(f"A/B인데 다각형 없음: {m.group(1)}")
            fr["label_status"] = "needs_fix"; continue
        W = polys[0]["original_width"]; H = polys[0]["original_height"]
        with Image.open(ROOT/b["file"]) as im:
            if im.size != (W, H):
                problems.append(f"크기 불일치 {m.group(1)}: 이미지 {im.size}, 라벨 {(W, H)}")
        mask = Image.new("L", (W, H), 0); d = ImageDraw.Draw(mask)
        for p in polys:
            pts = [(x * W / 100.0, y * H / 100.0) for x, y in p["value"]["points"]]
            if len(pts) >= 3: d.polygon(pts, fill=255)
        out = ROOT/"data/labels"/b["video_id"]/(Path(b["file"]).stem + "_mask.png")
        out.parent.mkdir(parents=True, exist_ok=True); mask.save(out)
        fr["label_status"] = "labeled"
    with open(ROOT/"manifest/frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(frames[0])); w.writeheader(); w.writerows(frames)
    lines = [f"라벨 결과 ({a.batch})", *(f"  {k}: {v}" for k, v in stats.items())]
    if times:
        per = mean(times)
        lines += [f"장당 작업 시간: 평균 {per:.0f}초, 중앙값 {median(times):.0f}초 (n={len(times)})",
                  f"  → batch01 78장 약 {per*78/60:.0f}분, 1단계 전체 약 270장 약 {per*270/3600:.1f}시간,"
                  f" 최종 약 420장 약 {per*420/3600:.1f}시간",
                  "  → 계획서 9절: 장당 2분 이상이면 batch02를 사람당 12장으로 줄임"
                  + (" ← 해당" if per >= 120 else " (해당 없음)")]
    lines += ["확인 필요:"] + [f"  {p}" for p in problems] if problems else ["확인 필요: 없음"]
    out = ROOT/f"reports/labeling_{a.batch}.txt"; out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print("저장:", out)

if __name__ == "__main__":
    main()
