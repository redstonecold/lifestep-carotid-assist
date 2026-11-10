"""data/frames 의 실제 파일과 manifest/videos.csv 로 manifest/frames.csv 재생성.
기존 grade/label_status/exclude_reason 값은 유지한다."""
import csv, re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
vids = {r["video_id"]: r for r in csv.DictReader(open(root/"manifest/videos.csv", encoding="utf-8"))}
old = {}
fp = root/"manifest/frames.csv"
if fp.exists():
    for r in csv.DictReader(open(fp, encoding="utf-8")): old[r["file"]] = r
rows = []
for vid, v in vids.items():
    if v["use"] != "Y": continue
    fps = float(v["fps"])
    for f in sorted((root/"data/frames"/vid).glob("*.png")):
        i = int(re.search(r"_f(\d+)\.png$", f.name).group(1))
        rel = f"data/frames/{vid}/{f.name}"; t = round(i/fps, 3)
        o = old.get(rel, {})
        reason = o.get("exclude_reason", "")
        if vid == "V13" and t < 0.8 and not reason: reason = "탐촉자 조정 구간(0.8초 이전)"
        rows.append({"person_id": v["person_id"], "video_id": vid, "frame_idx": i, "time_s": t, "file": rel,
                     "grade": o.get("grade", ""), "label_status": o.get("label_status", "unlabeled"), "exclude_reason": reason})
with open(fp, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
c = Counter((r["person_id"], r["video_id"]) for r in rows if not r["exclude_reason"])
for k in sorted(c): print(*k, c[k])
print("total usable", sum(c.values()), "excluded", sum(1 for r in rows if r["exclude_reason"]))
