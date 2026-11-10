"""라벨용 프레임 선정 (계획서 9·10절).

- batch01: P001~P003 사람당 26장(A 20 + B 최대 5 목표, C 발생 여유 포함), 총 78장
- batch02: 학습·검증 6명 사람당 15장, 시험 P004·P009 사람당 30장
- 영상별 배분은 ALLOC. 좋은 영상 위주, 나쁜 영상은 소량
- 영상 안에서는 사용 가능 구간(exclude_reason 없음)을 균등 시간 간격으로 나눠 이웃 프레임을 피함
- 사람을 번갈아 배치 → 처음 몇 장 안에 여러 사람이 섞임(장당 작업 시간 측정용)
- 라벨링용 이미지 이름: 001_P003_V03_frame_0123.png (변환 스크립트가 원래 프레임을 찾는 데 사용)
- 이미 만든 배치는 다시 만들지 않는다(라벨 작업 보호). 다시 만들려면 --force
출력: manifest/label_<batch>.csv, labeling/<batch>/images/
사용: python tools/select_label_frames.py batch02 [--force]
"""
import argparse, csv, shutil, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import TRAIN_PERSONS, VAL_PERSONS, TEST_PERSONS

ALLOC = {
    "batch01": {
        "V03": 5, "V06": 5, "V07": 5, "V05": 4, "V01": 3, "V02": 2, "V04": 2,  # P003
        "V14": 15, "V15": 11,                                                  # P002
        "V12": 15, "V13": 11,                                                  # P001 (검증)
    },
    "batch02": {
        "V17": 3, "V18": 6, "V19": 6,   # P005 (V17은 #450~ 구간, B등급 위주라 B 최대 3장에 맞춤)
        "V20": 15,                      # P006 (검증, #280~ 종단면 구간)
        "V22": 15,                      # P007 (회전 촬영, V21은 제외)
        "V23": 15,                      # P008
        "V25": 8, "V26": 7,             # P010
        "V27": 15,                      # P011
        "V16": 30,                      # P004 (시험)
        "V24": 30,                      # P009 (시험, 3초 클립이라 가능한 만큼)
    },
}

def split_of(p):
    return "test" if p in TEST_PERSONS else "val" if p in VAL_PERSONS else "train" if p in TRAIN_PERSONS else "?"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("batch", choices=list(ALLOC)); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_csv = ROOT / f"manifest/label_{a.batch}.csv"
    if out_csv.exists() and not a.force:
        raise SystemExit(f"{out_csv.name} 이 이미 있습니다. 라벨 작업을 덮어쓰지 않도록 중단합니다(다시 만들려면 --force).")
    rows = list(csv.DictReader(open(ROOT / "manifest/frames.csv", encoding="utf-8")))
    if a.force:  # 같은 배치를 다시 만들 때는 이전 선택을 되돌리고 이미지도 지운다
        for r in rows:
            if r["label_status"] == f"selected_{a.batch}": r["label_status"] = "unlabeled"
        shutil.rmtree(ROOT / "labeling" / a.batch / "images", ignore_errors=True)
    usable = {}
    for r in rows:
        if not r["exclude_reason"] and r["label_status"] == "unlabeled":
            usable.setdefault(r["video_id"], []).append(r)
    picked = {}
    for vid, n in ALLOC[a.batch].items():
        fr = sorted(usable[vid], key=lambda r: float(r["time_s"]))
        # 사용 가능한 프레임 안에서 균등 간격 (제외 구간이 중간에 있어도 이웃 프레임이 뽑히지 않게)
        sel = []
        for i in range(n):
            best = fr[min(len(fr) - 1, int((i + 0.5) * len(fr) / n))]
            if best not in sel: sel.append(best)
        if len(sel) < n: print(f"  {vid}: 후보가 적어 {n}장 중 {len(sel)}장만 선택")
        picked[vid] = sel
    persons = list(dict.fromkeys(picked[v][0]["person_id"] for v in ALLOC[a.batch]))
    queues = {}
    for p in persons:  # 사람 안에서 영상이 번갈아 나오게
        vids = [v for v in ALLOC[a.batch] if picked[v][0]["person_id"] == p]
        q = {v: list(picked[v]) for v in vids}; out = []
        while any(q.values()):
            for v in vids:
                if q[v]: out.append(q[v].pop(0))
        queues[p] = out
    order = []
    while any(queues.values()):
        for p in persons:
            if queues[p]: order.append(queues[p].pop(0))
    img_dir = ROOT / "labeling" / a.batch / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["order", "person_id", "video_id", "frame_idx", "time_s", "split", "label_image", "file"])
        for k, r in enumerate(order, 1):
            name = f"{k:03d}_{r['person_id']}_{r['video_id']}_{Path(r['file']).name}"
            shutil.copy2(ROOT / r["file"], img_dir / name)
            w.writerow([k, r["person_id"], r["video_id"], r["frame_idx"], r["time_s"], split_of(r["person_id"]), name, r["file"]])
    sel = {r["file"] for r in order}
    for r in rows:
        if r["file"] in sel: r["label_status"] = f"selected_{a.batch}"
    with open(ROOT / "manifest/frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("총", len(order), "장", dict(Counter(r["person_id"] for r in order)))
    print("처음 10장:", [r["person_id"] + "/" + r["video_id"] for r in order[:10]])

if __name__ == "__main__":
    main()
