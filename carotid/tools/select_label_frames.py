"""라벨용 프레임 선정 (계획서 9·10절).

- batch01: P001~P003 사람당 26장(A 20 + B 최대 5 목표, C 발생 여유 포함), 총 78장
- batch02: 학습·검증 6명 사람당 15장, 시험 P004·P009 사람당 30장 (2026-11-03 분할 기준)
- batch03: 2026-11-10 목표(학습 415 · 검증 40 · 시험 40)까지 추가 267장. ALLOC_F는 F세트 후보
  (기존 제외 구간: 횡단면·탐촉자 이동·V21 전체)에서 뽑는다. 세트는 라벨할 때 정한다
- repeat01: 반복 라벨 20장. 이미 A·C로 라벨된 프레임에서 사람별로 고루 뽑는다(표 6 기준선).
  frames.csv는 바꾸지 않고, 변환은 ls_export_to_masks.py --repeat
- 이미 뽑힌 프레임과 0.1초 이내인 프레임은 고르지 않는다(거의 같은 장면 방지)
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
    "batch03": {  # 계획서 9절 `목표` 표의 추가 장수 (F 후보는 ALLOC_F)
        # 3초 클립(P001·P002·P011)은 기존 라벨과 0.1초 이상 떨어진 후보가 이것뿐이다.
        # 모자란 28장은 긴 영상(P003 +10, P005 +8, P008 +10)으로 옮겼다(2026-11-10 실측).
        "V01": 6, "V02": 6, "V04": 6, "V03": 8, "V05": 7, "V06": 9, "V07": 7,  # P003 49 (V01·V02·V04는 탐촉자 이동·정맥 겹침 → F 후보 포함)
        "V17": 5, "V18": 14, "V19": 14,  # P005 33 (+F 15 = 48)
        "V23": 50,                       # P008 50
        "V22": 20,                       # P007 20 (+F 15 = 35)
        "V12": 2, "V13": 4,              # P001 6
        "V14": 5, "V15": 9,              # P002 14
        "V25": 13, "V26": 12,            # P010 25
        "V27": 5,                        # P011 5
        "V20": 19,                       # P006 검증 19 (+F 6 = 25)
        "V16": 10,                       # P004 시험 10 (직접 그림)
    },
}
ALLOC_F = {  # 기존 제외 구간에서 뽑는 F세트 후보 (계획서 9절 `F세트 출처`)
    "batch03": {"V17": 15, "V21": 10, "V22": 5, "V20": 6},
}
F_REASONS = {"종단면 구간 밖", "영상 제외", "탐촉자 조정 구간"}
MIN_GAP_S = 0.1
REPEAT_N = 20

def split_of(p):
    return "test" if p in TEST_PERSONS else "val" if p in VAL_PERSONS else "train" if p in TRAIN_PERSONS else "?"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("batch", choices=[*ALLOC, "repeat01"]); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.batch.startswith("repeat"):
        return select_repeat(a.batch, a.force)
    out_csv = ROOT / f"manifest/label_{a.batch}.csv"
    if out_csv.exists() and not a.force:
        raise SystemExit(f"{out_csv.name} 이 이미 있습니다. 라벨 작업을 덮어쓰지 않도록 중단합니다(다시 만들려면 --force).")
    rows = list(csv.DictReader(open(ROOT / "manifest/frames.csv", encoding="utf-8")))
    if a.force:  # 같은 배치를 다시 만들 때는 이전 선택을 되돌리고 이미지도 지운다
        for r in rows:
            if r["label_status"] == f"selected_{a.batch}": r["label_status"] = "unlabeled"
        shutil.rmtree(ROOT / "labeling" / a.batch / "images", ignore_errors=True)
    taken = {}   # 이미 뽑힌 프레임 시각 (영상별)
    for r in rows:
        if r["label_status"] != "unlabeled":
            taken.setdefault(r["video_id"], []).append(float(r["time_s"]))
    far = lambda r: all(abs(float(r["time_s"]) - t) > MIN_GAP_S for t in taken.get(r["video_id"], []))
    usable, f_pool = {}, {}
    for r in rows:
        if r["label_status"] == "unlabeled" and far(r):
            if not r["exclude_reason"]:
                usable.setdefault(r["video_id"], []).append(r)
            elif r["exclude_reason"] in F_REASONS:
                f_pool.setdefault(r["video_id"], []).append(r)
    jobs = [(v, n, usable) for v, n in ALLOC[a.batch].items()] + \
           [(v, n, f_pool) for v, n in ALLOC_F.get(a.batch, {}).items()]
    picked = {}
    for vid, n, pool in jobs:
        fr = sorted(pool.get(vid, []), key=lambda r: float(r["time_s"]))
        if not fr:
            print(f"  {vid}: 후보 없음"); continue
        # 사용 가능한 프레임 안에서 균등 간격 (제외 구간이 중간에 있어도 이웃 프레임이 뽑히지 않게)
        sel = []
        for i in range(n):
            best = fr[min(len(fr) - 1, int((i + 0.5) * len(fr) / n))]
            if best not in sel: sel.append(best)
        if len(sel) < n: print(f"  {vid}: 후보가 적어 {n}장 중 {len(sel)}장만 선택")
        picked[vid] = picked.get(vid, []) + sel
    persons = list(dict.fromkeys(picked[v][0]["person_id"] for v in picked))
    queues = {}
    for p in persons:  # 사람 안에서 영상이 번갈아 나오게
        vids = [v for v in picked if picked[v][0]["person_id"] == p]
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

def select_repeat(name, force):
    """반복 라벨 20장: A·C로 라벨된 프레임에서 사람별로 번갈아 고른다. frames.csv는 바꾸지 않는다."""
    import random
    out_csv = ROOT / f"manifest/label_{name}.csv"
    if out_csv.exists() and not force:
        raise SystemExit(f"{out_csv.name} 이 이미 있습니다(다시 만들려면 --force).")
    rows = [r for r in csv.DictReader(open(ROOT / "manifest/frames.csv", encoding="utf-8"))
            if r["label_status"] == "labeled" and r["grade"] in ("A", "C")]
    if len(rows) < REPEAT_N:
        raise SystemExit(f"A·C로 라벨된 프레임이 {len(rows)}장뿐입니다. batch01·02 라벨을 먼저 끝내세요.")
    rng = random.Random(20261110)
    by_p = {}
    for r in sorted(rows, key=lambda r: r["file"]):
        by_p.setdefault(r["person_id"], []).append(r)
    for v in by_p.values():
        rng.shuffle(v)
    order = []
    while len(order) < REPEAT_N:
        for p in sorted(by_p):
            if by_p[p] and len(order) < REPEAT_N:
                order.append(by_p[p].pop())
    img_dir = ROOT / "labeling" / name / "images"
    shutil.rmtree(img_dir, ignore_errors=True); img_dir.mkdir(parents=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["order", "person_id", "video_id", "frame_idx", "time_s", "split", "label_image", "file"])
        for k, r in enumerate(order, 1):
            nm = f"{k:03d}_{r['person_id']}_{r['video_id']}_{Path(r['file']).name}"
            shutil.copy2(ROOT / r["file"], img_dir / nm)
            w.writerow([k, r["person_id"], r["video_id"], r["frame_idx"], r["time_s"], split_of(r["person_id"]), nm, r["file"]])
    print("반복 라벨", len(order), "장", dict(Counter(r["person_id"] for r in order)))

if __name__ == "__main__":
    main()
