"""비식별 크롭 프레임 추출 (계획 2주차: 프레임 추출과 사람 단위 데이터 구조).

사용: python extract_frames.py <Data 폴더> <carotid 폴더> [--interval 0.5]
- manifest/videos.csv 의 use=Y 영상만 처리
- 원본 경로는 private/id_map.csv 에서만 읽고, 출력에는 video_id만 남김
- DICOM은 초음파 영역(Sequence of Ultrasound Regions)만 잘라 화면 상단의 기관명·이니셜을 제거
- 이미 추출된 프레임은 건너뜀(재실행 안전)
"""
import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image

def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def iter_avi(path, idxs):
    import cv2
    cap = cv2.VideoCapture(str(path)); want = set(idxs); i = 0
    while want:
        ok, fr = cap.read()
        if not ok: break
        if i in want:
            want.discard(i); yield i, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()

def iter_dicom(path, idxs):
    import pydicom
    arr = pydicom.dcmread(path).pixel_array  # (N,H,W,3) RGB
    for i in idxs:
        if i < len(arr):
            yield i, (arr[i].astype(np.float32) @ [0.299, 0.587, 0.114]).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("root")
    ap.add_argument("--interval", type=float, default=0.5)
    a = ap.parse_args()
    root, data = Path(a.root), Path(a.data)
    src = {r["video_id"]: r["source_relpath"] for r in read_csv(root/"private/id_map.csv")}
    out_rows = []
    for v in read_csv(root/"manifest/videos.csv"):
        if v["use"] != "Y": continue
        vid, n, fps = v["video_id"], int(v["frames"]), float(v["fps"])
        step = max(1, round(a.interval * fps))
        idxs = list(range(0, n, step))
        box = tuple(int(v[k]) for k in ("crop_x0","crop_y0","crop_x1","crop_y1"))
        od = root/"data/frames"/vid; od.mkdir(parents=True, exist_ok=True)
        todo = [i for i in idxs if not (od/f"{vid}_f{i:05d}.png").exists()]
        it = iter_avi if v["format"] == "AVI" else iter_dicom
        if todo:
            for i, g in it(data/src[vid], todo):
                Image.fromarray(g).crop(box).save(od/f"{vid}_f{i:05d}.png")
        for i in idxs:
            out_rows.append({"person_id": v["person_id"], "video_id": vid, "frame_idx": i,
                             "time_s": round(i/fps, 3), "file": f"data/frames/{vid}/{vid}_f{i:05d}.png",
                             "grade": "", "label_status": "unlabeled", "exclude_reason": ""})
        print(vid, len(idxs), "frames (new:", len(todo), ")")
    with open(root/"manifest/frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)

if __name__ == "__main__":
    main()
