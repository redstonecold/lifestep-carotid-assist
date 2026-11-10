"""새 대상자 DICOM(P004~P011) 등록 + 전체 프레임 비식별 추출 (계획서 5·10절 batch02).

- 원본 파일명에 실명이 있으므로 이 코드에는 경로를 쓰지 않는다. 사람 폴더 안 파일을 정렬해 순서대로
  영상 ID(V16~)를 붙이고, 원본 경로는 private/id_map.csv 에만 기록한다
- 중복 파일(해시 동일, 다운로드 사본 "(1)")은 등록하지 않는다
- 프레임: 초음파 영역 (73,64)-(807,568)만 잘라 흑백 저장 → data/frames/<사람>/<영상ID>/frame_NNNN.png
- 프레임 시각: DICOM FrameTimeVector 누적값(간격이 불균일하므로 고정 fps를 쓰지 않음)
- mm/px: 초음파 영역 PhysicalDeltaX(cm) × 10. 파일마다 읽고 0.0694와 다르면 중단
- 사용 구간: SEGMENTS 에 구간이 있는 영상은 구간 밖 프레임을 exclude_reason="종단면 구간 밖"으로 표시
- 제외 영상: EXCLUDED 의 영상은 videos.csv use=N, 모든 프레임 exclude_reason="영상 제외"
- 기존 videos.csv·frames.csv·id_map.csv 행은 유지하고, 새 영상 행만 추가·갱신. 재실행 안전
사용: python tools/add_dicom_videos.py [--dry-run]
"""
import argparse, csv, hashlib
from pathlib import Path
import numpy as np
import pydicom
from pydicom.pixels import iter_pixels
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "data/raw"
PERSONS = ["P004", "P005", "P006", "P007", "P008", "P009", "P010", "P011"]
FIRST_ID = 16
REGION = (73, 64, 807, 568)          # x0, y0, x1, y1 (모든 새 파일 헤더 값 동일)
MM_EXPECTED = 0.069444
# 영상 ID → 사용 구간 [(첫 프레임, 끝 프레임), ...]. DICOM 뷰어와 같은 1부터 센 번호, 양 끝 포함.
# 구간 밖은 횡단면·탐촉자 이동이라 라벨 제외. 여기 없는 영상은 전체 사용 (2026-11-03 사용자 확인)
SEGMENTS: dict[str, list[tuple[int, int]]] = {
    "V17": [(450, 562)],              # 앞쪽 횡단면. 종단면이 오른쪽 위에 일부만 보여 A등급은 어려움
    "V20": [(280, 559)],              # 앞쪽 횡단면
    "V22": [(1, 113), (190, 386)],    # #114~#189 탐촉자 이동
}
EXCLUDED = {"V21"}                    # 종단면 구간이 짧고 불확실해 영상 전체 제외 (사용자 판단)
NOTES = {  # 영상 ID → 품질 메모 (계획서 5절 새 영상 목록)
    "V16": "보통, 내강 얇음", "V17": "구간 선별(#450~): B등급 위주", "V18": "보통~좋음", "V19": "보통~좋음",
    "V20": "구간 선별(#280~)", "V21": "제외: 종단면 불확실", "V22": "보통~나쁨: 회전 촬영, #114~189 제외",
    "V23": "보통: 비스듬한 종단면", "V24": "좋음", "V25": "좋음", "V26": "좋음", "V27": "좋음",
}

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()

def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []

def write_csv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def discover():
    """사람별 DICOM을 정렬해 중복을 빼고 영상 ID를 붙인다. 반환: [(vid, person, path)]"""
    out, vid, seen = [], FIRST_ID, set()
    for p in PERSONS:
        # 다운로드 사본 "(1)"이 원본보다 뒤에 오도록 정렬
        files = sorted((RAW / p).glob("*.dcm"), key=lambda x: ("(1)" in x.name, x.name))
        for f in files:
            h = md5(f)
            if h in seen:
                print(f"  {p}: 중복 파일 1개 제외"); continue
            seen.add(h); out.append((f"V{vid:02d}", p, f)); vid += 1
    return out

def times_of(ds):
    n = int(ds.NumberOfFrames)
    v = np.array([float(x) for x in ds.FrameTimeVector][:n]) / 1000.0   # 첫 값 0, 이후 직전 프레임과의 간격(ms)
    return np.cumsum(v)

def mm_of(ds):
    r = ds.SequenceOfUltrasoundRegions[0]
    box = (int(r.RegionLocationMinX0), int(r.RegionLocationMinY0), int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1))
    return float(r.PhysicalDeltaX) * 10, box

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    vids = discover()
    videos = [r for r in read_csv(ROOT / "manifest/videos.csv") if int(r["video_id"][1:]) < FIRST_ID]
    frames_old = read_csv(ROOT / "manifest/frames.csv")
    keep = [r for r in frames_old if int(r["video_id"][1:]) < FIRST_ID]
    old = {r["file"]: r for r in frames_old}
    idmap = [r for r in read_csv(ROOT / "private/id_map.csv") if int(r["video_id"][1:]) < FIRST_ID]
    new_frames = []
    for vid, p, path in vids:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        mm, box = mm_of(ds)
        if abs(mm - MM_EXPECTED) > 1e-4 or box != (73, 64, 807, 568):
            raise SystemExit(f"{vid}: 보정값·영역이 예상과 다름 mm={mm} box={box}")
        ts = times_of(ds); n = len(ts); segs = SEGMENTS.get(vid)
        out_dir = ROOT / "data/frames" / p / vid
        print(f"{vid} {p}: {n}프레임, {ts[-1]:.2f}초, 평균 {(n-1)/ts[-1]:.1f}fps, 사용 구간 {segs or '전체'}")
        if not a.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            todo = [i for i in range(n) if not (out_dir / f"frame_{i:04d}.png").exists()]
            if todo:
                x0, y0, x1, y1 = REGION
                for i, arr in enumerate(iter_pixels(path)):   # RGB (YBR 압축본도 RGB로 변환되어 나옴)
                    if i in todo:
                        Image.fromarray(arr[y0:y1, x0:x1]).convert("L").save(out_dir / f"frame_{i:04d}.png")
        videos.append({"video_id": vid, "person_id": p, "format": "DICOM", "frames": n,
                       "fps": round((n - 1) / ts[-1], 3), "duration_s": round(float(ts[-1]), 2),
                       "crop_x0": REGION[0], "crop_y0": REGION[1], "crop_x1": REGION[2], "crop_y1": REGION[3],
                       "mm_per_px": round(mm, 6), "calib_source": "dicom_header", "ecg_on_screen": "N",
                       "use": "N" if vid in EXCLUDED else "Y", "note": NOTES.get(vid, "")})
        idmap.append({"video_id": vid, "person_id": p, "source_relpath": str(path.relative_to(ROOT.parent))})
        for i, t in enumerate(ts):
            rel = f"data/frames/{p}/{vid}/frame_{i:04d}.png"; o = old.get(rel, {})
            reason = o.get("exclude_reason", "")
            if reason in ("종단면 구간 밖", "영상 제외"): reason = ""   # 설정이 바뀌면 다시 계산
            if not reason and vid in EXCLUDED: reason = "영상 제외"
            if not reason and segs and not any(s <= i + 1 <= e for s, e in segs): reason = "종단면 구간 밖"
            new_frames.append({"person_id": p, "video_id": vid, "frame_idx": i, "time_s": round(float(t), 3),
                               "file": rel, "grade": o.get("grade", ""),
                               "label_status": o.get("label_status", "unlabeled"),
                               "exclude_reason": reason, "note": o.get("note", "")})
    if a.dry_run:
        print("dry-run: 파일을 쓰지 않음"); return
    write_csv(ROOT / "manifest/videos.csv", videos)
    write_csv(ROOT / "manifest/frames.csv", keep + new_frames)
    write_csv(ROOT / "private/id_map.csv", idmap)
    from collections import Counter
    c = Counter(r["person_id"] for r in new_frames if not r["exclude_reason"])
    print("새 영상", len(vids), "| 새 프레임", len(new_frames), "| 사람별 사용 가능", dict(sorted(c.items())))

if __name__ == "__main__":
    main()
