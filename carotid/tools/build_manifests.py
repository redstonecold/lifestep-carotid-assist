"""manifest/videos.csv, manifest/frames.csv 생성 (원본 경로는 private/id_map.csv에서 읽음).

- 영상 ID는 익명(V01~V07: P003 AVI, V12~V15: P001·P002 WMV). 파일명의 이니셜을 학습 데이터에 남기지 않음
- 프레임 시간: AVI는 고정 28.775fps, WMV는 원본 패킷 시간(ffprobe)
- 기존 frames.csv의 grade/label_status/exclude_reason/note 값은 file 기준으로 유지
- V13 0.8초 이전은 탐촉자 조정 구간으로 제외 표시
사용: python tools/build_manifests.py   (ffprobe 필요)
"""
import csv, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "data/raw"
# 원본 경로는 공개 저장소에 두지 않고 private/id_map.csv(video_id → source_relpath)에서 읽는다
SRC = {r["video_id"]: r["source_relpath"].removeprefix("data/raw/")
       for r in csv.DictReader(open(ROOT/"private/id_map.csv", encoding="utf-8"))}
VIDEOS = [  # video_id, person, format, (미사용), fps, crop, mm/px, 보정 출처, ECG, 사용 시작(초), 메모
    ("V01","P003","AVI","",28.775,(96,36,548,384),0.078125,"manual_tick_unconfirmed","Y",0,"구간 선별"),
    ("V02","P003","AVI","",28.775,(96,36,548,384),0.078125,"manual_tick_unconfirmed","Y",0,"나쁨: 정맥 겹침·탐촉자 이동"),
    ("V03","P003","AVI","",28.775,(96,36,548,384),0.078125,"manual_tick_unconfirmed","Y",0,"좋음"),
    ("V04","P003","AVI","",28.775,(96,36,548,384),0.078125,"manual_tick_unconfirmed","Y",0,"나쁨: 큰 정맥"),
    ("V05","P003","AVI","",28.775,(96,36,548,384),0.078125,"manual_tick_unconfirmed","Y",0,"좋음~보통: 어두움"),
    ("V06","P003","AVI","",28.775,(96,36,548,384),0.078125,"manual_tick_unconfirmed","Y",0,"좋음"),
    ("V07","P003","AVI","",28.775,(96,36,548,384),0.078125,"manual_tick_unconfirmed","Y",0,"좋음"),
    ("V12","P001","WMV","",0,(73,64,795,569),0.069444,"dicom_borrowed","N",0,"검증"),
    ("V13","P001","WMV","",0,(73,64,795,569),0.069444,"dicom_borrowed","N",0.8,"검증, 0.8초 이전 제외"),
    ("V14","P002","WMV","",0,(73,64,795,569),0.069444,"dicom_borrowed","N",0,"V14 DICOM 사본과 동일 내용"),
    ("V15","P002","WMV","",0,(73,64,795,569),0.069444,"dicom_borrowed","N",0,""),
]

def wmv_times(path):
    out = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","packet=pts_time",
                          "-of","csv=p=0",str(path)], capture_output=True, text=True, check=True).stdout
    return [float(x) for x in out.split()]

old = {}
fp = ROOT/"manifest/frames.csv"
if fp.exists():
    for r in csv.DictReader(open(fp, encoding="utf-8")): old[r["file"]] = r

vrows, frows = [], []
for vid, p, fmt, _, fps, box, mm, calib, ecg, start, note in VIDEOS:
    rel = SRC[vid]
    d = ROOT/"data/frames"/p/vid
    files = sorted(d.glob("frame_*.png"))
    if fmt == "WMV":
        ts = wmv_times(RAW/rel)
        if len(ts) != len(files):
            raise SystemExit(f"{vid}: 프레임 수 불일치 {len(files)} vs {len(ts)}")
        dur = ts[-1]; fps_out = round(len(ts)/max(dur,1e-6), 3)
    else:
        ts = [i/fps for i in range(len(files))]; dur = ts[-1]; fps_out = fps
    vrows.append({"video_id":vid,"person_id":p,"format":fmt,"frames":len(files),"fps":fps_out,"duration_s":round(dur,2),
                  "crop_x0":box[0],"crop_y0":box[1],"crop_x1":box[2],"crop_y1":box[3],"mm_per_px":mm,
                  "calib_source":calib,"ecg_on_screen":ecg,"use":"Y","note":note})
    for i,(f,t) in enumerate(zip(files, ts)):
        rel_f = f"data/frames/{p}/{vid}/{f.name}"
        o = old.get(rel_f, {})
        reason = o.get("exclude_reason","") or ("탐촉자 조정 구간" if t < start else "")
        frows.append({"person_id":p,"video_id":vid,"frame_idx":i,"time_s":round(t,3),"file":rel_f,
                      "grade":o.get("grade",""),"label_status":o.get("label_status","unlabeled"),
                      "exclude_reason":reason,"note":o.get("note","")})

def write(path, rows):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
write(ROOT/"manifest/videos.csv", vrows)
write(ROOT/"manifest/frames.csv", frows)
from collections import Counter
c = Counter((r["person_id"]) for r in frows if not r["exclude_reason"])
print("videos", len(vrows), "frames", len(frows), "usable by person", dict(c))
