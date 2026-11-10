"""발표용 비식별 대표 이미지 생성 → reports/slides/

- DICOM·WMV: 초음파 영역 (73,64)-(807,568)만 잘라 기관명·이름·날짜 제거, 흑백
- AVI: 화면에 개인정보가 없어 ECG가 보이도록 전체 화면 사용
- 원본 파일명(실명 포함)은 출력하지 않고 사람 ID만 쓴다
- 시험 전용 대상자(P004·P009)는 발표 이미지에 쓰지 않는다
사용: python tools/make_slide_images.py
"""
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydicom.pixels import pixel_array
import pydicom

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "data/raw"
OUT = ROOT / "reports/slides"
REGION = (73, 64, 807, 568)  # x0, y0, x1, y1
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

def dicom(person, idx, frac):
    """사람 폴더 안 DICOM을 정렬 순서 idx로 선택, 전체 길이의 frac 위치 프레임."""
    p = sorted(x for x in (RAW / person).rglob("*.dcm") if "(1)" not in x.name)[idx]
    n = int(pydicom.dcmread(p, stop_before_pixels=True).NumberOfFrames)
    x0, y0, x1, y1 = REGION
    return Image.fromarray(pixel_array(p, index=int(frac * (n - 1)))[y0:y1, x0:x1]).convert("L")

def avi(vid, frame):
    """원본 경로는 private/id_map.csv에서 video_id로 찾는다(공개 저장소에 파일명을 두지 않음)."""
    rel = next(r["source_relpath"] for r in csv.DictReader(open(ROOT / "private/id_map.csv", encoding="utf-8"))
               if r["video_id"] == vid)
    c = cv2.VideoCapture(str(ROOT.parent / rel)); c.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, f = c.read(); assert ok
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))

def wmv_frame(person, vid, frame):
    return Image.open(ROOT / f"data/frames/{person}/{vid}/frame_{frame:04d}.png").convert("L")

# (파일명, 제목, 설명, 이미지 생성)
ITEMS = [
    ("01_avi_ecg",        "AVI · P003",            "장시간(약 75초) · 화면 하단 ECG",   lambda: avi("V06", 300)),
    ("02_wmv_short",      "WMV · P001",            "3초 클립 · 동맥·정맥 동시",         lambda: wmv_frame("P001", "V13", 60)),
    ("03_dicom_short",    "DICOM 3초 · P010",      "곧은 종단면 · 벽 선명",            lambda: dicom("P010", 0, 0.5)),
    ("04_dicom_long",     "DICOM 15초 · P008",     "긴 영상 · 비스듬한 종단면",         lambda: dicom("P008", 0, 0.5)),
    ("05_transverse",     "횡단면 구간 · P007",    "혈관이 동그랗게 보임 → 라벨 제외", lambda: dicom("P007", 0, 0.0)),
    ("06_longitudinal",   "종단면 구간 · P007",    "같은 영상 후반 → 라벨 사용",       lambda: dicom("P007", 0, 0.85)),
    ("07_rotated",        "회전 촬영 · P007",      "탐촉자가 돌아간 상태 · 어려운 사례", lambda: dicom("P007", 1, 0.5)),
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for fname, title, desc, make in ITEMS:
        im = make(); im.save(OUT / f"{fname}.png"); tiles.append((im, title, desc))
        print("저장:", OUT / f"{fname}.png", im.size)

    # 한 장 요약 (4열)
    W, H, CAP = 480, 340, 56
    cols = 4; rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * W, rows * (H + CAP)), "white")
    d = ImageDraw.Draw(sheet)
    ft, fd = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 17)
    for i, (im, title, desc) in enumerate(tiles):
        r, c = divmod(i, cols)
        t = im.copy(); t.thumbnail((W - 12, H - 6))
        x = c * W + (W - t.width) // 2; y = r * (H + CAP) + 4
        sheet.paste(t.convert("RGB"), (x, y))
        d.text((c * W + 10, y + H - 2), title, fill="black", font=ft)
        d.text((c * W + 10, y + H + 26), desc, fill=(90, 90, 90), font=fd)
    sheet.save(OUT / "00_overview.png"); print("저장:", OUT / "00_overview.png", sheet.size)

if __name__ == "__main__":
    sys.exit(main())
