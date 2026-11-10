"""종단면 사용 구간을 정하기 위한 시간별 장면 모음 (계획서 10절 batch02 4단계).

영상마다 일정 간격으로 프레임을 뽑아 시각(초)을 붙여 한 장에 모은다.
선생님이 보고 "몇 초부터 종단면"을 정하면 tools/add_dicom_videos.py 의 SEGMENTS 에 적는다.
출력: reports/segments/<영상ID>.png
사용: python tools/segment_sheet.py V17 V20 V21 [--step 0.5]
"""
import argparse, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
TW, TH, COLS = 320, 220, 6

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("vids", nargs="+"); ap.add_argument("--step", type=float, default=0.5)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "manifest/frames.csv", encoding="utf-8")))
    out = ROOT / "reports/segments"; out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 18)
    for vid in a.vids:
        fr = [r for r in rows if r["video_id"] == vid]
        picks, nxt = [], 0.0
        for r in fr:
            if float(r["time_s"]) >= nxt:
                picks.append(r); nxt += a.step
        n_rows = -(-len(picks) // COLS)
        sheet = Image.new("L", (COLS * TW, n_rows * TH), 0); d = ImageDraw.Draw(sheet)
        for k, r in enumerate(picks):
            im = Image.open(ROOT / r["file"]).convert("L").resize((TW - 4, TH - 4))
            x, y = (k % COLS) * TW, (k // COLS) * TH
            sheet.paste(im, (x + 2, y + 2))
            # 프레임 번호는 DICOM 뷰어 기준(1부터), 괄호는 시각
            d.rectangle([x + 2, y + 2, x + 150, y + 26], fill=0)
            d.text((x + 6, y + 4), f"#{int(r['frame_idx']) + 1} ({float(r['time_s']):.1f}초)", fill=255, font=font)
        sheet.save(out / f"{vid}.png"); print("저장:", out / f"{vid}.png", f"{len(picks)}장")

if __name__ == "__main__":
    main()
