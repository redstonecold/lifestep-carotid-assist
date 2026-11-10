"""DICOM 기술 정보 점검(개인정보 태그는 출력하지 않음).

파일명에 실명이 있을 수 있으므로 경로 대신 private/id_map.csv 의 영상 ID를 쓴다.
등록되지 않은 파일(중복·WMV와 같은 내용)은 "<사람 폴더>/미등록#순번"으로 표시한다.
사용: python tools/inventory_dicom.py ../data/raw > reports/dicom_inventory.json
"""
import sys, json, csv
from pathlib import Path
import pydicom

ROOT = Path(__file__).resolve().parents[1]
ID_BY_PATH = {(ROOT.parent / r["source_relpath"]).resolve(): r["video_id"]
              for r in csv.DictReader(open(ROOT / "private/id_map.csv", encoding="utf-8"))}

SAFE = ["Modality","Manufacturer","ManufacturerModelName","SOPClassUID","Rows","Columns",
        "NumberOfFrames","CineRate","FrameTime","RecommendedDisplayFrameRate",
        "PhotometricInterpretation","SamplesPerPixel","BitsAllocated","StudyDate","SeriesDescription",
        "TransferSyntaxUID"]

def region_info(ds):
    out = []
    for r in getattr(ds, "SequenceOfUltrasoundRegions", []) or []:
        out.append({k: (str(getattr(r, k)) if hasattr(r, k) else None) for k in
            ["RegionSpatialFormat","RegionDataType","RegionLocationMinX0","RegionLocationMinY0",
             "RegionLocationMaxX1","RegionLocationMaxY1","PhysicalUnitsXDirection",
             "PhysicalUnitsYDirection","PhysicalDeltaX","PhysicalDeltaY"]})
    return out

res, unreg = [], {}
for p in sorted(Path(sys.argv[1]).rglob("*.dcm")):
    ds = pydicom.dcmread(p, stop_before_pixels=True)
    person = p.relative_to(sys.argv[1]).parts[0]
    vid = ID_BY_PATH.get(p.resolve())
    if vid is None:
        unreg[person] = unreg.get(person, 0) + 1
        vid = f"미등록#{unreg[person]}"
    d = {"person_id": person, "video_id": vid, "size_MB": round(p.stat().st_size/1e6,1)}
    for k in SAFE:
        v = getattr(ds, k, None)
        d[k] = str(v) if v is not None else None
    if ds.file_meta: d["TransferSyntaxUID"] = str(ds.file_meta.TransferSyntaxUID)
    d["regions"] = region_info(ds)
    d["PixelSpacing"] = str(getattr(ds, "PixelSpacing", None))
    res.append(d)
print(json.dumps(res, ensure_ascii=False, indent=1))
