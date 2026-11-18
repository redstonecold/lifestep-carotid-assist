"""영상 파일(DICOM · AVI · WMV) → 초음파 영역만 자른 흑백 프레임 + 프레임 시각 + mm/px (계획서 13절).

- DICOM: 초음파 영역(SequenceOfUltrasoundRegions)만 잘라 화면 위쪽 기관명·이름을 제거, mm/px = PhysicalDeltaX(cm)×10,
         프레임 시각 = FrameTimeVector 누적값(간격이 고르지 않음). 환자 정보 태그는 읽지 않는다
- AVI(P003 장비 640×448): (96,36,548,384) 영역, ECG 띠 제외, mm/px 0.078125 = 참고값(보정 미확정)
- WMV(E-CUBE 15 화면 녹화 880×660): (73,64,795,569) 영역, mm/px 0.069444(같은 장비 DICOM 헤더값)
- 그 밖의 크기: 화면 전체를 쓰고 mm 보정 없음 → 결과는 `mm 산출 불가`, 모델 입력 배율만 0.1mm/px로 가정
- AVI·WMV 프레임 시각은 디코더 타임스탬프(ms)를 쓴다
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PRESETS = {  # (가로, 세로) → (이름, 자르기 영역, mm/px, 보정 근거)
    (640, 448): ("AVI (ECG 화면 장비)", (96, 36, 548, 384), 0.078125, "참고값: 깊이 눈금 간격 가정, 보정 미확정"),
    (880, 660): ("E-CUBE 15 화면 녹화(WMV)", (73, 64, 795, 569), 0.069444, "같은 장비 DICOM 헤더값"),
}
FALLBACK_MM = 0.1


@dataclass
class Video:
    frames: list            # 흑백 uint8 배열
    times: np.ndarray       # 초
    mm_per_px: float
    mm_calibrated: bool
    mm_source: str
    device: str
    fmt: str
    notes: list = field(default_factory=list)


def _is_dicom(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(132)
    return head[128:132] == b"DICM" or path.suffix.lower() in (".dcm", ".dicom")


def load_dicom(path: Path) -> Video:
    import pydicom
    from pydicom.pixels import iter_pixels
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    n = int(getattr(ds, "NumberOfFrames", 1))
    notes = []
    try:
        r = ds.SequenceOfUltrasoundRegions[0]
        box = (int(r.RegionLocationMinX0), int(r.RegionLocationMinY0), int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1))
        mm, cal, src = float(r.PhysicalDeltaX) * 10, True, "DICOM 헤더(PhysicalDeltaX)"
    except (AttributeError, IndexError):
        box, mm, cal, src = None, FALLBACK_MM, False, "보정 정보 없음"
        notes.append("DICOM에 초음파 영역 정보가 없어 화면 전체를 사용")
    if hasattr(ds, "FrameTimeVector"):
        t = np.cumsum(np.array([float(x) for x in ds.FrameTimeVector][:n]) / 1000.0)
    else:
        ft = float(getattr(ds, "FrameTime", 0) or 0) / 1000.0 or 1 / float(getattr(ds, "CineRate", 30) or 30)
        t = np.arange(n) * ft
        notes.append("FrameTimeVector 없음: 고정 간격으로 가정")
    frames = []
    for arr in iter_pixels(path):
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if box:
            x0, y0, x1, y1 = box
            arr = arr[y0:y1, x0:x1]
        frames.append(np.ascontiguousarray(arr.astype(np.uint8)))
    return Video(frames, t[:len(frames)], mm, cal, src, str(getattr(ds, "ManufacturerModelName", "DICOM")), "DICOM", notes)


def load_video(path: Path) -> Video:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다")
    frames, times, i = [], [], 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    preset = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if preset is None:
            preset = PRESETS.get((fr.shape[1], fr.shape[0]), False)
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if preset:
            x0, y0, x1, y1 = preset[1]
            g = g[y0:y1, x0:x1]
        frames.append(np.ascontiguousarray(g))
        times.append(ms / 1000.0 if ms > 0 or i == 0 else i / fps)
        i += 1
    cap.release()
    if not frames:
        raise ValueError("프레임이 없습니다")
    t = np.array(times) - times[0]
    if len(t) > 1 and not np.all(np.diff(t) > 0):     # 타임스탬프가 이상하면 fps로
        t = np.arange(len(frames)) / fps
    fmt = path.suffix.lstrip(".").upper() or "VIDEO"
    if preset:
        name, _, mm, src = preset
        return Video(frames, t, mm, "참고값" not in src, src, name, fmt)
    return Video(frames, t, FALLBACK_MM, False, "보정 정보 없음", "알 수 없는 장비", fmt,
                 ["알려진 장비 화면 크기가 아니어서 화면 전체를 사용"])


def load_any(path: Path) -> Video:
    return load_dicom(path) if _is_dicom(path) else load_video(path)
