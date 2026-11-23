"""경동맥 내경 자동 측정 — 로컬 웹 앱 서버 (계획서 14절, 2026-11-18).

영상 입력 → 전체 프레임 분석(백그라운드) → 박동별 내경 요약 · 검토 구간 · 경계 겹쳐 보기.
경계 수정 후 재계산 · mm 수동 보정 · 의료진 승인 · 분석 PDF · 폴더 자동 감시(계획서 14절 1~8).
영상은 이 컴퓨터 밖으로 나가지 않는다(127.0.0.1에서만 열림). 분석 작업은 app_data/jobs/ (커밋 제외)에 저장.

실행:
    cd carotid && .venv/bin/python -m app.server [--checkpoint models/carotid_model.pt] [--port 8765]
브라우저: http://127.0.0.1:8765
"""
import argparse
import csv
import json
import queue
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import report
from app.i18n import LANGS
from app.ingest import load_any
from app.watch import FolderWatcher
from src.measure import analyze, measure_masks

JOBS = ROOT / "app_data" / "jobs"
STATIC = Path(__file__).parent / "static"
CHECKPOINT = ROOT / "models" / "carotid_model.pt"
SETTINGS = ROOT / "app_data" / "settings.json"
PORT = 8765

app = FastAPI(title="경동맥 내경 자동 측정")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
jobs: dict[str, dict] = {}
work: queue.Queue = queue.Queue()
edit_lock = threading.Lock()


# ── 작업 상태 ─────────────────────────────────────────────────────────────────

def _save_state(jid):
    j = jobs[jid]
    json.dump({k: v for k, v in j.items() if not k.startswith("_")},
              open(JOBS / jid / "job.json", "w"), ensure_ascii=False)


def _set(jid, **kw):
    jobs[jid].update(kw)
    _save_state(jid)


def _load_jobs():
    JOBS.mkdir(parents=True, exist_ok=True)
    for d in JOBS.iterdir():
        f = d / "job.json"
        if f.exists():
            j = json.load(open(f))
            if j.get("status") not in ("done", "error"):
                j.update(status="error", message="서버가 다시 시작되어 분석이 중단됨")
            jobs[d.name] = j


def _new_job(name: str, source: str) -> str:
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    (JOBS / jid).mkdir(parents=True)
    jobs[jid] = {"id": jid, "name": name, "source": source, "status": "queued", "progress": 0.0,
                 "message": "대기 중", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_state(jid)
    return jid


# ── 분석 ─────────────────────────────────────────────────────────────────────

def _clean(o):
    """NaN·무한대 → null (브라우저 JSON.parse는 NaN을 읽지 못함)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _contour(mask: np.ndarray) -> list:
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(c, 1.0, True)[:, 0, :].tolist()


def _dataset_video(video_id: str):
    v = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "manifest/videos.csv", encoding="utf-8"))}[video_id]
    rows = sorted((r for r in csv.DictReader(open(ROOT / "manifest/frames.csv", encoding="utf-8"))
                   if r["video_id"] == video_id), key=lambda r: int(r["frame_idx"]))
    mm = float(v["mm_per_px"])
    cal = v["format"] != "AVI"
    meta = {"format": v["format"], "device": "E-CUBE 15" if cal else "AVI (ECG 화면 장비)", "mm_per_px": mm,
            "mm_calibrated": cal, "mm_source": "DICOM 헤더" if cal else "참고값: 깊이 눈금 간격 가정, 보정 미확정",
            "notes": [], "person_id": v["person_id"], "video_id": video_id}
    return [ROOT / r["file"] for r in rows], np.array([float(r["time_s"]) for r in rows]), meta


def _run(jid: str):
    j = jobs[jid]
    d = JOBS / jid
    _set(jid, status="running", message="영상 읽는 중", progress=0.02)
    if j["source"].startswith("dataset:"):
        files, times, meta = _dataset_video(j["source"].split(":", 1)[1])
    else:
        vid = load_any(d / j["_upload"] if "_upload" in j else next(d.glob("source.*")))
        fdir = d / "frames"
        fdir.mkdir(exist_ok=True)
        files = []
        for i, g in enumerate(vid.frames):
            p = fdir / f"frame_{i:04d}.png"
            cv2.imwrite(str(p), g)
            files.append(p)
            if i % 50 == 0:
                _set(jid, progress=0.02 + 0.13 * i / len(vid.frames))
        times = vid.times
        meta = {"format": vid.fmt, "device": vid.device, "mm_per_px": vid.mm_per_px, "mm_calibrated": vid.mm_calibrated,
                "mm_source": vid.mm_source, "notes": vid.notes}
        for f in d.glob("source.*"):      # 원본 파일은 프레임 추출 뒤 지운다(환자 정보가 든 헤더를 남기지 않음)
            f.unlink()
    h, w = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE).shape
    meta.update(n_frames=len(files), duration_s=float(times[-1]), width=w, height=h, model=CHECKPOINT.name)
    (d / "files.json").write_text(json.dumps([str(f) for f in files]))
    _set(jid, message="AI 분석 중", progress=0.15)

    last = [0.0]

    def prog(i, n):
        if time.time() - last[0] > 0.5:
            last[0] = time.time()
            _set(jid, progress=0.15 + 0.8 * i / n, message=f"AI 분석 중 ({i}/{n} 프레임)")

    res = analyze(files, times, meta["mm_per_px"], CHECKPOINT, on_progress=prog, return_masks=True)
    masks = res.pop("_masks")
    # 경계 수정·mm 보정 때 모델을 다시 돌리지 않도록 마스크·확신도·시각을 남긴다
    np.savez_compressed(d / "masks.npz", masks=np.stack([m > 0 for m in masks]), confs=np.array(res.pop("confs")),
                        times=np.asarray(times, float))
    _save_result(jid, res, masks, meta)
    _set(jid, status="done", progress=1.0)


def _save_result(jid, res, masks, meta):
    res["contours"] = [_contour(m) for m in masks]
    res["meta"] = meta
    json.dump(_clean(res), open(JOBS / jid / "result.json", "w"), ensure_ascii=False, allow_nan=False)
    s = res["summary"]
    _set(jid, message=s["status"], summary=_clean(s), meta=meta)


def _recompute(jid, meta=None):
    """저장된 마스크로 측정·판정·박동 요약만 다시(모델 추론 없음). 결과가 바뀌면 의료진 승인은 풀린다."""
    d = JOBS / jid
    z = np.load(d / "masks.npz")
    masks = list(z["masks"])
    meta = meta or json.load(open(d / "result.json"))["meta"]
    edited = set(jobs[jid].get("edited", []))
    res = measure_masks(masks, list(z["confs"]), z["times"], meta["mm_per_px"], edited)
    res.pop("confs", None)
    _save_result(jid, res, [m.astype(np.uint8) for m in masks], meta)
    if jobs[jid].pop("approval", None):
        _set(jid, approval_cleared=time.strftime("%Y-%m-%d %H:%M:%S"))


def _worker():
    while True:
        jid = work.get()
        try:
            _run(jid)
        except Exception as e:          # 화면에 이유를 보여 주고 다음 작업으로
            _set(jid, status="error", message=f"분석 실패: {e}")
        finally:
            work.task_done()


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {"model": CHECKPOINT.name, "model_exists": CHECKPOINT.exists()}


@app.get("/api/demos")
def demos():
    out = []
    for v in csv.DictReader(open(ROOT / "manifest/videos.csv", encoding="utf-8")):
        if v["use"] == "Y":
            out.append({"video_id": v["video_id"], "person_id": v["person_id"], "format": v["format"],
                        "frames": int(v["frames"]), "duration_s": float(v["duration_s"]), "note": v["note"]})
    return out


@app.post("/api/demos/{video_id}")
def run_demo(video_id: str):
    if not any(v["video_id"] == video_id for v in demos()):
        raise HTTPException(404, "영상 없음")
    v = next(x for x in demos() if x["video_id"] == video_id)
    jid = _new_job(f"{v['person_id']} · {video_id} ({v['format']})", f"dataset:{video_id}")
    work.put(jid)
    return jobs[jid]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower() or ".bin"
    jid = _new_job(f"업로드 영상 {time.strftime('%H:%M')}", "upload")   # 파일 이름에 실명이 있을 수 있어 쓰지 않음
    with open(JOBS / jid / f"source{suffix}", "wb") as f:
        shutil.copyfileobj(file.file, f)
    work.put(jid)
    return jobs[jid]


@app.get("/api/jobs")
def list_jobs():
    return sorted(jobs.values(), key=lambda j: j["id"], reverse=True)


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    return jobs[jid]


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    if jid not in jobs or jobs[jid]["status"] in ("queued", "running"):
        raise HTTPException(400, "분석 중인 작업은 지울 수 없음")
    shutil.rmtree(JOBS / jid, ignore_errors=True)
    jobs.pop(jid)
    return {"ok": True}


@app.get("/api/jobs/{jid}/result")
def result(jid: str):
    f = JOBS / jid / "result.json"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="application/json")


_file_cache: dict[str, list] = {}


@app.get("/api/jobs/{jid}/frames/{i}.jpg")
def frame(jid: str, i: int):
    if jid not in _file_cache:
        f = JOBS / jid / "files.json"
        if not f.exists():
            raise HTTPException(404)
        _file_cache[jid] = json.loads(f.read_text())
    files = _file_cache[jid]
    if not 0 <= i < len(files):
        raise HTTPException(404)
    img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


# ── 경계 수정 · mm 보정 · 승인 (선택 7·8, 권장 5) ─────────────────────────────

def _done(jid):
    if jid not in jobs or jobs[jid]["status"] != "done" or not (JOBS / jid / "masks.npz").exists():
        raise HTTPException(400, "분석이 끝난 작업만 고칠 수 있습니다(이 기능 이전에 분석한 영상은 다시 분석하세요)")


def _write_mask(jid, i, mask_or_none):
    d = JOBS / jid
    z = dict(np.load(d / "masks.npz"))
    if not (d / "masks_model.npz").exists():            # 처음 고칠 때 모델 결과를 따로 보관(되돌리기용)
        shutil.copy(d / "masks.npz", d / "masks_model.npz")
    if not 0 <= i < len(z["masks"]):
        raise HTTPException(404)
    z["masks"][i] = mask_or_none if mask_or_none is not None else np.load(d / "masks_model.npz")["masks"][i]
    np.savez_compressed(d / "masks.npz", **z)


@app.post("/api/jobs/{jid}/frames/{i}/mask")
def edit_mask(jid: str, i: int, body: dict = Body(...)):
    """사람이 고친 경계(다각형, 원본 해상도 픽셀) 또는 empty=true(혈관 없음)로 이 프레임 마스크를 바꾸고 재계산."""
    _done(jid)
    with edit_lock:
        meta = json.load(open(JOBS / jid / "result.json"))["meta"]
        m = np.zeros((meta["height"], meta["width"]), np.uint8)
        if not body.get("empty"):
            pts = np.array(body.get("polygon") or [], np.float64)
            if len(pts) < 3:
                raise HTTPException(400, "점이 3개 이상 필요합니다")
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        _write_mask(jid, i, m > 0)
        jobs[jid]["edited"] = sorted(set(jobs[jid].get("edited", [])) | {i})
        _recompute(jid)
    return jobs[jid]


@app.post("/api/jobs/{jid}/frames/{i}/reset")
def reset_mask(jid: str, i: int):
    _done(jid)
    with edit_lock:
        if (JOBS / jid / "masks_model.npz").exists():
            _write_mask(jid, i, None)
        jobs[jid]["edited"] = sorted(set(jobs[jid].get("edited", [])) - {i})
        _recompute(jid)
    return jobs[jid]


@app.post("/api/jobs/{jid}/calibration")
def calibrate(jid: str, body: dict = Body(...)):
    """mm 보정값을 사람이 넣는다(보정 정보가 없는 영상). 마스크는 그대로, 측정만 다시."""
    _done(jid)
    v = float(body.get("mm_per_px", 0))
    if not 0.01 <= v <= 1.0:
        raise HTTPException(400, "mm/px 값이 범위(0.01~1.0)를 벗어남")
    with edit_lock:
        meta = json.load(open(JOBS / jid / "result.json"))["meta"]
        meta.update(mm_per_px=v, mm_calibrated=True, mm_source=f"사용자 입력({time.strftime('%Y-%m-%d %H:%M')})")
        _recompute(jid, meta)
    return jobs[jid]


@app.post("/api/jobs/{jid}/approve")
def approve(jid: str, body: dict = Body(...)):
    _done(jid)
    name = (body.get("reviewer") or "").strip()
    if not name:
        raise HTTPException(400, "검토자 이름을 입력하세요")
    _set(jid, approval={"reviewer": name[:60], "comment": (body.get("comment") or "").strip()[:500],
                        "time": time.strftime("%Y-%m-%d %H:%M")})
    jobs[jid].pop("approval_cleared", None)
    _save_state(jid)
    return jobs[jid]


@app.delete("/api/jobs/{jid}/approve")
def unapprove(jid: str):
    if jid in jobs and jobs[jid].pop("approval", None):
        _save_state(jid)
    return jobs.get(jid, {})


# ── 보고서 (필수 2) ───────────────────────────────────────────────────────────

@app.get("/api/jobs/{jid}/report/{name}")
def report_file(jid: str, name: str):
    f = JOBS / jid / "report" / Path(name).name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/jobs/{jid}/report.pdf")
def report_pdf(jid: str, lang: str = "ko"):
    if jid not in jobs or jobs[jid]["status"] != "done":
        raise HTTPException(400, "분석이 끝난 뒤에 만들 수 있습니다")
    lang = lang if lang in LANGS else "ko"
    d = JOBS / jid
    res = json.load(open(d / "result.json"))
    page = report.build(jobs[jid], res, json.loads((d / "files.json").read_text()), d / "report", f"http://127.0.0.1:{PORT}", lang)
    pdf = page.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    if not report.to_pdf(f"http://127.0.0.1:{PORT}/api/jobs/{jid}/report/{page.name}", pdf):
        return FileResponse(page)       # Chrome이 없으면 HTML(브라우저에서 인쇄)
    done = bool(jobs[jid].get("approval"))
    state = ("reviewed" if done else "AI_draft") if lang == "en" else ("검토완료" if done else "AI초안")
    return FileResponse(pdf, media_type="application/pdf", filename=f"carotid_{jid}_{state}.pdf",
                        content_disposition_type="inline")


# ── 폴더 자동 감시 (선택 6) ──────────────────────────────────────────────────

def _on_new_file(path: Path):
    jid = _new_job(f"자동 · {time.strftime('%H:%M:%S')}", "watch")     # 파일 이름(실명 가능)은 쓰지 않음
    shutil.copy(path, JOBS / jid / f"source{path.suffix.lower()}")
    work.put(jid)


watcher = FolderWatcher(_on_new_file)


def _settings():
    return json.load(open(SETTINGS)) if SETTINGS.exists() else {}


@app.get("/api/watch")
def watch_status():
    return {"path": str(watcher.path) if watcher.path else None, "saved": _settings().get("watch_dir")}


@app.post("/api/watch")
def watch_start(body: dict = Body(...)):
    try:
        watcher.start(Path(body.get("path", "")))
    except ValueError as e:
        raise HTTPException(400, str(e))
    SETTINGS.write_text(json.dumps({**_settings(), "watch_dir": str(watcher.path)}, ensure_ascii=False))
    return watch_status()


@app.delete("/api/watch")
def watch_stop():
    watcher.stop()
    SETTINGS.write_text(json.dumps({**_settings(), "watch_dir": None}, ensure_ascii=False))
    return watch_status()


def main():
    global CHECKPOINT, PORT
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT.relative_to(ROOT)))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    CHECKPOINT = ROOT / a.checkpoint
    PORT = a.port
    if not CHECKPOINT.exists():
        sys.exit(f"체크포인트 없음: {CHECKPOINT}")
    _load_jobs()
    threading.Thread(target=_worker, daemon=True).start()
    if _settings().get("watch_dir"):
        try:
            watcher.start(Path(_settings()["watch_dir"]))
            print(f"폴더 자동 감시: {watcher.path}")
        except ValueError as e:
            print(f"폴더 자동 감시를 다시 켜지 못함: {e}")
    print(f"모델: {CHECKPOINT}\n브라우저에서 http://127.0.0.1:{a.port} 를 여세요")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
