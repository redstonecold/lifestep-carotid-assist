"""폴더 자동 감시.

운영체제 파일 이벤트(macOS FSEvents, watchdog)로 새 영상 파일만 받는다. 폴더 전체를 주기적으로 다시 훑지 않는다.
파일 크기가 STABLE_S 동안 변하지 않으면(장비·복사 프로그램이 저장을 끝냈다고 보고) 한 번만 분석 대기열에 넣는다.
원본 파일은 건드리지 않는다(복사해서 분석).
"""
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

EXTS = {".dcm", ".dicom", ".avi", ".wmv", ".mp4", ".mov"}
STABLE_S = 2.0


class _Handler(FileSystemEventHandler):
    def __init__(self, on_file):
        self.on_file, self.seen, self.lock = on_file, set(), threading.Lock()

    def on_created(self, e):
        if not e.is_directory:
            self._consider(Path(e.src_path))

    def on_moved(self, e):
        if not e.is_directory:
            self._consider(Path(e.dest_path))

    def _consider(self, p: Path):
        if p.suffix.lower() not in EXTS or p.name.startswith("."):
            return
        with self.lock:
            if p in self.seen:
                return
            self.seen.add(p)
        threading.Thread(target=self._wait_stable, args=(p,), daemon=True).start()

    def _wait_stable(self, p: Path):
        last, since = -1, time.time()
        while time.time() - since < 600:            # 10분 안에 저장이 끝나지 않으면 포기
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                return
            if size == last and size > 0:
                if time.time() - stable_from >= STABLE_S:
                    self.on_file(p)
                    return
            else:
                last, stable_from = size, time.time()
            time.sleep(0.5)


class FolderWatcher:
    def __init__(self, on_file):
        self.on_file, self.obs, self.path = on_file, None, None

    def start(self, path: Path):
        self.stop()
        path = Path(path).expanduser()
        if not path.is_dir():
            raise ValueError(f"폴더가 없습니다: {path}")
        self.obs = Observer()
        self.obs.schedule(_Handler(self.on_file), str(path), recursive=False)
        self.obs.start()
        self.path = path

    def stop(self):
        if self.obs:
            self.obs.stop()
            self.obs.join(timeout=3)
        self.obs, self.path = None, None
