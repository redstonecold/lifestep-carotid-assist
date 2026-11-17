"""초기 모델 초안 → Label Studio 프로젝트 (계획서 9절 3단계: 초안 수정 방식).

- manifest/label_<batch>.csv 의 이미지를 초기 모델로 추론하고, 마스크 외곽선을 다각형 초안(prediction)으로 올린다
- 시험(split=test, P004)은 초안 없이 이미지만 올린다(모델 영향 배제, 직접 그림)
- 세트(A/B/C/F/X)는 초안에 넣지 않는다. 라벨할 때 직접 고른다
- 초안 마스크는 labeling/<batch>/drafts/ 에도 저장한다(나중에 초안 대비 수정량 측정용)
- 프로젝트 carotid_<batch> 가 없으면 만들고, 이미 올린 이미지·초안은 건너뛴다(다시 실행해도 됨)

Label Studio 토큰: 화면 오른쪽 위 계정 → Account & Settings → Personal Access Token(또는 Legacy Token)을 복사해
private/ls_token 파일에 저장(커밋 제외 폴더)하거나 LS_TOKEN 환경 변수에 넣는다
사용:
    python tools/ls_prelabel.py batch03 --checkpoint runs/default/best.pt
변환은 기존과 같다: python tools/ls_export_to_masks.py <export.json> --batch batch03
"""
import argparse, csv, os, sys
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import predict_files

LABEL = "경동맥 내강"
MIN_AREA_RATIO = 0.05   # 가장 큰 영역의 5% 미만 조각은 버림(점 같은 오검출)
MAX_POLYS = 2
EPS_PX = 1.5            # 외곽선 단순화(px). 점이 적어야 고치기 쉽다
UPLOAD_CHUNK = 50


class LS:
    def __init__(self, url, token):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        if token.count(".") == 2:   # Personal Access Token(JWT) → access 토큰으로 교환
            r = requests.post(f"{self.url}/api/token/refresh", json={"refresh": token}); r.raise_for_status()
            self.s.headers["Authorization"] = f"Bearer {r.json()['access']}"
        else:
            self.s.headers["Authorization"] = f"Token {token}"

    def call(self, method, path, **kw):
        r = self.s.request(method, f"{self.url}{path}", **kw)
        if not r.ok:
            raise SystemExit(f"Label Studio {method} {path} 실패 {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    def project(self, title, label_config):
        res = self.call("GET", "/api/projects", params={"title": title, "page_size": 100})
        for p in res.get("results", res) if isinstance(res, dict) else res:
            if p["title"] == title:
                return p["id"]
        p = self.call("POST", "/api/projects", json={"title": title, "label_config": label_config,
                                                     "show_collab_predictions": True})
        print(f"프로젝트 생성: {title} (id={p['id']})")
        return p["id"]

    def tasks(self, pid):
        out, page = [], 1
        while True:
            r = self.s.get(f"{self.url}/api/tasks", params={"project": pid, "page": page, "page_size": 500})
            if r.status_code == 404:
                break
            r.raise_for_status()
            j = r.json(); batch = j["tasks"] if isinstance(j, dict) else j
            out += batch
            if not batch or len(out) >= (j.get("total", len(out)) if isinstance(j, dict) else len(out)):
                break
            page += 1
        return out


def task_name(t):
    """업로드 시 붙는 'abcd1234-' 접두어를 떼고 원래 이미지 이름을 돌려준다."""
    n = t["data"]["image"].split("/")[-1]
    return n.split("-", 1)[1] if "-" in n else n


def mask_to_result(mask):
    H, W = mask.shape
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)
    if not cs:
        return []
    big = cv2.contourArea(cs[0])
    out = []
    for c in cs[:MAX_POLYS]:
        if cv2.contourArea(c) < big * MIN_AREA_RATIO:
            continue
        pts = cv2.approxPolyDP(c, EPS_PX, True)[:, 0, :]
        if len(pts) < 3:
            continue
        out.append({"from_name": "lumen", "to_name": "image", "type": "polygonlabels",
                    "original_width": W, "original_height": H, "image_rotation": 0,
                    "value": {"points": [[float(x) * 100 / W, float(y) * 100 / H] for x, y in pts],
                              "polygonlabels": [LABEL], "closed": True}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch"); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--url", default=os.environ.get("LS_URL", "http://localhost:8080"))
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    tf = ROOT / "private/ls_token"
    token = os.environ.get("LS_TOKEN") or (tf.read_text().strip() if tf.exists() else "")
    if not token:
        sys.exit("Label Studio 토큰을 private/ls_token 에 저장하거나 LS_TOKEN 환경 변수에 넣으세요(파일 위쪽 설명).")
    ckpt = ROOT / a.checkpoint
    rows = list(csv.DictReader(open(ROOT / f"manifest/label_{a.batch}.csv", encoding="utf-8")))
    mm = {r["video_id"]: float(r["mm_per_px"]) for r in csv.DictReader(open(ROOT / "manifest/videos.csv", encoding="utf-8"))}
    img_dir = ROOT / "labeling" / a.batch / "images"

    ls = LS(a.url, token)
    pid = ls.project(f"carotid_{a.batch}", (ROOT / "labeling/label_config.xml").read_text(encoding="utf-8"))
    have = {task_name(t) for t in ls.tasks(pid)}
    todo = [r for r in rows if r["label_image"] not in have]
    for i in range(0, len(todo), UPLOAD_CHUNK):
        chunk = todo[i:i + UPLOAD_CHUNK]
        # 필드 이름이 같으면 Label Studio가 마지막 파일만 받는다 → 파일마다 다른 이름
        files = [(r["label_image"], (r["label_image"], open(img_dir / r["label_image"], "rb"), "image/png")) for r in chunk]
        ls.call("POST", f"/api/projects/{pid}/import", files=files)
        print(f"  이미지 업로드 {i + len(chunk)}/{len(todo)}")

    tasks = {task_name(t): t for t in ls.tasks(pid)}
    need = [r for r in rows if r["split"] != "test" and r["label_image"] in tasks
            and not tasks[r["label_image"]].get("total_predictions")]
    if need:
        model_version = f"init_{ckpt.parent.name}"
        masks = predict_files([img_dir / r["label_image"] for r in need], ckpt,
                              mm_per_px=[mm[r["video_id"]] for r in need], threshold=a.threshold,
                              out_dir=ROOT / "labeling" / a.batch / "drafts")
        empty = 0
        for r, m in zip(need, masks):
            res = mask_to_result(m)
            empty += not res
            ls.call("POST", "/api/predictions", json={"task": tasks[r["label_image"]]["id"], "result": res,
                                                      "model_version": model_version})
        ls.call("PATCH", f"/api/projects/{pid}", json={"model_version": model_version, "show_collab_predictions": True})
        print(f"초안 {len(need)}장 올림 (빈 초안 {empty}장 = 모델이 혈관을 못 찾음 → F 후보이거나 직접 그림)")
    n_test = sum(r["split"] == "test" for r in rows)
    print(f"완료: 프로젝트 carotid_{a.batch} 작업 {len(tasks)}장, 시험 {n_test}장은 초안 없음(직접 그림)")
    print(f"열기: {a.url}/projects/{pid}/data")


if __name__ == "__main__":
    main()
