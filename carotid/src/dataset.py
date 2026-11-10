"""
데이터셋 — (이미지, 마스크) 쌍을 frames.csv 매니페스트에서 불러온다.

학습(train) 전처리 순서 (계획서 11절 `증강 파라미터`):
  1. PNG 흑백 로드 (크롭은 extract_frames.py에서 이미 완료)
  2. 0.1mm/px로 재표본화 (장비별 픽셀 간격 통일)
  3. 기하 증강 (반전·회전·크기·종횡비, 마스크는 최근접 보간)
  4. 256×256 크롭 (2/3 무작위, 1/3 내강 중심. F세트는 항상 무작위)
  5. 밝기 증강 (감마·깊이별 밝기·전역 밝기대비) → 화질 증강 (잡음·스펙클·흐림·JPEG)
  6. 이미지별 밝기 정규화 (mean=0, std=1)

검증(valid) 전처리: 재표본화 → 정규화 → 32의 배수 패딩. 증강 없음.
  지표는 원래 해상도에서 계산하므로 원본 크기 마스크도 함께 돌려준다.

사용 조건 (계획서 7절):
  - label_status == 'labeled'
  - grade in A·B·C·F  (X는 제외)
  - F세트는 마스크가 비어 있을 수 있다(검출 불가 프레임)
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

TARGET_MM_PER_PX = 0.1
CROP_SIZE = 256
PAD_MULTIPLE = 32

# 사람 단위 분할 (계획서 6절, 2026-11-10: 학습 9 · 검증 1 · 시험 1)
TRAIN_PERSONS = {"P001", "P002", "P003", "P005", "P007", "P008", "P009", "P010", "P011"}
VAL_PERSONS   = {"P006"}
TEST_PERSONS  = {"P004"}

GRADES = ("A", "B", "C", "F")
# 세트별 픽셀 손실 가중치 (계획서 11절 `손실 함수 처리`). B 0.7은 판단값이라 1차 탐색 7단계에서 1.0과 비교
GRADE_WEIGHT = {"A": 1.0, "B": 0.7, "C": 1.0, "F": 1.0}

# 증강 강도 (계획서 11절 표). 확률은 강도와 관계없이 같다.
AUG_PRESETS = {
    "weak": dict(rot=5, scale=0.10, aspect=0.0, gamma=(0.8, 1.25), depth=0.0, bc=0.10,
                 noise=0.0, speckle=0.0, blur=None, jpeg=None),
    "medium": dict(rot=15, scale=0.15, aspect=0.05, gamma=(0.7, 1.5), depth=0.2, bc=0.20,
                   noise=0.05, speckle=0.10, blur=(0.5, 1.0), jpeg=(50, 95)),
    "strong": dict(rot=25, scale=0.25, aspect=0.10, gamma=(0.6, 1.7), depth=0.3, bc=0.30,
                   noise=0.10, speckle=0.20, blur=(0.5, 1.5), jpeg=(30, 95)),
}
AUG_PROB = dict(flip=0.5, rot=0.5, scale=0.5, aspect=0.3, gamma=0.3, depth=0.3, bc=0.5,
                noise=0.15, speckle=0.2, blur=0.2, jpeg=0.3, clahe=0.3)
LUMEN_CROP_PROB = 1 / 3


# ── 저수준 이미지 처리 ────────────────────────────────────────────────────────

def _load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _rescale(arr: np.ndarray, src_mm: float, interp) -> np.ndarray:
    scale = src_mm / TARGET_MM_PER_PX
    if abs(scale - 1.0) < 0.01:
        return arr
    h, w = arr.shape[:2]
    pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    pil = pil.resize((round(w * scale), round(h * scale)), interp)
    return np.array(pil, dtype=np.float32) / 255.0


def _normalize(img: np.ndarray) -> np.ndarray:
    std = img.std()
    return (img - img.mean()) / (std if std > 1e-6 else 1.0)


def _pad_to(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    ph = max(0, size - h)
    pw = max(0, size - w)
    return np.pad(arr, ((0, ph), (0, pw)))


def _crop(img: np.ndarray, mask: np.ndarray, size: int, lumen_center: bool):
    """size×size 크롭. lumen_center면 내강 픽셀 하나를 중심으로(nnU-Net 전경 과표집과 같은 방식)."""
    img  = _pad_to(img,  size)
    mask = _pad_to(mask, size)
    h, w = img.shape
    if lumen_center and mask.any():
        ys, xs = np.nonzero(mask)
        k = random.randrange(len(ys))
        y = int(np.clip(ys[k] - size // 2, 0, h - size))
        x = int(np.clip(xs[k] - size // 2, 0, w - size))
    else:
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
    return img[y:y+size, x:x+size], mask[y:y+size, x:x+size]


def _geometric(img: np.ndarray, mask: np.ndarray, p: dict):
    """좌우 반전 + 회전·크기·종횡비를 한 번의 아핀 변환으로. 상하 반전·탄성 변형은 쓰지 않는다."""
    if random.random() < AUG_PROB["flip"]:
        img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
    angle = random.uniform(-p["rot"], p["rot"]) if random.random() < AUG_PROB["rot"] else 0.0
    s = 1 + random.uniform(-p["scale"], p["scale"]) if random.random() < AUG_PROB["scale"] else 1.0
    a = 1 + random.uniform(-p["aspect"], p["aspect"]) if p["aspect"] and random.random() < AUG_PROB["aspect"] else 1.0
    if angle == 0.0 and s == 1.0 and a == 1.0:
        return img, mask
    h, w = img.shape
    c = np.array([w / 2, h / 2])
    lin = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2] @ np.diag([s * a, s / a])  # 회전 × 가로·세로 배율
    m = np.hstack([lin, (c - lin @ c)[:, None]])                                        # 영상 중심 고정
    img  = cv2.warpAffine(img,  m, (w, h), flags=cv2.INTER_LINEAR,  borderValue=0)
    mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, mask


def _intensity(img: np.ndarray, p: dict, clahe: bool) -> np.ndarray:
    """밝기 증강. 전역 밝기·대비는 정규화로 대부분 상쇄되므로 감마·깊이별 밝기를 함께 쓴다."""
    if random.random() < AUG_PROB["gamma"]:
        img = np.power(np.clip(img, 1e-6, 1.0), random.uniform(*p["gamma"]))
    if p["depth"] and random.random() < AUG_PROB["depth"]:
        bottom = 1 + random.uniform(-p["depth"], p["depth"])   # 위는 ×1, 아래로 갈수록 ×bottom (TGC 모사)
        img = img * np.linspace(1.0, bottom, img.shape[0], dtype=np.float32)[:, None]
    if random.random() < AUG_PROB["bc"]:
        img = img * (1 + random.uniform(-p["bc"], p["bc"])) + random.uniform(-p["bc"], p["bc"])
    img = np.clip(img, 0, 1)
    if clahe and random.random() < AUG_PROB["clahe"]:
        u8 = (img * 255).astype(np.uint8)
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255
    return img


def _quality(img: np.ndarray, p: dict) -> np.ndarray:
    """화질 증강: 가우시안 잡음 · 스펙클(곱셈) 잡음 · 흐림 · JPEG 압축."""
    if p["noise"] and random.random() < AUG_PROB["noise"]:
        img = img + np.random.normal(0, random.uniform(0, p["noise"]), img.shape).astype(np.float32)
    if p["speckle"] and random.random() < AUG_PROB["speckle"]:
        img = img * (1 + np.random.normal(0, random.uniform(0, p["speckle"]), img.shape).astype(np.float32))
    img = np.clip(img, 0, 1)
    if p["blur"] and random.random() < AUG_PROB["blur"]:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(*p["blur"]))
    if p["jpeg"] and random.random() < AUG_PROB["jpeg"]:
        q = random.randint(*p["jpeg"])
        _, buf = cv2.imencode(".jpg", (img * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return img


# ── 데이터셋 ──────────────────────────────────────────────────────────────────

class CarotidDataset(Dataset):
    """
    Args:
        root           carotid/ 폴더 경로
        split          'train' | 'val' | 'test'
        augment        증강 강도 'weak' | 'medium' | 'strong' | None (train에서만 적용)
        persons        사용할 person_id 집합 (기본: split에 맞는 분할 상수). 교차검증에서 바꾼다
        grade_weight   세트별 손실 가중치 (기본 GRADE_WEIGHT)
        crop_size      학습 크롭 크기 (256 또는 384)
        clahe          CLAHE 증강 사용 여부
        fraction       학습 곡선용. 사람별로 라벨의 이 비율만 사용 (사람별 비율 유지)
        seed           fraction 표본 추출 시드
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: str | None = "medium",
        persons: set | None = None,
        grade_weight: dict | None = None,
        crop_size: int = CROP_SIZE,
        clahe: bool = False,
        fraction: float = 1.0,
        seed: int = 0,
    ):
        self.root    = Path(root)
        self.split   = split
        self.aug     = AUG_PRESETS[augment] if (augment and split == "train") else None
        self.persons = persons or {"train": TRAIN_PERSONS, "val": VAL_PERSONS, "test": TEST_PERSONS}[split]
        self.grade_weight = grade_weight or GRADE_WEIGHT
        self.crop_size = crop_size
        self.clahe   = clahe

        self._mm = {
            r["video_id"]: float(r["mm_per_px"])
            for r in _read_csv(self.root / "manifest" / "videos.csv")
        }
        self.samples = self._build_samples()
        if fraction < 1.0:
            self.samples = _subsample_per_person(self.samples, fraction, seed)

    def _build_samples(self):
        samples = []
        for r in _read_csv(self.root / "manifest" / "frames.csv"):
            if r["person_id"] not in self.persons:
                continue
            if r["label_status"] != "labeled" or r["grade"] not in GRADES:
                continue
            img_path  = self.root / r["file"]
            mask_path = (self.root / "data" / "labels" / r["video_id"]
                         / (img_path.stem + "_mask.png"))
            if not img_path.exists() or not mask_path.exists():
                continue
            samples.append({
                "person_id": r["person_id"],
                "video_id":  r["video_id"],
                "frame_idx": int(r["frame_idx"]),
                "file":      r["file"],
                "grade":     r["grade"],
                "img_path":  img_path,
                "mask_path": mask_path,
                "mm_per_px": self._mm.get(r["video_id"], TARGET_MM_PER_PX),
            })
        return samples

    def data_version(self) -> str:
        """라벨 구성이 바뀌었는지 확인하는 짧은 지문 (중단 재개 시 비교)."""
        import hashlib
        key = "|".join(sorted(f"{s['file']}:{s['grade']}" for s in self.samples))
        return hashlib.sha1(key.encode()).hexdigest()[:10]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        img0 = _load_gray(s["img_path"])
        mask0 = (_load_gray(s["mask_path"]) > 0.5).astype(np.float32)

        mm = s["mm_per_px"]
        img  = _rescale(img0,  mm, Image.BILINEAR)
        mask = _rescale(mask0, mm, Image.NEAREST)

        # 검증·시험: 추론(predict.py)과 같이 전체 프레임 → 32의 배수 패딩.
        # 지표는 원래 해상도에서 재므로 원본 마스크와 메타정보를 함께 돌려준다.
        if self.split != "train":
            img = _normalize(img)
            img, h, w = pad_for_inference(img)
            mask, _, _ = pad_for_inference(mask)
            meta = {k: s[k] for k in ("person_id", "video_id", "frame_idx", "file", "grade", "mm_per_px")}
            meta.update(h=h, w=w, weight=self.grade_weight[s["grade"]])
            return (
                torch.from_numpy(img).unsqueeze(0).float(),
                torch.from_numpy(mask).unsqueeze(0).float(),
                mask0,
                meta,
            )

        if self.aug:
            img, mask = _geometric(img, mask, self.aug)
        lumen_center = s["grade"] != "F" and random.random() < LUMEN_CROP_PROB
        img, mask = _crop(img, mask, self.crop_size, lumen_center)
        if self.aug:
            img = _intensity(img, self.aug, self.clahe)
            img = _quality(img, self.aug)
        img = _normalize(img)

        return (
            torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float(),   # (1, H, W)
            torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float(),  # (1, H, W)
            torch.tensor(self.grade_weight[s["grade"]], dtype=torch.float32),   # 세트 가중치
        )


def eval_collate(batch):
    """검증은 영상마다 크기가 달라 1장씩 평가한다. 리스트 포장을 벗긴다."""
    assert len(batch) == 1
    img, mask, mask0, meta = batch[0]
    return img.unsqueeze(0), mask.unsqueeze(0), mask0, meta


# ── 균형 샘플러 ───────────────────────────────────────────────────────────────

class BalancedPersonSampler(Sampler):
    """
    epoch마다 사람을 먼저 무작위로 고른 뒤 그 사람의 프레임을 하나 선택한다.
    P003이 프레임 대부분을 차지할 때 편중을 막는다.

    epoch당 샘플 수 = len(persons) × samples_per_person
    """

    def __init__(self, dataset: CarotidDataset, samples_per_person: int = 50):
        self._by_person: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            self._by_person[s["person_id"]].append(i)
        self._persons = list(self._by_person)
        self._spp     = samples_per_person

    def __len__(self):
        return len(self._persons) * self._spp

    def __iter__(self):
        order = []
        for _ in range(self._spp):
            for p in random.sample(self._persons, len(self._persons)):
                order.append(random.choice(self._by_person[p]))
        return iter(order)


# ── 유틸 ────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _subsample_per_person(samples: list[dict], fraction: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_p: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_p[s["person_id"]].append(s)
    out = []
    for p in sorted(by_p):
        group = sorted(by_p[p], key=lambda s: s["file"])
        out += rng.sample(group, max(1, round(len(group) * fraction)))
    return out


def pad_for_inference(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """추론용: 32의 배수로 패딩. 원래 h, w도 반환."""
    h, w = img.shape
    ph = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pw = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    return np.pad(img, ((0, ph), (0, pw))), h, w
