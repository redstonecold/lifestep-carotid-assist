"""
데이터셋 — (이미지, 마스크) 쌍을 frames.csv 매니페스트에서 불러온다.

전처리 순서:
  1. PNG 흑백 로드 (크롭은 extract_frames.py에서 이미 완료)
  2. 0.1mm/px로 재표본화 (장비별 픽셀 간격 통일)
  3. 증강 (학습 시)
  4. 256×256 무작위 크롭
  5. 이미지별 밝기 정규화 (mean=0, std=1)

사용 조건:
  - label_status == 'labeled'
  - grade in ('A', 'B')  — val은 A만
  - exclude_reason 비어 있음
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

TARGET_MM_PER_PX = 0.1
CROP_SIZE = 256
PAD_MULTIPLE = 32

# 사람 단위 분할 (계획서 6절, 2026-11-03 추첨). 시험 전용 P004·P009는 여기 넣지 않는다.
TRAIN_PERSONS = {"P002", "P003", "P005", "P007", "P008", "P010", "P011"}
VAL_PERSONS   = {"P001", "P006"}
TEST_PERSONS  = {"P004", "P009"}


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


def _random_crop(img: np.ndarray, mask: np.ndarray, size: int):
    img  = _pad_to(img,  size)
    mask = _pad_to(mask, size)
    h, w = img.shape
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img[y:y+size, x:x+size], mask[y:y+size, x:x+size]


def _augment(img: np.ndarray, mask: np.ndarray):
    # 좌우 반전
    if random.random() < 0.5:
        img  = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    # 밝기·대비 ±10%
    img = np.clip(img * (1 + random.uniform(-0.1, 0.1)) + random.uniform(-0.1, 0.1), 0, 1)
    # 감마 0.8~1.2
    img = np.power(np.clip(img, 1e-6, 1.0), random.uniform(0.8, 1.2))
    return img, mask


# ── 데이터셋 ──────────────────────────────────────────────────────────────────

class CarotidDataset(Dataset):
    """
    Args:
        root           carotid/ 폴더 경로
        split          'train' | 'val'
        augment        True이면 증강 적용 (split='train'일 때만 실제 동작)
        train_persons  학습 대상 person_id 집합 (기본: TRAIN_PERSONS)
        val_persons    검증 대상 person_id 집합 (기본: VAL_PERSONS)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = True,
        train_persons: set | None = None,
        val_persons:   set | None = None,
    ):
        self.root    = Path(root)
        self.split   = split
        self.augment = augment and (split == "train")
        self.train_persons = train_persons or TRAIN_PERSONS
        self.val_persons   = val_persons   or VAL_PERSONS

        self._mm = {
            r["video_id"]: float(r["mm_per_px"])
            for r in _read_csv(self.root / "manifest" / "videos.csv")
        }
        self.samples = self._build_samples()

    def _build_samples(self):
        persons = self.train_persons if self.split == "train" else self.val_persons
        samples = []
        for r in _read_csv(self.root / "manifest" / "frames.csv"):
            if r["person_id"] not in persons:
                continue
            if r["label_status"] != "labeled":
                continue
            if r["grade"] not in ("A", "B"):
                continue
            if self.split == "val" and r["grade"] != "A":
                continue
            if r["exclude_reason"]:
                continue
            img_path  = self.root / r["file"]
            mask_path = (self.root / "data" / "labels" / r["video_id"]
                         / (img_path.stem + "_mask.png"))
            if not img_path.exists() or not mask_path.exists():
                continue
            samples.append({
                "person_id": r["person_id"],
                "video_id":  r["video_id"],
                "img_path":  img_path,
                "mask_path": mask_path,
                "mm_per_px": self._mm.get(r["video_id"], TARGET_MM_PER_PX),
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        img  = _load_gray(s["img_path"])
        mask = (_load_gray(s["mask_path"]) > 0.5).astype(np.float32)

        mm = s["mm_per_px"]
        img  = _rescale(img,  mm, Image.BILINEAR)
        mask = _rescale(mask, mm, Image.NEAREST)

        # 검증: 추론(predict.py)과 같이 전체 프레임 → 32의 배수 패딩, 원래 크기 (h, w)도 반환
        if self.split != "train":
            img = _normalize(img)
            img, h, w = pad_for_inference(img)
            mask, _, _ = pad_for_inference(mask)
            return (
                torch.from_numpy(img).unsqueeze(0).float(),
                torch.from_numpy(mask).unsqueeze(0).float(),
                (h, w),
            )

        if self.augment:
            img, mask = _augment(img, mask)

        img, mask = _random_crop(img, mask, CROP_SIZE)
        img = _normalize(img)

        return (
            torch.from_numpy(img).unsqueeze(0).float(),    # (1, H, W)
            torch.from_numpy(mask).unsqueeze(0).float(),   # (1, H, W)
        )


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


def pad_for_inference(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """추론용: 32의 배수로 패딩. 원래 h, w도 반환."""
    h, w = img.shape
    ph = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pw = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    return np.pad(img, ((0, ph), (0, pw))), h, w
