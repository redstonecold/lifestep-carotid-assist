"""
U-Net + ResNet34 모델 생성 및 체크포인트 유틸리티.

사용:
    from src.model import build_model, get_device, save_checkpoint, load_checkpoint
"""

from pathlib import Path
import torch
import segmentation_models_pytorch as smp

PRETRAINED_PATH = Path(__file__).parents[1] / "models" / "pretrained" / "unet_resnet34_imagenet_init.pth"


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """
    Args:
        pretrained  True이면 PRETRAINED_PATH의 ImageNet 가중치로 인코더 초기화.
                    파일이 없으면 경고만 출력하고 무작위 초기화로 진행.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # 수동 로드
        in_channels=1,
        classes=1,
        activation=None,        # sigmoid는 손실 함수 안에서 처리
    )

    if pretrained:
        if PRETRAINED_PATH.exists():
            state = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            loaded = len(state) - len(unexpected)
            print(f"[model] pretrained 로드: {loaded}/{len(state)} keys"
                  f"  (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"[model] WARNING: pretrained 파일 없음 — 무작위 초기화로 진행\n"
                  f"        경로: {PRETRAINED_PATH}")

    return model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, path: Path, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(model: torch.nn.Module, path: Path,
                    device: torch.device | None = None) -> dict:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})
