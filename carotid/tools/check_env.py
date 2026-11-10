"""모델 다운로드 + Mac 학습 환경 점검.

1. U-Net + ResNet34 ImageNet 사전학습 가중치를 받아 models/pretrained/ 에 저장
2. Apple GPU(MPS) 사용 가능 여부 확인
3. 계획서 고정값(배치 8, 입력 256) 과 입력 384 로 학습 1스텝 시간·메모리 측정
4. 1회 학습 시간(계획서 예상 15~20분) 추정
결과는 reports/env_check.txt 에 저장된다.

사용: source .venv/bin/activate && python tools/check_env.py [--offline]
"""
import argparse, platform, sys, time
from pathlib import Path
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/pretrained/unet_resnet34_imagenet_init.pth"
lines = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

def build(pretrained=True):
    if WEIGHTS.exists():
        m = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1)
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        log("가중치: 저장된 파일 사용", WEIGHTS.name)
        return m
    m = smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)
    if pretrained:
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), WEIGHTS)
        log("가중치: 다운로드 후 저장", WEIGHTS, f"({WEIGHTS.stat().st_size/1e6:.1f} MB)")
    return m

def sync(dev):
    if dev.type == "mps": torch.mps.synchronize()

def bench(m, dev, size, batch, steps=5):
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.randn(batch, 1, size, size, device=dev); y = (torch.rand(batch, 1, size, size, device=dev) > 0.8).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ts = []
    for i in range(steps + 1):
        sync(dev); t = time.time()
        opt.zero_grad(); loss = loss_fn(m(x), y); loss.backward(); opt.step()
        sync(dev); ts.append(time.time() - t)
    step = sum(ts[1:]) / steps  # 첫 스텝(준비 시간) 제외
    mem = torch.mps.driver_allocated_memory() / 1e9 if dev.type == "mps" else float("nan")
    return step, mem

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true", help="다운로드 없이 무작위 가중치로 점검")
    a = ap.parse_args()
    log("=== 환경 ===")
    log("OS:", platform.platform(), "| CPU:", platform.machine(), "| Python:", sys.version.split()[0])
    log("torch:", torch.__version__, "| smp:", smp.__version__)
    mps = torch.backends.mps.is_available()
    log("Apple GPU(MPS):", "사용 가능" if mps else "사용 불가 → CPU로 진행")
    dev = torch.device("mps" if mps else "cpu")

    log("\n=== 모델 ===")
    m = build(pretrained=not a.offline).to(dev)
    log(f"U-Net + ResNet34, 파라미터 {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    log("\n=== 학습 속도·메모리 (가짜 데이터) ===")
    results = {}
    for size, batch in [(256, 8), (384, 8), (256, 16)]:
        try:
            step, mem = bench(m, dev, size, batch)
            results[(size, batch)] = step
            log(f"입력 {size}, 배치 {batch}: 1스텝 {step:.2f}초" + (f", GPU 메모리 {mem:.1f}GB" if mem == mem else ""))
        except RuntimeError as e:
            log(f"입력 {size}, 배치 {batch}: 실패 ({str(e).splitlines()[0][:80]})")
        if dev.type == "mps": torch.mps.empty_cache()

    log("\n=== 1회 학습 시간 추정 (계획서 예상: 15~20분) ===")
    if (256, 8) in results:
        for n_labels in (100, 150, 300):
            steps_per_epoch = -(-n_labels // 8)
            for epochs in (40, 80):
                mins = results[(256, 8)] * steps_per_epoch * epochs / 60
                log(f"라벨 {n_labels}장, {epochs} epoch: 약 {mins:.0f}분 (검증 시간 제외)")
        log("※ 조기 종료로 실제 epoch 수는 보통 상한(80)보다 적다.")

    log("\n=== 추론 속도 ===")
    m.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 288, 352, device=dev); m(x); sync(dev); t = time.time()
        for _ in range(20): m(x)
        sync(dev); per = (time.time() - t) / 20
    log(f"프레임 1장 {per*1000:.0f}ms → AVI 1개(약 2,200프레임) 약 {per*2200/60:.1f}분")

    out = ROOT / "reports/env_check.txt"; out.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n결과 저장: {out}")

if __name__ == "__main__":
    main()
