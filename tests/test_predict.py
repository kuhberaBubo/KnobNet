"""
export된 모델로 추론 테스트
- test_model.py 실행 후 생성된 knobnet_smoke.pt를 로드
- 가짜 오디오 샘플로 predict 확인
python tests/test_predict.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from utils.config import MERT_SR, CLIP_SAMPLES, KNOB_PARAMS
from model.model import KnobNet

EXPORT_PATH = Path(__file__).parent.parent / "data" / "models" / "tmp" / "knobnet_smoke.pt"


def separator(title):
    print(f"\n{'-'*50}\n  {title}\n{'-'*50}")


def main():
    # ── 1. 모델 로드 ─────────────────────────────────────
    separator("1. KnobNet.from_exported")
    if not EXPORT_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일 없음: {EXPORT_PATH}\n"
            "먼저 python tests/test_model.py 를 실행하세요."
        )

    device = torch.device("cpu")
    model = KnobNet.from_exported(EXPORT_PATH, device=device)
    model.summary()

    # ── 2. 가짜 오디오로 추론 ────────────────────────────
    separator("2. 단일 샘플 추론")
    rng = np.random.default_rng(0)
    input_audio = torch.from_numpy(rng.standard_normal(CLIP_SAMPLES).astype(np.float32) * 0.1)
    ref_audio   = torch.from_numpy(rng.standard_normal(CLIP_SAMPLES).astype(np.float32) * 0.1)

    # 배치 차원 추가
    input_audio = input_audio.unsqueeze(0).to(device)
    ref_audio   = ref_audio.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_audio, ref_audio)

    print(f"input shape  : {tuple(input_audio.shape)}")
    print(f"preds shape  : {tuple(preds.shape)}")
    for name, val in zip(KNOB_PARAMS, preds[0].tolist()):
        print(f"  {name:8s} : {val:.4f}")

    # ── 3. 배치 추론 ─────────────────────────────────────
    separator("3. 배치(4) 추론")
    B = 4
    inp_batch = torch.from_numpy(rng.standard_normal((B, CLIP_SAMPLES)).astype(np.float32) * 0.1).to(device)
    ref_batch = torch.from_numpy(rng.standard_normal((B, CLIP_SAMPLES)).astype(np.float32) * 0.1).to(device)

    with torch.no_grad():
        preds_batch = model(inp_batch, ref_batch)

    print(f"batch preds shape : {tuple(preds_batch.shape)}")
    header = "  ".join(f"{n:>8}" for n in KNOB_PARAMS)
    print(f"{'sample':>6}  {header}")
    print("-" * 40)
    for i, row in enumerate(preds_batch.tolist()):
        vals = "  ".join(f"{v:8.4f}" for v in row)
        print(f"  [{i}]  {vals}")

    separator("완료 - 추론 정상 동작")


if __name__ == "__main__":
    main()
