"""
전체 파이프라인 smoke test
- 가짜 데이터 2개로 dataset → model → train 루프까지 돌아가는지 확인
python tests/test_model.py
"""
import csv
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import MERT_SR, CLIP_SAMPLES, KNOB_PARAMS
from dataset.dataset import KnobDataset
from dataset.loader import make_loaders
from model.model import KnobNet
from train.train import (
    make_optimizer, run_epoch, evaluate,
    evaluate_per_param, evaluate_accuracy,
    log_param_mae, log_accuracy,
)

EXPORT_PATH = Path(__file__).parent.parent / "data" / "models" / "tmp" / "knobnet_smoke.pt"


def make_fake_dataset(root: Path, n: int = 2):
    """가짜 WAV + samples.csv 생성"""
    input_dir = root / "input"
    wet_dir   = root / "wet"
    input_dir.mkdir(parents=True)
    wet_dir.mkdir(parents=True)

    audio = np.random.randn(CLIP_SAMPLES).astype(np.float32) * 0.1

    rows = []
    for i in range(n):
        input_name = f"input_{i:03d}.wav"
        wet_name   = f"sample_{i:05d}.wav"
        sf.write(input_dir / input_name, audio, MERT_SR)
        sf.write(wet_dir   / wet_name,   audio, MERT_SR)
        knobs = {p: round(np.random.uniform(0, 1), 4) for p in KNOB_PARAMS}
        rows.append({"input_file": f"input/{input_name}", "output_file": wet_name, **knobs})

    with open(wet_dir / "samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_file", "output_file"] + KNOB_PARAMS)
        writer.writeheader()
        writer.writerows(rows)


def separator(title):
    print(f"\n{'-'*50}\n  {title}\n{'-'*50}")


def main():
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # ── 1. 가짜 데이터 생성 ──────────────────────────────
        separator("1. 가짜 데이터 생성")
        make_fake_dataset(root, n=2)
        print(f"dataset_root : {root}")
        print(f"WAV 생성     : input × 2, wet × 2")

        # ── 2. Dataset ───────────────────────────────────────
        separator("2. KnobDataset 로드")
        ds = KnobDataset(root, wet_dir="wet")
        print(f"샘플 수      : {len(ds)}")
        inp, ref, knobs = ds[0]
        print(f"input_audio  : {tuple(inp.shape)}  dtype={inp.dtype}")
        print(f"ref_audio    : {tuple(ref.shape)}")
        print(f"knobs        : {knobs.tolist()}")

        # ── 3. DataLoader ────────────────────────────────────
        separator("3. make_loaders")
        # 샘플 2개라 val_split 최소화
        train_loader, val_loader = make_loaders(root, wet_dir="wet", batch_size=2, val_split=0.5)

        # ── 4. 모델 forward ──────────────────────────────────
        separator("4. KnobNet forward")
        model = KnobNet(num_knobs=len(KNOB_PARAMS), freeze_mert=True).to(device)
        model.summary()

        inp_b, ref_b, knobs_b = next(iter(train_loader))
        preds = model(inp_b.to(device), ref_b.to(device))
        print(f"preds shape  : {tuple(preds.shape)}")
        print(f"preds        : {preds.detach().tolist()}")

        # ── 5. 학습 루프 ─────────────────────────────────────
        separator("5. 학습 2 epoch")
        criterion = nn.L1Loss()
        optimizer = make_optimizer(model, lr=1e-3, phase=1)

        for epoch in range(1, 3):
            train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
            val_loss   = evaluate(model, val_loader, criterion, device)
            mae        = evaluate_per_param(model, val_loader, device)
            acc        = evaluate_accuracy(model, val_loader, device, tolerance=0.1)
            print(f"  epoch {epoch}  train={train_loss:.4f}  val={val_loss:.4f}")
            log_param_mae(mae)
            log_accuracy(acc, tolerance=0.1)

        # ── 6. 모델 export ───────────────────────────────────
        separator("6. 모델 export")
        EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.export(EXPORT_PATH)
        print(f"저장 경로 : {EXPORT_PATH}")
        print(f"파일 크기 : {EXPORT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    separator("완료 - 모든 파이프라인 정상 동작")


if __name__ == "__main__":
    main()
