"""
학습된 모델로 전체 데이터셋에 대해 노브 예측 후 CSV 저장

Usage:
    python scripts/predict_all.py <model.pt> <dataset_root> <wet_dir> <output.csv>

Example (local):
    python scripts/predict_all.py knobnet_final.pt data data/wet/black predictions.csv

Example (Colab):
    python scripts/predict_all.py /content/drive/MyDrive/KnobNet/knobnet_final.pt \
        /content/KnobNet /content/KnobNet/data/wet/black predictions.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from dataset.dataset import KnobDataset
from model.model import KnobNet
from utils.config import KNOB_PARAMS


def main():
    if len(sys.argv) < 5:
        print("Usage: python scripts/predict_all.py <model.pt> <dataset_root> <wet_dir> <output.csv>")
        sys.exit(1)

    model_path   = Path(sys.argv[1])
    dataset_root = Path(sys.argv[2])
    wet_dir      = sys.argv[3]
    output_csv   = Path(sys.argv[4])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {model_path}")
    print(f"Data   : {dataset_root} / {wet_dir}")

    # 모델 로드
    model = KnobNet.from_exported(model_path, device=device)
    model.eval()

    # 데이터셋 로드 (shuffle=False → items 순서 보존)
    dataset = KnobDataset(dataset_root, wet_dir=wet_dir)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Samples: {len(dataset):,}")

    # CSV 헤더 먼저 작성
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_cols = [f"pred_{p}" for p in KNOB_PARAMS]
    true_cols = [f"true_{p}" for p in KNOB_PARAMS]

    total      = len(dataset)
    step       = max(1, total // 100)   # 1% 단위
    processed  = 0
    item_idx   = 0
    next_report = step

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_file", "output_file"] + true_cols + pred_cols)

        with torch.no_grad():
            for input_audio, ref_audio, knobs in loader:
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    preds = model(input_audio.to(device), ref_audio.to(device)).cpu()

                for j in range(len(knobs)):
                    input_path, ref_path, _, _ = dataset.items[item_idx]
                    writer.writerow([
                        input_path,
                        ref_path,
                        *[f"{v:.6f}" for v in knobs[j].tolist()],
                        *[f"{v:.6f}" for v in preds[j].tolist()],
                    ])
                    item_idx += 1

                processed += len(knobs)
                if processed >= next_report:
                    pct = processed / total * 100
                    print(f"  {pct:5.1f}%  ({processed:,} / {total:,})", flush=True)
                    next_report += step

    print(f"Done  : {output_csv}  ({total:,} rows)")


if __name__ == "__main__":
    main()
