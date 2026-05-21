"""
vol_input / vol_ref 의존도 ablation 분석

각 vol 피처를 데이터셋 평균으로 고정했을 때 예측이 얼마나 변하는지 측정.
MERT 추론은 한 번만 하고 head만 4가지 조건으로 재실행.

Usage:
    python scripts/ablation_vol.py <model.pt> <dataset_root> <wet_dir> <output.csv>

Example:
    python scripts/ablation_vol.py knobnet_final.pt . data/wet/black ablation_vol.csv
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


def run_head(model, vol_input, I_feat, vol_ref, O_feat):
    x = torch.cat([vol_input, I_feat, vol_ref, O_feat], dim=-1)
    return model.head(x)


def main():
    if len(sys.argv) < 5:
        print("Usage: python scripts/ablation_vol.py <model.pt> <dataset_root> <wet_dir> <output.csv>")
        sys.exit(1)

    model_path   = Path(sys.argv[1])
    dataset_root = Path(sys.argv[2])
    wet_dir      = sys.argv[3]
    output_csv   = Path(sys.argv[4])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {model_path}")
    print(f"Data   : {dataset_root} / {wet_dir}")

    model = KnobNet.from_exported(model_path, device=device)
    model.eval()

    dataset = KnobDataset(dataset_root, wet_dir=wet_dir)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    total   = len(dataset)
    print(f"Samples: {total:,}\n")

    K = len(KNOB_PARAMS)
    CONDITIONS = ["baseline", "fix_vol_ref", "fix_vol_input", "fix_both"]

    # ── 1단계: 전체 vol_input / vol_ref 평균 수집 ─────────────────────────────
    print("Step 1/2  collecting vol statistics ...")
    sum_vi, sum_vr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for input_audio, ref_audio, _ in loader:
            feats = model.extract_features(input_audio.to(device), ref_audio.to(device))
            sum_vi += feats["vol_input"].sum().item()
            sum_vr += feats["vol_ref"].sum().item()
            n += input_audio.size(0)

    mean_vi = torch.tensor([[sum_vi / n]], device=device)
    mean_vr = torch.tensor([[sum_vr / n]], device=device)
    print(f"  mean vol_input = {mean_vi.item():.6f}")
    print(f"  mean vol_ref   = {mean_vr.item():.6f}\n")

    # ── 2단계: 4가지 조건으로 head 재실행 ────────────────────────────────────
    print("Step 2/2  running ablation ...")

    mae_sum  = {c: torch.zeros(K) for c in CONDITIONS}
    rows = []   # CSV 저장용 per-sample 결과
    cnt  = 0
    step = max(1, total // 100)
    next_report = step

    with torch.no_grad():
        for input_audio, ref_audio, knobs in loader:
            input_audio = input_audio.to(device)
            ref_audio   = ref_audio.to(device)
            knobs_d     = knobs.to(device)
            B = input_audio.size(0)

            feats = model.extract_features(input_audio, ref_audio)
            vi = feats["vol_input"]            # (B, 1)
            vr = feats["vol_ref"]              # (B, 1)
            If = feats["I_feat"]               # (B, 768)
            Of = feats["O_feat"]               # (B, 768)

            vi_fix = mean_vi.expand(B, 1)
            vr_fix = mean_vr.expand(B, 1)

            preds = {
                "baseline":      run_head(model, vi,     If, vr,     Of),
                "fix_vol_ref":   run_head(model, vi,     If, vr_fix, Of),
                "fix_vol_input": run_head(model, vi_fix, If, vr,     Of),
                "fix_both":      run_head(model, vi_fix, If, vr_fix, Of),
            }

            for c in CONDITIONS:
                mae_sum[c] += (preds[c] - knobs_d).abs().sum(dim=0).cpu()

            # per-sample 행 저장
            for j in range(B):
                row = {
                    "vol_input": f"{vi[j, 0].item():.7f}",
                    "vol_ref":   f"{vr[j, 0].item():.7f}",
                }
                for p_idx, p in enumerate(KNOB_PARAMS):
                    row[f"true_{p}"] = f"{knobs[j, p_idx].item():.6f}"
                for c in CONDITIONS:
                    for p_idx, p in enumerate(KNOB_PARAMS):
                        row[f"{c}_{p}"] = f"{preds[c][j, p_idx].item():.6f}"
                rows.append(row)

            cnt += B
            if cnt >= next_report:
                print(f"  {cnt/total*100:5.1f}%  ({cnt:,}/{total:,})", flush=True)
                next_report += step

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved : {output_csv}  ({cnt:,} rows)")

    # ── 콘솔 요약 ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Ablation 결과  (n={cnt:,})")
    print(f"{'─'*65}")
    print(f"  {'조건':<18}" + "".join(f"  {p:>8}" for p in KNOB_PARAMS) + "   avg")
    print(f"{'─'*65}")
    for c in CONDITIONS:
        mae = mae_sum[c] / cnt
        print(f"  {c:<18}" + "".join(f"  {v:8.4f}" for v in mae.tolist()) + f"   {mae.mean():.4f}")

    print(f"{'─'*65}")
    print("  [delta = fix − baseline]  양수 = 해당 피처 제거 시 MAE 증가 = 의존도 높음")
    print(f"{'─'*65}")
    for c in ["fix_vol_ref", "fix_vol_input", "fix_both"]:
        delta = (mae_sum[c] - mae_sum["baseline"]) / cnt
        print(f"  {c:<18}" + "".join(f"  {v:+8.4f}" for v in delta.tolist()) + f"   {delta.mean():+.4f}")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
