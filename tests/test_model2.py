"""
실제 데이터(data/wet/black) 100개 샘플로 CPU 학습 테스트
- data/wet/black/samples.csv 에서 100개 행만 추출
- 실제 오디오 파일로 KnobDataset → KnobNet → train 루프 검증
python tests/test_model2.py
"""
import csv
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from utils.config import MERT_SR, CLIP_SAMPLES, KNOB_PARAMS
from dataset.dataset import KnobDataset
from dataset.loader import make_loaders
from model.model import KnobNet
from train.train import (
    make_optimizer, run_epoch, evaluate,
    evaluate_per_param, evaluate_accuracy,
    log_param_mae, log_accuracy,
)

PROJECT_ROOT = Path(__file__).parent.parent
REAL_CSV     = PROJECT_ROOT / "data" / "wet" / "black" / "samples.csv"
REAL_WET_DIR = PROJECT_ROOT / "data" / "wet" / "black"
N_SAMPLES    = 20


def separator(title):
    print(f"\n{'-'*50}\n  {title}\n{'-'*50}")


def build_subset_csv(dst_dir: Path, n: int) -> int:
    """
    REAL_CSV 에서 n개 행을 읽어 dst_dir/samples.csv 로 저장.
    - 빈 knob 값 → 0.0
    - input_file / output_file → 절대 경로 (KnobDataset의 경로 해석에 맞춤)
    반환: 실제로 쓴 행 수
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_csv = dst_dir / "samples.csv"

    written = 0
    with open(REAL_CSV, newline="", encoding="utf-8") as fin, \
         open(out_csv, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if written >= n:
                break
            # 빈 knob 컬럼 → 0.0
            for p in KNOB_PARAMS:
                if row.get(p, "").strip() == "":
                    row[p] = "0.0"
            # 절대 경로 변환
            # input_file: CSV에 "data\clean-sequences\..." 형태로 저장되어 있음
            row["input_file"]  = str(PROJECT_ROOT / row["input_file"])
            # output_file: "sample_XXXXX.wav" → 실제 wet 디렉터리의 절대 경로
            row["output_file"] = str(REAL_WET_DIR / row["output_file"])
            writer.writerow(row)
            written += 1

    return written


def main():
    device = torch.device("cpu")

    # ── 전제조건 확인 ──────────────────────────────────────────
    if not REAL_CSV.exists():
        raise FileNotFoundError(
            f"실제 데이터 CSV 없음: {REAL_CSV}\n"
            "먼저 scripts/vst_sample.py 로 wet 데이터를 생성하세요."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # ── 1. subset CSV 생성 ────────────────────────────────
        separator("1. 실제 데이터 100개 subset CSV 생성")
        written = build_subset_csv(tmp_dir, N_SAMPLES)
        print(f"원본 CSV  : {REAL_CSV}")
        print(f"Subset CSV: {tmp_dir / 'samples.csv'}  ({written}개 행)")

        # ── 2. KnobDataset 로드 ───────────────────────────────
        separator("2. KnobDataset 로드 (실제 오디오)")
        # dataset_root=PROJECT_ROOT: input_file이 절대 경로이므로 그대로 사용됨
        # wet_dir=str(tmp_dir): Windows에서 절대경로 / 절대경로 = 절대경로로 처리됨
        ds = KnobDataset(dataset_root=PROJECT_ROOT, wet_dir=str(tmp_dir))
        print(f"로드된 샘플 수: {len(ds)}")
        if len(ds) == 0:
            raise RuntimeError(
                "샘플을 하나도 로드하지 못했습니다.\n"
                "input/output 파일 경로를 확인하세요."
            )
        inp, ref, knobs = ds[0]
        print(f"input_audio shape : {tuple(inp.shape)}  dtype={inp.dtype}")
        print(f"ref_audio   shape : {tuple(ref.shape)}")
        print(f"knobs             : {[round(v, 4) for v in knobs.tolist()]}")

        # ── 3. make_loaders ────────────────────────────────────
        separator("3. make_loaders (train/val 분리)")
        train_loader, val_loader = make_loaders(
            dataset_root=PROJECT_ROOT,
            wet_dir=str(tmp_dir),
            batch_size=8,
            val_split=0.2,
        )

        # ── 4. 모델 forward ────────────────────────────────────
        separator("4. KnobNet forward")
        model = KnobNet(num_knobs=len(KNOB_PARAMS), freeze_mert=True).to(device)
        model.summary()

        inp_b, ref_b, knobs_b = next(iter(train_loader))
        preds = model(inp_b.to(device), ref_b.to(device))
        print(f"preds shape : {tuple(preds.shape)}")
        print(f"preds       : {[[round(v, 4) for v in row] for row in preds.detach().tolist()]}")

        # ── 5. 학습 루프 2 epoch ───────────────────────────────
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

    separator("완료 - 실제 데이터 파이프라인 정상 동작")


if __name__ == "__main__":
    main()
