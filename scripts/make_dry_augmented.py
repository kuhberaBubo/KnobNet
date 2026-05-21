"""
dry 오디오 볼륨 augmentation 스크립트

각 WAV 파일에 무작위 dB 스케일링(-24 ~ +24 dB)을 적용해서 저장.
출력 파일명: 폴더명_폴더명_파일명_±X.XdB.wav

Usage:
    python scripts/make_dry_augmented.py [--data-root PATH] [--out-dir PATH]
                                          [--n-per-file N] [--db-min X] [--db-max Y]
                                          [--workers N] [--seed N]

Example:
    python scripts/make_dry_augmented.py
    python scripts/make_dry_augmented.py --n-per-file 3 --workers 8
"""
import argparse
import os
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import soundfile as sf


# ── 소스 디렉터리 (data_root 기준 상대경로) ─────────────────────────────────
SOURCE_DIRS = [
    "clean-sequences",
    "flat",
]


def collect_wav_files(data_root: Path) -> list[Path]:
    """소스 디렉터리에서 WAV 파일 경로 수집"""
    wavs = []
    for src in SOURCE_DIRS:
        for path in sorted((data_root / src).rglob("*.wav")):
            wavs.append(path)
    return wavs


def make_output_name(wav_path: Path, data_root: Path, db: float) -> str:
    """
    data_root 기준 상대 경로 → 출력 파일명
    예: data/flat/bluesdriver/foo.wav, db=+12.3 → flat_bluesdriver_foo_+12.3dB.wav
    """
    rel = wav_path.relative_to(data_root)
    parts = list(rel.parts)        # ['flat', 'bluesdriver', 'foo.wav']
    parts[-1] = rel.stem           # ['flat', 'bluesdriver', 'foo']
    sign = "+" if db >= 0 else ""
    return "_".join(parts) + f"_{sign}{db:.1f}dB.wav"


def process_one(args):
    """단일 (파일, dB 목록) → 출력 저장. 멀티프로세싱 worker."""
    wav_path, db_list, data_root, out_dir = args
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        for db in db_list:
            scale = 10 ** (db / 20.0)
            out_audio = (audio * scale).astype(np.float32)
            # 클리핑 방지 (float32 범위 내 유지)
            out_audio = np.clip(out_audio, -1.0, 1.0)
            name = make_output_name(wav_path, data_root, db)
            sf.write(str(out_dir / name), out_audio, sr)
        return True, str(wav_path)
    except Exception as e:
        return False, f"{wav_path}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default="data",         help="data 루트 경로")
    parser.add_argument("--out-dir",    default=None,            help="출력 폴더 (기본: data/26.05.21/dry)")
    parser.add_argument("--n-per-file", type=int,   default=1,   help="파일당 augmented 버전 수")
    parser.add_argument("--db-min",     type=float, default=-24, help="dB 하한")
    parser.add_argument("--db-max",     type=float, default=24,  help="dB 상한")
    parser.add_argument("--workers",    type=int,   default=None, help="프로세스 수 (기본: CPU 코어 수)")
    parser.add_argument("--seed",       type=int,   default=42,   help="랜덤 시드")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir   = Path(args.out_dir).resolve() if args.out_dir else data_root / "26.05.21" / "dry"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or cpu_count()

    print(f"Data root : {data_root}")
    print(f"Output    : {out_dir}")
    print(f"dB range  : [{args.db_min}, {args.db_max}]")
    print(f"N per file: {args.n_per_file}")
    print(f"Workers   : {n_workers}")

    # WAV 파일 수집
    wav_files = collect_wav_files(data_root)
    print(f"Found     : {len(wav_files):,} WAV files\n")

    # 랜덤 dB 값 미리 생성 (재현 가능)
    rng = random.Random(args.seed)
    tasks = []
    for wav_path in wav_files:
        db_list = [rng.uniform(args.db_min, args.db_max) for _ in range(args.n_per_file)]
        tasks.append((wav_path, db_list, data_root, out_dir))

    total    = len(tasks)
    done     = 0
    errors   = []
    step     = max(1, total // 100)    # 1% 단위 출력

    with Pool(processes=n_workers) as pool:
        for ok, msg in pool.imap_unordered(process_one, tasks, chunksize=32):
            done += 1
            if not ok:
                errors.append(msg)
            if done % step == 0 or done == total:
                pct = done / total * 100
                print(f"  {pct:5.1f}%  ({done:,} / {total:,})", flush=True)

    print(f"\nDone : {done - len(errors):,} / {total:,} files saved to {out_dir}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
