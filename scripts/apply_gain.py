"""
폴더 하위의 모든 wav 파일에 gain을 적용하는 스크립트

사용법:
    python apply_gain.py                         # INPUT_DIR 하위 wav를 OUTPUT_DIR에 저장
    python apply_gain.py --in-place              # 원본 덮어쓰기
"""

from pathlib import Path

import soundfile as sf

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("data")  # --in-place 사용 시 무시됨
GAIN_DB = -32.0


def apply_gain_to_dir(input_dir: Path, output_dir: Path, gain_db: float, in_place: bool = False):
    gain = 10 ** (gain_db / 20)
    wav_files = list(input_dir.rglob("*.wav"))

    if not wav_files:
        print(f"No wav files found in {input_dir}")
        return

    for wav_file in sorted(wav_files):
        audio, sr = sf.read(wav_file)

        if in_place:
            dest = wav_file
        else:
            dest = output_dir / wav_file.relative_to(input_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)

        sf.write(dest, audio * gain, sr)
        print(f"{'(in-place) ' if in_place else ''}{wav_file} -> {dest}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--gain-db", type=float, default=GAIN_DB)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    apply_gain_to_dir(args.input_dir, args.output_dir, args.gain_db, args.in_place)
