"""
data/raw/egfxset의 폴더의 있는 raw audio를 정리하는 코드
egfxset/Clean/Bridge/4_14.wav -> data/clean/clean_bridge_414.wav
egfxset/Rat/Bridge/4_14.wav   -> data/rat/rat_bridge_414.wav
"""

from pathlib import Path

import numpy as np
import soundfile as sf

EGFXSET_DIR = Path("data/raw/egfxset")
OUTPUT_DIR = Path("data/flat")
GAIN_DB = -32.0


def flatten_egfxset(egfxset_dir: Path, output_dir: Path, gain_db: float = GAIN_DB):
    for effect_dir in sorted(egfxset_dir.iterdir()):
        if not effect_dir.is_dir():
            continue
        effect_name = effect_dir.name.lower()  # e.g. "clean", "rat"
        dest_dir = output_dir / effect_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for pickup_dir in sorted(effect_dir.iterdir()):
            if not pickup_dir.is_dir():
                continue
            pickup_name = pickup_dir.name.lower()  # e.g. "bridge", "neck"

            for wav_file in sorted(pickup_dir.glob("*.wav")):
                # 4_14.wav -> 414
                number = wav_file.stem.replace("_", "")
                new_name = f"{effect_name}_{pickup_name}_{number}.wav"
                dest_path = dest_dir / new_name
                audio, sr = sf.read(wav_file)
                sf.write(dest_path, audio * (10 ** (gain_db / 20)), sr)
                print(f"{wav_file} -> {dest_path}")


if __name__ == "__main__":
    flatten_egfxset(EGFXSET_DIR, OUTPUT_DIR)
