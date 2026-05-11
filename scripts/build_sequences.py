"""
음 이동 있는 시퀀스 오디오생성하는 코드
data/raw/egfxset를 기준으로 하죠

clean_bridge_sequence_001.wav
이런 식으로 간단하게 하죠

같은 픽업에 대해서 오디오 n개를 고르고
최소 길이를 0.5초로 해서 오디오 상 n-1개의 지점을 골라서
각 오디오를 처음부터 잘라서 입력
"""

import random
import numpy as np
import soundfile as sf
from pathlib import Path

# Options
EFFECT_DIR = Path("data/raw/egfxset/Clean")
OUTPUT_DIR = Path("data/clean-sequences")
NUM_NOTES = 3
NUM_SEQUENCES = 3000
MIN_DURATION = 0.5  # seconds
GAIN_DB = -32.0
SEED = 42


def sample_cut_points(total_duration: float, num_notes: int, min_duration: float, rng: random.Random) -> list:
    """num_notes-1개의 cut point 생성. 각 구간이 min_duration 이상임을 보장."""
    slack = total_duration - num_notes * min_duration
    points = sorted(rng.uniform(0, slack) for _ in range(num_notes - 1))
    cut_points = [points[i] + (i + 1) * min_duration for i in range(num_notes - 1)]
    return cut_points


def build_sequence(effect_dir: Path, num_notes: int, min_duration: float, rng: random.Random):
    # 1. 픽업 폴더 랜덤 선택
    pickup_dirs = [d for d in effect_dir.iterdir() if d.is_dir()]
    pickup_dir = rng.choice(pickup_dirs)

    # 2. 픽업 폴더에서 num_notes개 wav 샘플링
    wav_files = list(pickup_dir.glob("*.wav"))
    chosen_files = rng.sample(wav_files, num_notes)

    # 3. 오디오 로드
    audios = []
    sample_rate = None
    for f in chosen_files:
        data, sr = sf.read(f)
        audios.append(data)
        if sample_rate is None:
            sample_rate = sr

    total_samples = len(audios[0])
    total_duration = total_samples / sample_rate

    # 4. cut point 생성 및 구간 경계를 샘플 단위(정수)로 변환
    cut_points = sample_cut_points(total_duration, num_notes, min_duration, rng)
    cut_samples = [round(cp * sample_rate) for cp in cut_points]
    boundaries = [0] + cut_samples + [total_samples]

    # 5. 각 음을 앞에서부터 구간 길이만큼 잘라서 붙이기
    segments = []
    for i, audio in enumerate(audios):
        seg_samples = boundaries[i + 1] - boundaries[i]
        segments.append(audio[:seg_samples])

    sequence = np.concatenate(segments, axis=0)
    return sequence, sample_rate, pickup_dir.name.lower()


def main():
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    effect_name = EFFECT_DIR.name.lower()

    for i in range(NUM_SEQUENCES):
        sequence, sr, pickup_name = build_sequence(EFFECT_DIR, NUM_NOTES, MIN_DURATION, rng)
        filename = f"{effect_name}_{pickup_name}_sequence_{i + 1:03d}.wav"
        output_path = OUTPUT_DIR / filename
        sf.write(output_path, sequence * (10 ** (GAIN_DB / 20)), sr)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
