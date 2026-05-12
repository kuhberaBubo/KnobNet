import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from utils.config import MERT_SR, CLIP_SAMPLES, KNOB_PARAMS


class KnobDataset(Dataset):

    def __init__(
        self,
        dataset_root: str | Path,
        wet_dir: str = "wet",
        input_dirs: list[str] | None = None,   # None이면 전체, 지정하면 해당 폴더만
        augment: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.wet_dir = self.dataset_root / wet_dir
        self.input_dirs = input_dirs
        self.augment = augment
        self.items = self._load_csv()

    def _load_csv(self) -> list:
        csv_path = self.wet_dir / "samples.csv"
        items = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                input_path = self._resolve_input(row["input_file"])
                ref_path = self.wet_dir / row["output_file"]
                if not input_path.exists() or not ref_path.exists():
                    continue
                if not self._in_input_dirs(input_path):
                    continue
                knobs = torch.tensor([float(row[p]) for p in KNOB_PARAMS], dtype=torch.float32)
                items.append((input_path, ref_path, knobs, row["input_file"]))
        return items

    def _in_input_dirs(self, input_path: Path) -> bool:
        if self.input_dirs is None:
            return True
        return any(
            part in self.input_dirs
            for part in input_path.parts
        )

    def _resolve_input(self, input_file: str) -> Path:
        p = Path(input_file)
        return p if p.is_absolute() else self.dataset_root / p

    def _load_audio(self, path: Path) -> torch.Tensor:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != MERT_SR:
            t = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(t, sr, MERT_SR).squeeze(0).numpy()

        # 뒤에서 자르거나 pad
        if len(audio) >= CLIP_SAMPLES:
            audio = audio[:CLIP_SAMPLES]
        else:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))

        return torch.from_numpy(audio.copy())  # (CLIP_SAMPLES,)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        input_path, ref_path, knobs, _ = self.items[idx]
        return self._load_audio(input_path), self._load_audio(ref_path), knobs

    def unique_inputs(self) -> dict:
        """input_file → [sample indices] 매핑, loader의 train/val 분리에 사용"""
        groups: dict[str, list[int]] = {}
        for i, (*_, input_file) in enumerate(self.items):
            groups.setdefault(input_file, []).append(i)
        return groups
