import csv
import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from dataset.dataset import KnobDataset
from model.model import KnobNet, KnobNetJoint
from utils.config import KNOB_PARAMS


# ── 캐시 키 ───────────────────────────────────────────────────────────────────

def _cache_key(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode()).hexdigest() + ".pt"


def _layer_cache_dir(base: Path, layer_idx: int) -> Path:
    """layer_idx별 캐시 서브디렉터리 반환."""
    return base / (f"layer{layer_idx:02d}" if layer_idx != -1 else "last")


def _head_forward(model, x: torch.Tensor) -> torch.Tensor:
    """캐시 임베딩 기반 forward (head_a → head_b 2단계)."""
    pred_dt    = model.head_a(x)
    x_b        = torch.cat([x, pred_dt.detach()], dim=-1)
    pred_level = model.head_b(x_b)
    return torch.cat([pred_dt[:, :1], pred_level, pred_dt[:, 1:]], dim=-1)


# ── 임베딩 캐시 생성 ──────────────────────────────────────────────────────────

def cache_embeddings(
    dataset_root: str | Path,
    wet_dir: str = "wet",
    input_dirs: list[str] | None = None,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    layer_idx: int = -1,
):
    """오디오 파일을 MERT로 인코딩해 캐시 저장. 이미 캐시된 파일은 스킵."""
    dataset_root = Path(dataset_root)
    base_dir  = Path(cache_dir) if cache_dir else dataset_root / "cache"
    cache_dir = _layer_cache_dir(base_dir, layer_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"cache dir : {cache_dir}")
    print(f"device    : {device}")

    ds = KnobDataset(dataset_root, wet_dir=wet_dir, input_dirs=input_dirs)
    unique_paths: set[Path] = set()
    for input_path, ref_path, _, _ in ds.items:
        unique_paths.add(input_path)
        unique_paths.add(ref_path)

    print(f"유니크 오디오 파일 : {len(unique_paths):,}개")

    model = KnobNet(freeze_mert=True, layer_idx=layer_idx).to(device)
    model.eval()

    skipped = 0
    for path in tqdm(sorted(unique_paths), desc="caching"):
        out_path = cache_dir / _cache_key(path)
        if out_path.exists():
            skipped += 1
            continue

        audio = ds._load_audio(path).unsqueeze(0).to(device)   # (1, T)
        with torch.no_grad():
            vol  = model._rms(audio).cpu()                       # (1, 1)
            feat = model._encode(model._normalize(audio)).cpu()  # (1, 768)

        torch.save({"feat": feat, "vol": vol}, out_path)

    done = len(unique_paths) - skipped
    print(f"완료: {done:,}개 인코딩 / {skipped:,}개 스킵")


# ── 캐시 데이터셋 + 로더 ──────────────────────────────────────────────────────

class _CachedKnobDataset(Dataset):
    """오디오 대신 사전 계산된 MERT 임베딩을 로드."""

    def __init__(
        self,
        dataset_root: str | Path,
        wet_dir: str = "wet",
        input_dirs: list[str] | None = None,
        cache_dir: str | Path | None = None,
        layer_idx: int = -1,
    ):
        self.dataset_root = Path(dataset_root)
        self.wet_dir = self.dataset_root / wet_dir
        self.input_dirs = input_dirs
        base_dir = Path(cache_dir) if cache_dir else self.dataset_root / "cache"
        self.cache_dir = _layer_cache_dir(base_dir, layer_idx)
        self.items = self._load_csv()

    def _load_csv(self) -> list:
        items = []
        with open(self.wet_dir / "samples.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                inp = self._resolve(row["input_file"])
                ref = self.wet_dir / row["output_file"]
                if not inp.exists() or not ref.exists():
                    continue
                if not self._in_dirs(inp):
                    continue
                knobs = torch.tensor([float(row[p]) for p in KNOB_PARAMS], dtype=torch.float32)
                items.append((inp, ref, knobs, row["input_file"]))
        return items

    def _resolve(self, f: str) -> Path:
        p = Path(f.replace("\\", "/"))
        return p if p.is_absolute() else self.dataset_root / p

    def _in_dirs(self, path: Path) -> bool:
        if self.input_dirs is None:
            return True
        return any(part in self.input_dirs for part in path.parts)

    def _load(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(self.cache_dir / _cache_key(path), weights_only=True)
        return data["feat"].squeeze(0), data["vol"].squeeze(0)  # (768,), (1,)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        inp, ref, knobs, _ = self.items[idx]
        I_feat, vol_i = self._load(inp)
        O_feat, vol_r = self._load(ref)
        return I_feat, vol_i, O_feat, vol_r, knobs

    def unique_inputs(self) -> dict:
        groups: dict[str, list[int]] = {}
        for i, (*_, input_file) in enumerate(self.items):
            groups.setdefault(input_file, []).append(i)
        return groups


def make_loaders_cached(
    dataset_root,
    wet_dir: str = "wet",
    input_dirs: list[str] | None = None,
    cache_dir=None,
    batch_size: int = 256,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
    layer_idx: int = -1,
) -> tuple[DataLoader, DataLoader]:
    """캐시된 임베딩 기반 DataLoader 생성 (오디오 로드·MERT 실행 없음)."""
    ds = _CachedKnobDataset(dataset_root, wet_dir=wet_dir,
                            input_dirs=input_dirs, cache_dir=cache_dir,
                            layer_idx=layer_idx)

    groups = ds.unique_inputs()
    unique_inputs = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_inputs)

    val_count = max(1, int(len(unique_inputs) * val_split))
    val_inputs = set(unique_inputs[:val_count])
    train_inputs = set(unique_inputs[val_count:])

    train_idx = [i for f, idxs in groups.items() if f in train_inputs for i in idxs]
    val_idx   = [i for f, idxs in groups.items() if f in val_inputs   for i in idxs]

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, **kw)

    print(f"input 파일 수 : {len(unique_inputs):,}  (train {len(train_inputs):,} / val {len(val_inputs):,})")
    print(f"sample 수     : train {len(train_idx):,} / val {len(val_idx):,}")
    return train_loader, val_loader


# ── 캐시 기반 학습 루프 ───────────────────────────────────────────────────────

def _to_head_input(I_feat, vol_i, O_feat, vol_r, device) -> torch.Tensor:
    return torch.cat([
        vol_i.to(device),
        I_feat.to(device),
        vol_r.to(device),
        O_feat.to(device),
    ], dim=-1)   # (B, 1538)


def run_epoch_cached(model, loader, optimizer, criterion, device, scaler=None) -> float:
    """캐시 임베딩 기반 1 epoch 학습, 평균 train loss 반환 (head만 업데이트)."""
    model.train()
    total = 0.0
    amp = scaler is not None
    bar = tqdm(loader, desc="  train", leave=False)
    for I_feat, vol_i, O_feat, vol_r, knobs in bar:
        x = _to_head_input(I_feat, vol_i, O_feat, vol_r, device)
        knobs = knobs.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=amp):
            loss = criterion(_head_forward(model, x), knobs)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.head_parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head_parameters(), 1.0)
            optimizer.step()

        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


def evaluate_cached(model, loader, criterion, device) -> float:
    """캐시 임베딩 기반 validation loss 계산 (gradient 없음)."""
    model.eval()
    total = 0.0
    amp = str(device).startswith("cuda")
    with torch.no_grad():
        for I_feat, vol_i, O_feat, vol_r, knobs in tqdm(loader, desc="    val", leave=False):
            x = _to_head_input(I_feat, vol_i, O_feat, vol_r, device)
            with torch.autocast(device_type="cuda", enabled=amp):
                total += criterion(_head_forward(model, x), knobs.to(device)).item()
    return total / len(loader)


def evaluate_all_cached(model, loader, criterion, device, tolerance: float = 0.1) -> dict:
    """val loss, MAE, accuracy를 단일 forward pass로 계산  →  val pass 3회 → 1회로 단축."""
    model.eval()
    amp = str(device).startswith("cuda")
    loss_total  = 0.0
    mae_total   = torch.zeros(len(KNOB_PARAMS))
    correct     = torch.zeros(len(KNOB_PARAMS))
    correct_all = 0
    n_samples   = 0
    with torch.no_grad():
        for I_feat, vol_i, O_feat, vol_r, knobs in tqdm(loader, desc="    val", leave=False):
            x = _to_head_input(I_feat, vol_i, O_feat, vol_r, device)
            with torch.autocast(device_type="cuda", enabled=amp):
                preds = _head_forward(model, x)
            loss_total += criterion(preds, knobs.to(device)).item()
            preds_cpu   = preds.cpu()
            diff        = (preds_cpu - knobs).abs()
            mae_total  += diff.mean(dim=0)
            within      = diff <= tolerance
            correct    += within.float().sum(dim=0)
            correct_all += within.all(dim=1).float().sum().item()
            n_samples  += len(knobs)
    mae = mae_total / len(loader)
    acc = {name: (correct[i] / n_samples).item() for i, name in enumerate(KNOB_PARAMS)}
    acc["all"] = correct_all / n_samples
    return {
        "val_loss": loss_total / len(loader),
        "mae":      {name: mae[i].item() for i, name in enumerate(KNOB_PARAMS)},
        "acc":      acc,
    }


def evaluate_per_param_cached(model, loader, device) -> dict[str, float]:
    """파라미터별 MAE 계산 (캐시 임베딩 기반)  →  {"drive": 0.043, "level": 0.021, ...}"""
    model.eval()
    totals = torch.zeros(len(KNOB_PARAMS))
    amp = str(device).startswith("cuda")
    with torch.no_grad():
        for I_feat, vol_i, O_feat, vol_r, knobs in loader:
            with torch.autocast(device_type="cuda", enabled=amp):
                preds = _head_forward(model, _to_head_input(I_feat, vol_i, O_feat, vol_r, device)).cpu()
            totals += (preds - knobs).abs().mean(dim=0)
    mae = totals / len(loader)
    return {name: mae[i].item() for i, name in enumerate(KNOB_PARAMS)}


def evaluate_accuracy_cached(model, loader, device, tolerance: float = 0.1) -> dict[str, float]:
    """
    tolerance 이내를 정답으로 볼 때 파라미터별 정답률 (0~1)
    예) tolerance=0.1 → 예측이 ±0.1 이내면 정답
    반환: {"drive": 0.82, "level": 0.74, "filter": 0.91, "all": 0.68}
          all = 세 파라미터 모두 맞춰야 정답
    """
    model.eval()
    correct = torch.zeros(len(KNOB_PARAMS))
    correct_all = 0
    total = 0
    amp = str(device).startswith("cuda")
    with torch.no_grad():
        for I_feat, vol_i, O_feat, vol_r, knobs in loader:
            with torch.autocast(device_type="cuda", enabled=amp):
                preds = _head_forward(model, _to_head_input(I_feat, vol_i, O_feat, vol_r, device)).cpu()
            within = (preds - knobs).abs() <= tolerance
            correct     += within.float().sum(dim=0)
            correct_all += within.all(dim=1).float().sum().item()
            total       += len(knobs)
    acc = {name: (correct[i] / total).item() for i, name in enumerate(KNOB_PARAMS)}
    acc["all"] = correct_all / total
    return acc


# ── Joint 캐시 ────────────────────────────────────────────────────────────────

def _pair_cache_key(inp: Path, ref: Path) -> str:
    """(input_path, ref_path) 쌍의 캐시 파일명."""
    combined = str(inp.resolve()) + "|" + str(ref.resolve())
    return hashlib.md5(combined.encode()).hexdigest() + ".pt"


def _joint_cache_dir(base: Path, layer_idx: int) -> Path:
    """joint 캐시 서브디렉터리 반환."""
    suffix = f"layer{layer_idx:02d}" if layer_idx != -1 else "last"
    return base / f"joint_{suffix}"


def cache_embeddings_joint(
    dataset_root: str | Path,
    wet_dir: str = "wet",
    input_dirs: list[str] | None = None,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    layer_idx: int = -1,
):
    """input+ref 쌍을 concat해 MERT로 인코딩 후 쌍 단위 캐시 저장.
    이미 캐시된 쌍은 스킵."""
    dataset_root = Path(dataset_root)
    base_dir  = Path(cache_dir) if cache_dir else dataset_root / "cache"
    cache_dir = _joint_cache_dir(base_dir, layer_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"joint cache dir : {cache_dir}")
    print(f"device          : {device}")

    ds = KnobDataset(dataset_root, wet_dir=wet_dir, input_dirs=input_dirs)
    unique_pairs: set[tuple[Path, Path]] = set()
    for input_path, ref_path, _, _ in ds.items:
        unique_pairs.add((input_path, ref_path))

    print(f"유니크 오디오 쌍 : {len(unique_pairs):,}개")

    model = KnobNetJoint(freeze_mert=True, layer_idx=layer_idx).to(device)
    model.eval()

    skipped = 0
    for inp_path, ref_path in tqdm(sorted(unique_pairs), desc="caching joint"):
        out_path = cache_dir / _pair_cache_key(inp_path, ref_path)
        if out_path.exists():
            skipped += 1
            continue

        inp_audio = ds._load_audio(inp_path).unsqueeze(0).to(device)  # (1, T)
        ref_audio = ds._load_audio(ref_path).unsqueeze(0).to(device)  # (1, T)

        with torch.no_grad():
            vol_i = model._rms(inp_audio).cpu()                          # (1, 1)
            vol_r = model._rms(ref_audio).cpu()                          # (1, 1)
            I_feat, O_feat = model._encode_joint(
                model._normalize(inp_audio),
                model._normalize(ref_audio),
            )
            I_feat = I_feat.cpu()   # (1, 768)
            O_feat = O_feat.cpu()   # (1, 768)

        torch.save({"I_feat": I_feat, "vol_i": vol_i,
                    "O_feat": O_feat, "vol_r": vol_r}, out_path)

    done = len(unique_pairs) - skipped
    print(f"완료: {done:,}개 인코딩 / {skipped:,}개 스킵")


class _CachedKnobDatasetJoint(Dataset):
    """쌍 단위 joint 캐시 기반 Dataset."""

    def __init__(
        self,
        dataset_root: str | Path,
        wet_dir: str = "wet",
        input_dirs: list[str] | None = None,
        cache_dir: str | Path | None = None,
        layer_idx: int = -1,
    ):
        self.dataset_root = Path(dataset_root)
        self.wet_dir = self.dataset_root / wet_dir
        self.input_dirs = input_dirs
        base_dir = Path(cache_dir) if cache_dir else self.dataset_root / "cache"
        self.cache_dir = _joint_cache_dir(base_dir, layer_idx)
        self.items = self._load_csv()

    def _load_csv(self) -> list:
        items = []
        with open(self.wet_dir / "samples.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                inp = self._resolve(row["input_file"])
                ref = self.wet_dir / row["output_file"]
                if not inp.exists() or not ref.exists():
                    continue
                if not self._in_dirs(inp):
                    continue
                knobs = torch.tensor([float(row[p]) for p in KNOB_PARAMS], dtype=torch.float32)
                items.append((inp, ref, knobs, row["input_file"]))
        return items

    def _resolve(self, f: str) -> Path:
        p = Path(f.replace("\\", "/"))
        return p if p.is_absolute() else self.dataset_root / p

    def _in_dirs(self, path: Path) -> bool:
        if self.input_dirs is None:
            return True
        return any(part in self.input_dirs for part in path.parts)

    def _load(self, inp_path: Path, ref_path: Path):
        data = torch.load(self.cache_dir / _pair_cache_key(inp_path, ref_path),
                          weights_only=True)
        return (data["I_feat"].squeeze(0), data["vol_i"].squeeze(0),
                data["O_feat"].squeeze(0), data["vol_r"].squeeze(0))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        inp, ref, knobs, _ = self.items[idx]
        I_feat, vol_i, O_feat, vol_r = self._load(inp, ref)
        return I_feat, vol_i, O_feat, vol_r, knobs

    def unique_inputs(self) -> dict:
        groups: dict[str, list[int]] = {}
        for i, (*_, input_file) in enumerate(self.items):
            groups.setdefault(input_file, []).append(i)
        return groups


def make_loaders_cached_joint(
    dataset_root,
    wet_dir: str = "wet",
    input_dirs: list[str] | None = None,
    cache_dir=None,
    batch_size: int = 256,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
    layer_idx: int = -1,
) -> tuple[DataLoader, DataLoader]:
    """joint 캐시 기반 DataLoader 생성 (run_epoch_cached 재사용 가능)."""
    ds = _CachedKnobDatasetJoint(dataset_root, wet_dir=wet_dir,
                                  input_dirs=input_dirs, cache_dir=cache_dir,
                                  layer_idx=layer_idx)

    groups = ds.unique_inputs()
    unique_inputs = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_inputs)

    val_count = max(1, int(len(unique_inputs) * val_split))
    val_inputs = set(unique_inputs[:val_count])
    train_inputs = set(unique_inputs[val_count:])

    train_idx = [i for f, idxs in groups.items() if f in train_inputs for i in idxs]
    val_idx   = [i for f, idxs in groups.items() if f in val_inputs   for i in idxs]

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, **kw)

    print(f"input 파일 수 : {len(unique_inputs):,}  (train {len(train_inputs):,} / val {len(val_inputs):,})")
    print(f"sample 수     : train {len(train_idx):,} / val {len(val_idx):,}")
    return train_loader, val_loader


# ── 사용 예시 ──────────────────────────────────────────────────────────────────
#
# from train.cache import (
#     cache_embeddings, make_loaders_cached,
#     run_epoch_cached, evaluate_cached,
#     evaluate_per_param_cached, evaluate_accuracy_cached,
# )
# from train.train import (
#     make_optimizer, save_checkpoint, load_checkpoint,
#     log_epoch, log_param_mae, log_accuracy,
# )
# from model.model import KnobNet, KnobNetJoint
# import torch.nn as nn
#
# # ── Step 1: 한 번만 실행 (GPU 필요) ──
# cache_embeddings(PROJECT_DIR, wet_dir="data/wet/black")
#
# # ── Step 2: 캐시 로더 생성 ──
# train_loader_c, val_loader_c = make_loaders_cached(
#     dataset_root = PROJECT_DIR,
#     wet_dir      = "data/wet/black",
#     batch_size   = 256,
# )
#
# # ── Step 3: Phase 1 학습 (head만) ──
# model = KnobNet(num_knobs=3, freeze_mert=True).to(device)
# criterion = nn.L1Loss()
# optimizer = make_optimizer(model, lr=1e-3, phase=1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
#
# best_val = float("inf")
# for epoch in range(1, 31):
#     train_loss = run_epoch_cached(model, train_loader_c, optimizer, criterion, device)
#     val_loss   = evaluate_cached(model, val_loader_c, criterion, device)
#     scheduler.step()
#
#     improved = val_loss < best_val
#     if improved:
#         best_val = val_loss
#         save_checkpoint(save_path, model, epoch, phase=1, val_loss=val_loss,
#                         optimizer=optimizer, scheduler=scheduler)
#     log_epoch(1, epoch, 30, train_loss, val_loss, improved)
#     log_param_mae(evaluate_per_param_cached(model, val_loader_c, device))
#     log_accuracy(evaluate_accuracy_cached(model, val_loader_c, device, 0.1), 0.1)
