import random

from torch.utils.data import DataLoader, Subset

from .dataset import KnobDataset


def make_loaders(
    dataset_root,
    wet_dir: str = "wet",
    input_dirs: list[str] | None = None,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:

    # augment 여부만 다른 동일한 두 dataset (인덱스 순서 동일)
    train_ds = KnobDataset(dataset_root, wet_dir=wet_dir, input_dirs=input_dirs, augment=True)
    val_ds   = KnobDataset(dataset_root, wet_dir=wet_dir, input_dirs=input_dirs, augment=False)

    # input 파일 기준으로 train/val 분리
    groups = train_ds.unique_inputs()           # {input_file: [indices]}
    unique_inputs = sorted(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(unique_inputs)

    val_count   = max(1, int(len(unique_inputs) * val_split))
    val_inputs  = set(unique_inputs[:val_count])
    train_inputs = set(unique_inputs[val_count:])

    train_idx = [i for f, idxs in groups.items() if f in train_inputs for i in idxs]
    val_idx   = [i for f, idxs in groups.items() if f in val_inputs   for i in idxs]

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"input 파일 수 : {len(unique_inputs):,}  (train {len(train_inputs):,} / val {len(val_inputs):,})")
    print(f"sample 수     : train {len(train_idx):,} / val {len(val_idx):,}")

    return train_loader, val_loader
