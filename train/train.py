from pathlib import Path

import torch
import torch.nn as nn

from utils.config import KNOB_PARAMS


# ── Core ──────────────────────────────────────────────────────────────────────

def make_optimizer(model, lr: float, phase: int, mert_lr_scale: float = 0.1):
    """phase 1: head만 / phase 2: head + MERT (낮은 lr)"""
    if phase == 1:
        return torch.optim.AdamW(model.head_parameters(), lr=lr, weight_decay=1e-4)
    return torch.optim.AdamW([
        {"params": model.head_parameters(), "lr": lr},
        {"params": model.mert_parameters(), "lr": lr * mert_lr_scale},
    ], weight_decay=1e-4)


def run_epoch(model, loader, optimizer, criterion, device) -> float:
    """1 epoch 학습, 평균 train loss 반환"""
    model.train()
    total = 0.0
    for input_audio, ref_audio, knobs in loader:
        input_audio = input_audio.to(device)
        ref_audio   = ref_audio.to(device)
        knobs       = knobs.to(device)

        optimizer.zero_grad()
        loss = criterion(model(input_audio, ref_audio), knobs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, criterion, device) -> float:
    """validation loss 계산 (gradient 없음)"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for input_audio, ref_audio, knobs in loader:
            input_audio = input_audio.to(device)
            ref_audio   = ref_audio.to(device)
            knobs       = knobs.to(device)
            total += criterion(model(input_audio, ref_audio), knobs).item()
    return total / len(loader)


def evaluate_per_param(model, loader, device) -> dict[str, float]:
    """파라미터별 MAE 계산  →  {"gain": 0.043, "level": 0.021, "filter": 0.087}"""
    model.eval()
    totals = torch.zeros(len(KNOB_PARAMS))
    with torch.no_grad():
        for input_audio, ref_audio, knobs in loader:
            preds = model(input_audio.to(device), ref_audio.to(device)).cpu()
            totals += (preds - knobs).abs().mean(dim=0)
    mae = totals / len(loader)
    return {name: mae[i].item() for i, name in enumerate(KNOB_PARAMS)}


def log_param_mae(mae: dict[str, float]):
    """파라미터별 MAE 출력"""
    parts = "  ".join(f"{name}={v:.4f}" for name, v in mae.items())
    print(f"  MAE  {parts}")


def save_checkpoint(path, model, epoch: int, phase: int, val_loss: float):
    """체크포인트 저장"""
    torch.save({
        "phase":       phase,
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_loss":    val_loss,
    }, path)


def load_checkpoint(path, model):
    """체크포인트 로드, 저장된 메타 정보 반환"""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return {k: v for k, v in ckpt.items() if k != "model_state"}


# ── 출력 / 디버그 ──────────────────────────────────────────────────────────────

def log_epoch(phase: int, epoch: int, total_epochs: int,
              train_loss: float, val_loss: float, improved: bool):
    """포맷된 epoch 로그"""
    mark = " *" if improved else ""
    print(f"[phase {phase}] ep {epoch:3d}/{total_epochs}"
          f"  train={train_loss:.4f}  val={val_loss:.4f}{mark}")


def print_batch(input_audio, ref_audio, knobs):
    """배치 shape 및 통계 확인"""
    print(f"input_audio : {tuple(input_audio.shape)}  "
          f"min={input_audio.min():.3f}  max={input_audio.max():.3f}")
    print(f"ref_audio   : {tuple(ref_audio.shape)}  "
          f"min={ref_audio.min():.3f}  max={ref_audio.max():.3f}")
    print(f"knobs       : {tuple(knobs.shape)}")
    for i, name in enumerate(KNOB_PARAMS):
        v = knobs[:, i]
        print(f"  {name:8s}  mean={v.mean():.3f}  std={v.std():.3f}  "
              f"[{v.min():.3f}, {v.max():.3f}]")


def print_predictions(model, loader, device, n: int = 8):
    """예측값 vs 실제값 비교 출력"""
    model.eval()
    input_audio, ref_audio, knobs = next(iter(loader))
    input_audio = input_audio[:n].to(device)
    ref_audio   = ref_audio[:n].to(device)
    knobs       = knobs[:n]

    with torch.no_grad():
        preds = model(input_audio, ref_audio).cpu()

    header = "  ".join(f"{name:>8}" for name in KNOB_PARAMS)
    print(f"\n{'sample':>6}  {'':>5}  {header}")
    print(f"{'─'*55}")
    for i in range(len(knobs)):
        pred_str = "  ".join(f"{preds[i, j]:8.3f}" for j in range(preds.shape[1]))
        true_str = "  ".join(f"{knobs[i, j]:8.3f}" for j in range(knobs.shape[1]))
        print(f"  [{i:3d}]  pred   {pred_str}")
        print(f"         true   {true_str}")
        print()


def check_gradients(model):
    """각 파라미터 그룹의 gradient norm 출력 (역전파 확인용)"""
    print("\n[gradient norms]")
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            print(f"  {name:60s}  {norm:.2e}")
        else:
            status = "frozen" if not param.requires_grad else "no grad yet"
            print(f"  {name:60s}  ({status})")


# ── 사용 예시 ──────────────────────────────────────────────────────────────────
#
# from pathlib import Path
# import torch, torch.nn as nn
# from dataset import make_loaders
# from model.model import KnobNet
# from train.train import (
#     make_optimizer, run_epoch, evaluate, evaluate_per_param,
#     save_checkpoint, load_checkpoint,
#     log_epoch, log_param_mae, print_batch, print_predictions, check_gradients,
# )
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# save_path = Path("data/models/knobnet/best.pt")
# save_path.parent.mkdir(parents=True, exist_ok=True)
#
# # ── 데이터 ──
# train_loader, val_loader = make_loaders(
#     dataset_root = "data/dataset",
#     input_dirs   = ["clean-sequences", "flat"],
#     batch_size   = 16,
# )
#
# # 배치 하나 확인
# input_audio, ref_audio, knobs = next(iter(train_loader))
# print_batch(input_audio, ref_audio, knobs)
#
# # ── 모델 ──
# model = KnobNet(num_knobs=3, freeze_mert=True).to(device)
# model.summary()
#
# criterion = nn.L1Loss()
#
# # ── Phase 1: MERT frozen ──
# optimizer = make_optimizer(model, lr=1e-3, phase=1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
#
# best_val = float("inf")
# for epoch in range(1, 31):
#     train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
#     val_loss   = evaluate(model, val_loader, criterion, device)
#     scheduler.step()
#
#     improved = val_loss < best_val
#     if improved:
#         best_val = val_loss
#         save_checkpoint(save_path, model, epoch, phase=1, val_loss=val_loss)
#     log_epoch(1, epoch, 30, train_loss, val_loss, improved)
#     if epoch % 5 == 0:
#         mae = evaluate_per_param(model, val_loader, device)
#         log_param_mae(mae)
#
# print_predictions(model, val_loader, device, n=8)
#
# # ── Phase 2: MERT unfreeze ──
# model.unfreeze_mert()
# model.summary()
# optimizer = make_optimizer(model, lr=1e-3, phase=2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
#
# best_val = float("inf")
# for epoch in range(1, 11):
#     train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
#     val_loss   = evaluate(model, val_loader, criterion, device)
#     scheduler.step()
#
#     improved = val_loss < best_val
#     if improved:
#         best_val = val_loss
#         save_checkpoint(save_path, model, epoch, phase=2, val_loss=val_loss)
#     log_epoch(2, epoch, 10, train_loss, val_loss, improved)
#
# # ── 체크포인트 로드 ──
# meta = load_checkpoint(save_path, model)
# print(meta)  # {"phase": 2, "epoch": 7, "val_loss": 0.043}
