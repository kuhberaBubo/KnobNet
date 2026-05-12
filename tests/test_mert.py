"""
MERT 로컬 CPU 탐색 스크립트
python test_mert.py [audio_path]
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_SR = 24000
CLIP_SEC = 5.0


def separator(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── 1. 모델 로드 ──────────────────────────────────────
separator("1. 모델 & Feature Extractor 로드")
print(f"model_id: {MODEL_ID}")

processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()

print(f"hidden_size      : {model.config.hidden_size}")
print(f"num_hidden_layers: {model.config.num_hidden_layers}")
print(f"processor sr     : {processor.sampling_rate}")
print(f"do_normalize     : {processor.do_normalize}")
print(f"파라미터 수      : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ── 2. 오디오 로드 ────────────────────────────────────
separator("2. 오디오 로드")

audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.wav")
audio, sr = sf.read(audio_path, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

print(f"파일    : {audio_path}")
print(f"원본 sr : {sr} Hz, 샘플 수: {len(audio)}, 길이: {len(audio)/sr:.2f}s")
print(f"값 범위 : [{audio.min():.4f}, {audio.max():.4f}]  std={audio.std():.4f}")

if sr != MERT_SR:
    t = torch.from_numpy(audio).unsqueeze(0)
    audio = torchaudio.functional.resample(t, sr, MERT_SR).squeeze(0).numpy()
    print(f"리샘플  : {sr} → {MERT_SR} Hz, 샘플 수: {len(audio)}")

# ── 3. Feature Extractor ──────────────────────────────
separator("3. Feature Extractor")

inputs = processor(audio, sampling_rate=MERT_SR, return_tensors="pt")
iv = inputs["input_values"]

print(f"input_values shape: {iv.shape}   (batch=1, time)")
print(f"정규화 후 mean    : {iv.mean().item():.6f}")
print(f"정규화 후 std     : {iv.std().item():.6f}")

# ── 4. Forward → output 구조 ─────────────────────────
separator("4. MERT Forward — output 구조")

with torch.no_grad():
    out = model(input_values=iv, output_hidden_states=True)

lhs = out.last_hidden_state
print(f"last_hidden_state : {lhs.shape}   (batch, time_frames, hidden)")

T_in = iv.shape[-1]
T_out = lhs.shape[1]
print(f"\n다운샘플 비율     : {T_in} → {T_out}  ({T_in/T_out:.1f}x)")
print(f"프레임 주기       : {T_in / T_out / MERT_SR * 1000:.1f} ms/frame")

print(f"\nhidden_states 레이어 수: {len(out.hidden_states)}  (0=CNN embed, 1-12=transformer)")
for i, hs in enumerate(out.hidden_states):
    label = "CNN embed" if i == 0 else f"layer {i:2d}"
    print(f"  {label}: {hs.shape}")

# ── 5. Embedding 방식 비교 ────────────────────────────
separator("5. Head 입력 embedding")

emb_last = lhs.mean(dim=1)
print(f"A) last layer mean-pool  : {emb_last.shape}")

all_layers = torch.stack(out.hidden_states[1:], dim=0)   # (12, 1, T, 768)
weights = torch.softmax(torch.ones(12), dim=0)
emb_weighted = (all_layers * weights[:, None, None, None]).sum(0).mean(1)
print(f"B) 12-layer weighted sum : {emb_weighted.shape}")

cosine_sim = torch.nn.functional.cosine_similarity(emb_last, emb_weighted).item()
print(f"   A↔B cosine similarity : {cosine_sim:.4f}")

# ── 6. 배치 처리 ──────────────────────────────────────
separator("6. 배치 처리 (2개 클립)")

clip_samples = int(MERT_SR * CLIP_SEC)

def to_clip(a):
    if len(a) >= clip_samples:
        return a[:clip_samples]
    return np.pad(a, (0, clip_samples - len(a)))

clip1 = to_clip(audio)
clip2 = to_clip(audio[:clip_samples // 2])   # 짧은 거 패딩

batch = processor([clip1, clip2], sampling_rate=MERT_SR, return_tensors="pt")
print(f"배치 input_values : {batch['input_values'].shape}")

with torch.no_grad():
    bout = model(input_values=batch["input_values"])

print(f"배치 last_hidden  : {bout.last_hidden_state.shape}")
print(f"배치 embedding    : {bout.last_hidden_state.mean(dim=1).shape}")

# ── 7. 속도 측정 ──────────────────────────────────────
separator("7. CPU 추론 속도")

x = batch["input_values"]
with torch.no_grad():
    _ = model(input_values=x)   # warmup

N = 3
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        _ = model(input_values=x)
elapsed = (time.time() - t0) / N

print(f"배치 2 × {CLIP_SEC}s  →  {elapsed*1000:.0f} ms/iter  (CPU)")

print(f"\n{'─' * 50}")
print("  완료")
print(f"{'─' * 50}\n")
