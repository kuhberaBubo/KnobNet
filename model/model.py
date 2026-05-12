import torch
import torch.nn as nn
from transformers import AutoModel

from utils.config import MERT_MODEL_ID, MERT_SR, KNOB_PARAMS


class KnobNet(nn.Module):

    def __init__(self, num_knobs: int = 3, mert_model_id: str = MERT_MODEL_ID, freeze_mert: bool = True):
        super().__init__()
        self.num_knobs = num_knobs

        self.mert = AutoModel.from_pretrained(mert_model_id, trust_remote_code=True)

        hidden = self.mert.config.hidden_size       # 768
        head_in = hidden * 2 + 2                    # I_feat + O_feat + vol_input + vol_ref

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_knobs),
            nn.Sigmoid(),
        )

        if freeze_mert:
            self.freeze_mert()

    # ── 내부 유틸 ────────────────────────────────────────────────────────────────

    def _rms(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T) → (B, 1)
        return audio.pow(2).mean(dim=-1, keepdim=True).sqrt()

    def _normalize(self, audio: torch.Tensor) -> torch.Tensor:
        # MERT processor와 동일: zero-mean / unit-variance per sample
        mean = audio.mean(dim=-1, keepdim=True)
        std = audio.std(dim=-1, keepdim=True)
        return (audio - mean) / (std + 1e-7)

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T) normalized → (B, hidden)
        out = self.mert(input_values=audio, output_hidden_states=False)
        return out.last_hidden_state.mean(dim=1)

    # ── forward ──────────────────────────────────────────────────────────────────

    def forward(self, input_audio: torch.Tensor, reference_audio: torch.Tensor) -> torch.Tensor:
        # input_audio, reference_audio: (B, T) raw audio at MERT_SR
        # returns: (B, num_knobs) ∈ [0, 1]

        vol_input = self._rms(input_audio)       # (B, 1) — normalize 전에 뽑음
        vol_ref   = self._rms(reference_audio)   # (B, 1)

        I_feat = self._encode(self._normalize(input_audio))       # (B, 768)
        O_feat = self._encode(self._normalize(reference_audio))   # (B, 768)

        x = torch.cat([vol_input, I_feat, vol_ref, O_feat], dim=-1)  # (B, 1538)
        return self.head(x)                                            # (B, num_knobs)

    # ── phase 전환 ───────────────────────────────────────────────────────────────

    def freeze_mert(self):
        for param in self.mert.parameters():
            param.requires_grad = False

    def unfreeze_mert(self):
        for param in self.mert.parameters():
            param.requires_grad = True

    # ── optimizer param group 분리용 ─────────────────────────────────────────────

    def mert_parameters(self):
        return self.mert.parameters()

    def head_parameters(self):
        return self.head.parameters()

    # ── 디버그 ───────────────────────────────────────────────────────────────────

    def extract_features(self, input_audio: torch.Tensor, reference_audio: torch.Tensor) -> dict:
        """중간 representation 반환 — forward 흐름 검사용"""
        with torch.no_grad():
            vol_input = self._rms(input_audio)
            vol_ref   = self._rms(reference_audio)
            I_feat    = self._encode(self._normalize(input_audio))
            O_feat    = self._encode(self._normalize(reference_audio))
            head_in   = torch.cat([vol_input, I_feat, vol_ref, O_feat], dim=-1)
        return {
            "vol_input":  vol_input,    # (B, 1)
            "vol_ref":    vol_ref,      # (B, 1)
            "I_feat":     I_feat,       # (B, 768)
            "O_feat":     O_feat,       # (B, 768)
            "head_input": head_in,      # (B, 1538)
        }

    def count_parameters(self) -> dict:
        """파라미터 수 집계"""
        mert_total     = sum(p.numel() for p in self.mert.parameters())
        mert_trainable = sum(p.numel() for p in self.mert.parameters() if p.requires_grad)
        head_total     = sum(p.numel() for p in self.head.parameters())
        return {
            "mert_total":      mert_total,
            "mert_trainable":  mert_trainable,
            "head_total":      head_total,
            "total_trainable": mert_trainable + head_total,
        }

    def summary(self):
        """모델 상태 한눈에 출력"""
        c = self.count_parameters()
        mert_frozen = not any(p.requires_grad for p in self.mert.parameters())
        print(f"KnobNet  (num_knobs={self.num_knobs})")
        print(f"  MERT : {'frozen    ' if mert_frozen else 'trainable'}  "
              f"{c['mert_trainable']:>10,} / {c['mert_total']:,}")
        print(f"  Head :             {c['head_total']:>10,}")
        print(f"  Total trainable  : {c['total_trainable']:>10,}")
