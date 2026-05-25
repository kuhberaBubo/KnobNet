"""
KnobNet 평가 GUI
python scripts/eval_gui.py

1. Input WAV 로드
2. 노브 슬라이더로 Reference 생성 (VST 적용)
3. 모델이 노브값 예측 → 예측 노브로 VST 재적용
4. Reference / Model Output 재생 + 멜 스펙트로그램 비교
"""
import json
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SETTINGS_FILE = Path(__file__).parent / "eval_gui_settings.json"


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_settings(model: str, vst: str, wav: str):
    SETTINGS_FILE.write_text(
        json.dumps({"model": model, "vst": vst, "wav": wav}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from pedalboard import load_plugin

from model.model import KnobNet
from utils.config import MERT_SR, CLIP_SAMPLES, KNOB_PARAMS

# ── 상수 ──────────────────────────────────────────────────────────────────────
DEFAULT_VST   = Path("vst/NA Black.vst3")
DEFAULT_MODEL = Path("data/models/knobnet/best.pt")
PARAM_MAX     = 10.0
FIXED_PARAMS  = {"power": True, "bypass": False}


# ── 오디오 유틸 ──────────────────────────────────────────────────────────────

def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """WAV 로드 → mono float32"""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def to_model_tensors(audio: np.ndarray, sr: int) -> torch.Tensor:
    """모델 입력: 24kHz 리샘플 후 5초 청크 분할.
    5초 미만이면 패딩 1개, 초과하면 5초마다 잘라 여러 개 반환 → (N, CLIP_SAMPLES)"""
    if sr != MERT_SR:
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(t, sr, MERT_SR).squeeze(0).numpy()

    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        return torch.from_numpy(audio.copy()).unsqueeze(0)  # (1, CLIP_SAMPLES)

    chunks = []
    for start in range(0, len(audio), CLIP_SAMPLES):
        chunk = audio[start: start + CLIP_SAMPLES]
        if len(chunk) < CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
        chunks.append(torch.from_numpy(chunk.copy()))
    return torch.stack(chunks)  # (N, CLIP_SAMPLES)


def run_vst(plugin, audio: np.ndarray, sr: int, knobs_raw: dict) -> np.ndarray:
    """VST 적용. knobs_raw: {"drive": 5.0, "level": 3.0, "tone": 7.0}
    tone → filter 변환: filter = PARAM_MAX - tone  (filter = 1 - tone, normalized 기준)
    """
    vst_params = {
        ("filter" if k == "tone" else k): (PARAM_MAX - v if k == "tone" else v)
        for k, v in knobs_raw.items()
    }
    for name, val in FIXED_PARAMS.items():
        try:
            setattr(plugin, name, val)
        except Exception:
            pass
    for name, val in vst_params.items():
        try:
            setattr(plugin, name, val)
        except (ValueError, TypeError):
            setattr(plugin, name, bool(val > 0.5))
    out = plugin.process(audio, sr)
    if out.ndim > 1:
        out = out.mean(axis=1)
    return out


def compute_mel_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """멜 스펙트로그램 (dB) → (n_mels, T)"""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t = torch.from_numpy(audio).unsqueeze(0)
    if sr != MERT_SR:
        t = torchaudio.functional.resample(t, sr, MERT_SR)
        sr = MERT_SR
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
    )(t).squeeze(0)
    return torchaudio.transforms.AmplitudeToDB(top_db=80)(mel).numpy()


def _load_model(path: Path) -> KnobNet:
    """model.export()로 저장한 파일 로드"""
    return KnobNet.from_exported(path, device=torch.device("cpu"))


# ── GUI ──────────────────────────────────────────────────────────────────────

class EvalApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KnobNet Evaluator")
        self.root.minsize(700, 600)

        self.model: KnobNet | None = None
        self.plugin = None
        self.input_audio: np.ndarray | None = None
        self.input_sr: int | None = None
        self.output_ref: np.ndarray | None = None
        self.output_model_audio: np.ndarray | None = None

        self._build()

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _build(self):
        self._build_files()
        self._build_knobs()
        self._build_controls()
        self._build_pred()
        self._build_play()
        self._build_spec()

    def _build_files(self):
        f = ttk.LabelFrame(self.root, text="Files")
        f.pack(fill="x", padx=10, pady=5)
        f.columnconfigure(1, weight=1)

        s = _load_settings()
        self.model_var = self._file_row(f, 0, "Model (.pt)",  s.get("model") or str(DEFAULT_MODEL), "PyTorch", "*.pt")
        self.vst_var   = self._file_row(f, 1, "VST (.vst3)",  s.get("vst")   or str(DEFAULT_VST),   "VST3",    "*.vst3")
        self.wav_var   = self._file_row(f, 2, "Input WAV",    s.get("wav")   or "",                  "WAV",     "*.wav")
        ttk.Button(f, text="Load All", command=self._load_all).grid(
            row=3, column=0, columnspan=3, pady=5)

    def _file_row(self, parent, row, label, default, ftlabel, ftpat) -> tk.StringVar:
        var = tk.StringVar(value=str(default) if default else "")
        ttk.Label(parent, text=label, width=14, anchor="w").grid(
            row=row, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(parent, textvariable=var, width=55).grid(
            row=row, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(parent, text="Browse",
                   command=lambda v=var, ft=(ftlabel, ftpat): self._browse(v, ft)
                   ).grid(row=row, column=2, padx=5)
        return var

    def _browse(self, var: tk.StringVar, ft: tuple):
        p = filedialog.askopenfilename(filetypes=[(ft[0], ft[1]), ("All", "*")])
        if p:
            var.set(p)

    def _build_knobs(self):
        f = ttk.LabelFrame(self.root, text="Reference Knob Settings  (0 – 10)")
        f.pack(fill="x", padx=10, pady=5)
        self.knob_vars: dict[str, tk.DoubleVar] = {}

        for col, name in enumerate(KNOB_PARAMS):
            f.columnconfigure(col, weight=1)
            cell = ttk.Frame(f)
            cell.grid(row=0, column=col, padx=15, pady=6, sticky="ew")

            ttk.Label(cell, text=name.capitalize(), font=("", 10, "bold")).pack()
            var = tk.DoubleVar(value=5.0)
            self.knob_vars[name] = var
            val_lbl = ttk.Label(cell, text="5.000")
            val_lbl.pack()

            def make_cb(v, lbl):
                def cb(_): lbl.config(text=f"{v.get():.3f}")
                return cb

            ttk.Scale(cell, from_=0, to=PARAM_MAX, variable=var,
                      orient="horizontal", length=180,
                      command=make_cb(var, val_lbl)).pack()

    def _build_controls(self):
        f = ttk.Frame(self.root)
        f.pack(fill="x", padx=10, pady=4)
        ttk.Button(f, text="Apply & Predict", command=self._run).pack(side="left")
        self.status_var = tk.StringVar(value="Load All을 눌러 파일을 로드하세요.")
        ttk.Label(f, textvariable=self.status_var).pack(side="left", padx=10)

    def _build_pred(self):
        f = ttk.LabelFrame(self.root, text="Model Prediction  (normalized 0–1)")
        f.pack(fill="x", padx=10, pady=5)
        self.pred_main_lbls: dict[str, ttk.Label] = {}
        self.pred_diff_lbls: dict[str, ttk.Label] = {}

        for col, name in enumerate(KNOB_PARAMS):
            f.columnconfigure(col, weight=1)
            cell = ttk.Frame(f)
            cell.grid(row=0, column=col, padx=20, pady=6)
            ttk.Label(cell, text=name.capitalize()).pack()
            main_lbl = ttk.Label(cell, text="—", font=("", 13, "bold"))
            main_lbl.pack()
            diff_lbl = ttk.Label(cell, text="", foreground="gray")
            diff_lbl.pack()
            self.pred_main_lbls[name] = main_lbl
            self.pred_diff_lbls[name] = diff_lbl

    def _build_play(self):
        f = ttk.Frame(self.root)
        f.pack(fill="x", padx=10, pady=4)
        ttk.Button(f, text="▶ Reference (User)",  command=self._play_ref).pack(side="left", padx=5)
        ttk.Button(f, text="▶ Model Output",       command=self._play_model).pack(side="left", padx=5)
        ttk.Button(f, text="■ Stop",               command=sd.stop).pack(side="left", padx=5)

    def _build_spec(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 3))
        self.fig.tight_layout(pad=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        for ax, title in zip(self.axes, ["Reference (User Knobs)", "Model Output"]):
            ax.set_title(title, fontsize=10)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()

    # ── 파일 로드 ────────────────────────────────────────────────────────────

    def _load_all(self):
        errors = []

        mp = Path(self.model_var.get())
        if mp.exists():
            try:
                self.model = _load_model(mp)
                self.model.eval()
            except Exception as e:
                errors.append(f"Model: {e}")
        else:
            errors.append(f"Model 파일 없음: {mp}")

        vp = Path(self.vst_var.get())
        if vp.exists():
            try:
                self.plugin = load_plugin(str(vp))
            except Exception as e:
                errors.append(f"VST: {e}")
        else:
            errors.append(f"VST 파일 없음: {vp}")

        wp = self.wav_var.get()
        if wp and Path(wp).exists():
            try:
                self.input_audio, self.input_sr = load_wav(Path(wp))
            except Exception as e:
                errors.append(f"WAV: {e}")
        elif not wp:
            pass  # wav는 선택사항 (나중에 Apply 전에 체크)

        if errors:
            messagebox.showerror("Load Error", "\n".join(errors))
        else:
            _save_settings(self.model_var.get(), self.vst_var.get(), self.wav_var.get())
            loaded = ["Model", "VST"] + (["WAV"] if wp else [])
            self.status_var.set(f"로드 완료: {', '.join(loaded)}")

    # ── 메인 처리 ────────────────────────────────────────────────────────────

    def _run(self):
        # wav는 Apply 시점에 로드 허용
        wp = self.wav_var.get()
        if wp and Path(wp).exists() and self.input_audio is None:
            try:
                self.input_audio, self.input_sr = load_wav(Path(wp))
            except Exception as e:
                messagebox.showerror("WAV Load Error", str(e)); return

        if self.model is None:
            messagebox.showwarning("", "Model을 먼저 로드하세요."); return
        if self.plugin is None:
            messagebox.showwarning("", "VST를 먼저 로드하세요."); return
        if self.input_audio is None:
            messagebox.showwarning("", "Input WAV를 먼저 로드하세요."); return

        self.status_var.set("처리 중…")
        self.root.update()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            knobs_raw = {n: v.get() for n, v in self.knob_vars.items()}

            # Step 1: 유저 노브로 VST 적용 → reference
            self.output_ref = run_vst(self.plugin, self.input_audio, self.input_sr, knobs_raw)

            # Step 2: 모델 추론 (input, reference) → predicted knobs [0, 1]
            # 5초 청크로 분할 후 평균
            inp_chunks = to_model_tensors(self.input_audio, self.input_sr)  # (N, CLIP_SAMPLES)
            ref_chunks = to_model_tensors(self.output_ref,  self.input_sr)
            n = min(len(inp_chunks), len(ref_chunks))
            with torch.no_grad():
                all_preds = torch.stack([
                    self.model(inp_chunks[i].unsqueeze(0), ref_chunks[i].unsqueeze(0))[0]
                    for i in range(n)
                ])  # (N, num_knobs)
            preds = all_preds.median(dim=0).values.tolist()

            # ── DEBUG: vol 정보 출력 (MERT 재호출 없이 직접 계산) ──────────
            try:
                inp0 = inp_chunks[0]   # (CLIP_SAMPLES,)
                ref0 = ref_chunks[0]
                vol_input_dbg = float(inp0.pow(2).mean().sqrt())
                vol_ref_dbg   = float(ref0.pow(2).mean().sqrt())
                wet_rms_raw   = float(np.sqrt(np.mean(self.output_ref ** 2)))
                print(f"[DEBUG] wet RMS (raw, before resample) = {wet_rms_raw:.6f}")
                print(f"[DEBUG] vol_input = {vol_input_dbg:.6f}")
                print(f"[DEBUG] vol_ref   = {vol_ref_dbg:.6f}")
                print(f"[DEBUG] preds     = {[f'{v:.4f}' for v in preds]}")
                knobs_norm = {n: knobs_raw[n] / PARAM_MAX for n in KNOB_PARAMS}
                print(f"[DEBUG] true(norm)= {[f'{knobs_norm[n]:.4f}' for n in KNOB_PARAMS]}")
            except Exception as dbg_e:
                print(f"[DEBUG] error: {dbg_e}")
            # ────────────────────────────────────────────────────────────────

            # Step 3: 예측 노브로 VST 재적용 → model output
            pred_raw = {n: preds[i] * PARAM_MAX for i, n in enumerate(KNOB_PARAMS)}
            self.output_model_audio = run_vst(self.plugin, self.input_audio, self.input_sr, pred_raw)

            self.root.after(0, lambda: self._update(preds, knobs_raw))

        except Exception:
            err = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", err))

    def _update(self, preds: list, knobs_raw: dict):
        # 예측값 표시
        for i, name in enumerate(KNOB_PARAMS):
            pred_norm = preds[i]
            user_norm = knobs_raw[name] / PARAM_MAX
            diff = pred_norm - user_norm
            self.pred_main_lbls[name].config(text=f"{pred_norm:.3f}")
            color = "green" if abs(diff) <= 0.1 else "red"
            self.pred_diff_lbls[name].config(
                text=f"user {user_norm:.3f}  Δ{diff:+.3f}", foreground=color)

        # 멜 스펙트로그램
        for idx, (audio, title) in enumerate([
            (self.output_ref,         "Reference (User Knobs)"),
            (self.output_model_audio, "Model Output"),
        ]):
            ax = self.axes[idx]
            ax.clear()
            ax.set_title(title, fontsize=10)
            mel = compute_mel_db(audio, self.input_sr)
            duration = len(audio) / self.input_sr
            ax.imshow(mel, aspect="auto", origin="lower",
                      extent=[0, duration, 0, mel.shape[0]],
                      cmap="magma", vmin=-80, vmax=0)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mel bin")

        self.fig.tight_layout(pad=2)
        self.canvas.draw()
        self.status_var.set("완료")

    # ── 재생 ─────────────────────────────────────────────────────────────────

    def _play_ref(self):
        if self.output_ref is not None:
            sd.stop(); sd.play(self.output_ref, self.input_sr)

    def _play_model(self):
        if self.output_model_audio is not None:
            sd.stop(); sd.play(self.output_model_audio, self.input_sr)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    EvalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
