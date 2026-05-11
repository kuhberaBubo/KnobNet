"""
VST3 플러그인을 사용해서 wet 데이터를 생성하는 프로그램

사용법:
    python vst_sample.py info <vst_path>
    python vst_sample.py sample <vst_path>
    python vst_sample.py calibrate <vst_path> [--input-file ...]
"""

import argparse
import csv
import random
from pathlib import Path

# ── sample 옵션 ───────────────────────────────────────────────────────────────
VST_PATH = Path("vst/NA Black.vst3")
SAMPLE_INPUT_DIRS = [Path("data/clean-sequences"), Path("data/flat")]
SAMPLE_OUTPUT_DIR = Path("data/wet/black")
SAMPLE_STRATEGY = "uniform"   # "uniform" or "grid"
SAMPLE_SEED = 42

import numpy as np
import sounddevice as sd
import soundfile as sf
from pedalboard import load_plugin


# ── 공통 ──────────────────────────────────────────────────────────────────────

def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return audio * (10 ** (gain_db / 20))


def process_audio(plugin, audio: np.ndarray, sr: int, param_config: dict, input_gain_db: float) -> np.ndarray:
    for name, value in param_config.items():
        try:
            setattr(plugin, name, value)
        except (ValueError, TypeError):
            setattr(plugin, name, bool(value > 0.5))
    return plugin.process(apply_gain(audio, input_gain_db), sr)


# ── info ──────────────────────────────────────────────────────────────────────

def cmd_info(args):
    plugin = load_plugin(args.vst_path)
    print(f"\nPlugin : {plugin.name}")
    print(f"Params : {len(plugin.parameters)}\n")
    for name, param in plugin.parameters.items():
        print(f"  {name}")
        print(f"    range  : {param.min_value:.4f} ~ {param.max_value:.4f}")
        print(f"    default: {param.default_raw_value:.4f}")
        print()


# ── sample ────────────────────────────────────────────────────────────────────

# 샘플링할 파라미터 (나머지는 FIXED_PARAMS 값으로 고정)
SAMPLE_PARAMS = ["gain", "level", "filter"]
FIXED_PARAMS = {"power": True, "bypass": False}
N_PER_FILE = 10


def _sample_uniform(parameters: dict, n: int, rng: random.Random) -> list:
    return [
        {name: rng.uniform(p.min_value, p.max_value) for name, p in parameters.items()}
        for _ in range(n)
    ]


def _sample_grid(parameters: dict, n: int, rng: random.Random) -> list:
    steps = max(2, round(n ** (1 / max(len(parameters), 1))))
    grids = {
        name: np.linspace(p.min_value, p.max_value, steps).tolist()
        for name, p in parameters.items()
    }
    return [
        {name: rng.choice(values) for name, values in grids.items()}
        for _ in range(n)
    ]


def cmd_sample(args):
    rng = random.Random(args.seed)
    plugin = load_plugin(args.vst_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 샘플링 대상 파라미터만 추출
    sample_parameters = {
        name: param for name, param in plugin.parameters.items()
        if name in SAMPLE_PARAMS
    }

    csv_path = output_dir / "samples.csv"
    csv_exists = csv_path.exists()
    sample_counter = sum(1 for _ in output_dir.glob("sample_*.wav"))

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input_file", "output_file"] + SAMPLE_PARAMS)
        if not csv_exists:
            writer.writeheader()

        for input_dir in [Path(d) for d in args.input_dirs]:
            wav_files = sorted(input_dir.rglob("*.wav"))
            if not wav_files:
                print(f"No wav files found in {input_dir}")
                continue

            if args.strategy == "uniform":
                configs = _sample_uniform(sample_parameters, args.n_per_file, rng)
            else:
                configs = _sample_grid(sample_parameters, args.n_per_file, rng)

            for wav_file in wav_files:
                audio, sr = sf.read(wav_file)

                # 고정 파라미터 적용
                for name, value in FIXED_PARAMS.items():
                    try:
                        setattr(plugin, name, value)
                    except (ValueError, TypeError, AttributeError):
                        pass

                for config in configs:
                    sample_counter += 1
                    output_name = f"sample_{sample_counter:05d}.wav"
                    wet = process_audio(plugin, audio, sr, config, input_gain_db=0.0)
                    sf.write(output_dir / output_name, wet, sr)

                    writer.writerow({
                        "input_file": str(wav_file),
                        "output_file": output_name,
                        **config,
                    })
                    print(f"[{sample_counter}] {wav_file.name} -> {output_name}")

    print(f"\nDone. CSV saved: {csv_path}")


# ── calibrate ─────────────────────────────────────────────────────────────────

def cmd_calibrate(args):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    plugin = load_plugin(args.vst_path)

    audio_data = {"audio": None, "sr": None}
    debounce_id = [None]

    if args.input_file:
        audio, sr = sf.read(args.input_file)
        audio_data["audio"] = audio
        audio_data["sr"] = sr

    root = tk.Tk()
    root.title(f"Calibrate — {plugin.name}")
    root.resizable(True, True)

    # ── 파일 로드 ──
    file_frame = ttk.LabelFrame(root, text="Input File")
    file_frame.pack(fill="x", padx=10, pady=5)

    file_label = ttk.Label(file_frame, text=args.input_file or "No file loaded")
    file_label.pack(side="left", padx=5, pady=3)

    def load_file():
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            audio, sr = sf.read(path)
            audio_data["audio"] = audio
            audio_data["sr"] = sr
            file_label.config(text=Path(path).name)

    ttk.Button(file_frame, text="Browse", command=load_file).pack(side="right", padx=5)

    # ── Input Gain ──
    gain_frame = ttk.LabelFrame(root, text="Input Gain (dB)")
    gain_frame.pack(fill="x", padx=10, pady=5)

    gain_var = tk.DoubleVar(value=0.0)
    gain_val_label = ttk.Label(gain_frame, text=" 0.0 dB", width=8)
    gain_val_label.pack(side="right", padx=5)

    def on_gain_change(_):
        gain_val_label.config(text=f"{gain_var.get():+.1f} dB")
        if realtime_var.get():
            schedule_play()

    ttk.Scale(gain_frame, from_=-60, to=24, variable=gain_var,
              orient="horizontal", command=on_gain_change).pack(fill="x", padx=5, pady=5)

    # ── VST 파라미터 슬라이더 (스크롤 가능) ──
    params_outer = ttk.LabelFrame(root, text="VST Parameters")
    params_outer.pack(fill="both", expand=True, padx=10, pady=5)

    canvas = tk.Canvas(params_outer, height=300)
    scrollbar = ttk.Scrollbar(params_outer, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    param_vars = {}
    for name, param in plugin.parameters.items():
        row = ttk.Frame(inner)
        row.pack(fill="x", padx=5, pady=2)

        ttk.Label(row, text=name, width=22, anchor="w").pack(side="left")

        var = tk.DoubleVar(value=param.default_raw_value)
        param_vars[name] = var

        val_lbl = ttk.Label(row, text=f"{param.default_raw_value:.3f}", width=8)
        val_lbl.pack(side="right")

        def make_cb(v, lbl):
            def cb(_):
                lbl.config(text=f"{v.get():.3f}")
                if realtime_var.get():
                    schedule_play()
            return cb

        ttk.Scale(row, from_=param.min_value, to=param.max_value, variable=var,
                  orient="horizontal", command=make_cb(var, val_lbl)
                  ).pack(side="left", fill="x", expand=True, padx=5)

    # ── 컨트롤 바 ──
    ctrl_frame = ttk.Frame(root)
    ctrl_frame.pack(fill="x", padx=10, pady=8)

    realtime_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(ctrl_frame, text="Real-time", variable=realtime_var).pack(side="left")

    status_label = ttk.Label(ctrl_frame, text="")
    status_label.pack(side="left", padx=10)

    def get_wet():
        if audio_data["audio"] is None:
            messagebox.showwarning("No file", "먼저 입력 wav 파일을 로드하세요.")
            return None, None
        config = {name: var.get() for name, var in param_vars.items()}
        wet = process_audio(plugin, audio_data["audio"], audio_data["sr"], config, gain_var.get())
        return wet, audio_data["sr"]

    def play():
        wet, sr = get_wet()
        if wet is None:
            return
        status_label.config(text="▶ Playing...")
        sd.stop()
        sd.play(wet, sr)
        duration_ms = int(len(wet) / sr * 1000) + 300
        root.after(duration_ms, lambda: status_label.config(text=""))

    def schedule_play():
        if debounce_id[0] is not None:
            root.after_cancel(debounce_id[0])
        debounce_id[0] = root.after(250, play)

    def print_values():
        print(f"\n[Calibrate] Input gain : {gain_var.get():+.1f} dB")
        for name, var in param_vars.items():
            print(f"  {name}: {var.get():.4f}")
        print()

    ttk.Button(ctrl_frame, text="■ Stop", command=sd.stop).pack(side="right", padx=5)
    ttk.Button(ctrl_frame, text="▶ Play", command=play).pack(side="right", padx=5)
    ttk.Button(ctrl_frame, text="Print Values", command=print_values).pack(side="right", padx=5)

    root.mainloop()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VST3 sampler / calibrator")
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p = sub.add_parser("info", help="VST 파라미터 목록 출력")
    p.add_argument("vst_path")
    p.set_defaults(func=cmd_info)

    # sample
    p = sub.add_parser("sample", help="Wet 샘플 생성 + CSV 저장")
    p.add_argument("vst_path", nargs="?", default=str(VST_PATH))
    p.add_argument("--input-dirs", nargs="+", default=[str(d) for d in SAMPLE_INPUT_DIRS], metavar="DIR")
    p.add_argument("--output-dir", default=str(SAMPLE_OUTPUT_DIR))
    p.add_argument("--strategy", choices=["uniform", "grid"], default=SAMPLE_STRATEGY)
    p.add_argument("--n-per-file", type=int, default=N_PER_FILE)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.set_defaults(func=cmd_sample)

    # calibrate
    p = sub.add_parser("calibrate", help="입력 볼륨 인터랙티브 캘리브레이션")
    p.add_argument("vst_path")
    p.add_argument("--input-file", default=None)
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
