"""
Phase 2 checkpoint → export 형식 변환 (MERT 초기화 없음)
python scripts/export_checkpoint.py <checkpoint.pt> <output.pt>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.config import KNOB_PARAMS, MERT_MODEL_ID


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/export_checkpoint.py <checkpoint.pt> <output.pt>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    print(f"Loading: {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    print(f"  phase={ckpt.get('phase')}  epoch={ckpt.get('epoch')}  val_loss={ckpt.get('val_loss', '?'):.4f}")

    export = {
        "model_state":   ckpt["model_state"],
        "num_knobs":     len(KNOB_PARAMS),
        "mert_model_id": MERT_MODEL_ID,
        "knob_params":   KNOB_PARAMS,
        "layer_idx":     ckpt.get("layer_idx", -1),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, dst)
    print(f"Saved : {dst}")


if __name__ == "__main__":
    main()
