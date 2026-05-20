"""노브 분포 분석 스크립트
Usage: python scripts/plot_knob_dist.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data/wet/black/samples.csv"

df = pd.read_csv(CSV_PATH)[["drive", "level", "filter"]]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# ── 1행: 히스토그램 ───────────────────────────────────────────
for i, col in enumerate(["drive", "level", "filter"]):
    ax = axes[0, i]
    ax.hist(df[col], bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_title(f"{col} distribution")
    ax.set_xlabel("value (0~1)")
    ax.set_ylabel("count")
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"mean={df[col].mean():.2f}")
    ax.legend()

# ── 2행: 2D scatter ───────────────────────────────────────────
pairs = [("level", "drive"), ("level", "filter"), ("drive", "filter")]
for i, (x_col, y_col) in enumerate(pairs):
    ax = axes[1, i]
    c_col = "filter" if i < 2 else "level"
    sc = ax.scatter(df[x_col], df[y_col], alpha=0.05, s=2, c=df[c_col], cmap="viridis")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    plt.colorbar(sc, ax=ax, label=c_col)
    if x_col == "level":
        ax.axvline(0.2, color="red", linestyle="--", linewidth=1, label="level=0.2")
        ax.legend(fontsize=8)

plt.suptitle(f"Knob Distribution Analysis  (N={len(df):,})", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("knob_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 수치 요약 ──────────────────────────────────────────────────
print("=== Overall stats ===")
print(df.describe().round(3))

low_level = df[df["level"] < 0.2]
print(f"\n=== level < 0.2  ({len(low_level):,} samples, {len(low_level)/len(df)*100:.1f}% of total) ===")
print(low_level[["drive", "filter"]].describe().round(3))

print("\n=== Correlation ===")
print(df.corr().round(3))
