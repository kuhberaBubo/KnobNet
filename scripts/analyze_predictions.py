"""
predictions.csv 분석 - 구간별 MAE, 정답률, shortcut 분석

Usage:
    python scripts/analyze_predictions.py predictions.csv
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

TOLERANCE = 0.1
PARAMS = ["drive", "level", "filter"]


def mae(df, param):
    return (df[f"pred_{param}"] - df[f"true_{param}"]).abs().mean()


def acc(df, param):
    return ((df[f"pred_{param}"] - df[f"true_{param}"]).abs() <= TOLERANCE).mean()


def acc_all(df):
    within = all(
        (df[f"pred_{p}"] - df[f"true_{p}"]).abs() <= TOLERANCE
        for p in PARAMS
    )
    return within.mean()


def print_stats(label, df):
    mae_vals = {p: mae(df, p) for p in PARAMS}
    acc_vals = {p: acc(df, p) for p in PARAMS}
    all_acc  = pd.Series(
        [(df[f"pred_{p}"] - df[f"true_{p}"]).abs() <= TOLERANCE for p in PARAMS]
    ).all().mean() if False else (
        ((df[f"pred_drive"]  - df[f"true_drive"]).abs()  <= TOLERANCE) &
        ((df[f"pred_level"]  - df[f"true_level"]).abs()  <= TOLERANCE) &
        ((df[f"pred_filter"] - df[f"true_filter"]).abs() <= TOLERANCE)
    ).mean()

    print(f"\n{'─'*55}")
    print(f"  {label}  (n={len(df):,})")
    print(f"{'─'*55}")
    print(f"  MAE   drive={mae_vals['drive']:.4f}  level={mae_vals['level']:.4f}  filter={mae_vals['filter']:.4f}")
    print(f"  Acc   drive={acc_vals['drive']*100:.1f}%  level={acc_vals['level']*100:.1f}%  filter={acc_vals['filter']*100:.1f}%  all={all_acc*100:.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_predictions.py predictions.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print(f"Loaded {len(df):,} rows")

    # ── 1. 전체 통계 ──────────────────────────────────────────────
    print_stats("전체", df)

    # ── 2. level 구간별 통계 ──────────────────────────────────────
    bins   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0~0.2", "0.2~0.4", "0.4~0.6", "0.6~0.8", "0.8~1.0"]
    df["level_bin"] = pd.cut(df["true_level"], bins=bins, labels=labels, include_lowest=True)

    print(f"\n{'═'*55}")
    print("  [level 구간별]")
    for lbl in labels:
        sub = df[df["level_bin"] == lbl]
        print_stats(f"level {lbl}", sub)

    # ── 3. drive 구간별 통계 ──────────────────────────────────────
    df["drive_bin"] = pd.cut(df["true_drive"], bins=bins, labels=labels, include_lowest=True)

    print(f"\n{'═'*55}")
    print("  [drive 구간별]")
    for lbl in labels:
        sub = df[df["drive_bin"] == lbl]
        print_stats(f"drive {lbl}", sub)

    # ── 4. shortcut 분석: level 낮을 때 drive 예측 편향 ──────────
    print(f"\n{'═'*55}")
    print("  [shortcut 분석: low level 구간에서 pred_drive 분포]")
    print(f"{'─'*55}")
    low  = df[df["true_level"] < 0.2]
    high = df[df["true_level"] > 0.8]
    print(f"  low level  구간  pred_drive 평균={low['pred_drive'].mean():.4f}   true_drive 평균={low['true_drive'].mean():.4f}")
    print(f"  high level 구간  pred_drive 평균={high['pred_drive'].mean():.4f}  true_drive 평균={high['true_drive'].mean():.4f}")

    corr = df["true_level"].corr(df["pred_drive"] - df["true_drive"])
    print(f"\n  level vs drive_error 상관계수: {corr:.4f}")
    print(f"  (양수: level 높을수록 drive 과추정 / 음수: level 낮을수록 drive 과추정)")

    # ── 5. low level + high drive (가장 어려운 케이스) ────────────
    print(f"\n{'═'*55}")
    print("  [low level + high drive (가장 어려운 케이스)]")
    hard = df[(df["true_level"] < 0.2) & (df["true_drive"] > 0.8)]
    print_stats(f"level<0.2 & drive>0.8", hard)

    # ── 6. level < 0.1 세부 구간 ─────────────────────────────────
    print(f"\n{'═'*55}")
    print("  [level < 0.1 세부 구간]")
    fine_bins   = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    fine_labels = ["0.00~0.02", "0.02~0.04", "0.04~0.06", "0.06~0.08", "0.08~0.10"]
    df["level_fine"] = pd.cut(df["true_level"], bins=fine_bins, labels=fine_labels, include_lowest=True)
    for lbl in fine_labels:
        sub = df[df["level_fine"] == lbl]
        print_stats(f"level {lbl}", sub)

    # ── 7. level < 0.1 × drive 구간 교차 ────────────────────────
    print(f"\n{'═'*55}")
    print("  [level < 0.1 × drive 구간 교차]")
    low01 = df[df["true_level"] < 0.1]
    for dlbl in labels:
        sub = low01[low01["drive_bin"] == dlbl]
        if len(sub) > 0:
            print_stats(f"level<0.1 & drive {dlbl}", sub)

    # ── 8. level < 0.1 pred_level 분포 ───────────────────────────
    print(f"\n{'═'*55}")
    print("  [level < 0.1 에서 pred_level 분포]")
    print(f"{'─'*55}")
    low01 = df[df["true_level"] < 0.1]
    print(f"  n={len(low01):,}")
    print(f"  pred_level  mean={low01['pred_level'].mean():.4f}  "
          f"median={low01['pred_level'].median():.4f}  "
          f"std={low01['pred_level'].std():.4f}")
    print(f"  true_level  mean={low01['true_level'].mean():.4f}  "
          f"median={low01['true_level'].median():.4f}")
    for q in [0.25, 0.5, 0.75, 0.90, 0.95]:
        print(f"  pred_level  p{int(q*100):02d} = {low01['pred_level'].quantile(q):.4f}")


if __name__ == "__main__":
    main()
