"""Validate the deployed queue score's ranking quality.

w06 validated oof_rf_score with GroupKFold precision@K. During capstone validation,
oof_rf_score was replaced with a fold-fair calibrated version (oof_rf_score_calibrated),
and the queue was changed to rank on it alone, with no rule-based combination (see
w07_pipeline_utils.py for why). This script validates that score directly, since it's what
actually gets deployed.

One extra check runs before trusting anything else here: calibration (percentile rank
within each fold) can't invert the order of two rows with genuinely different raw scores,
so fold-internal precision@K for oof_rf_score_calibrated should stay close to fold-internal
precision@K for the original raw oof_rf_score. Small drift is expected from real ties in
the raw score; a large drift means something else changed and is worth investigating.

Method: GroupKFold is deterministic given the same data, row order, and grouping column, so
re-splitting the already-scored population reproduces the same 5 folds used to produce
oof_rf_score in 01_load_and_score.py. No retraining happens here.

Built-in sanity check: baseline_rule and oof_rf_score_raw precision@K are recomputed here
too, using this same fold reconstruction. Compare these against your own w06 Cell 10
printout before trusting the calibrated numbers below them.

ASSUMES: work/data/processed/w07_scored_population.csv exists, is in the same row order
01_load_and_score.py wrote it in, and has an oof_rf_score_calibrated column. Rerun
01_load_and_score.py fresh if either is in doubt.

Usage:
    python work/scripts/03_validate_deployed_score.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w07_pipeline_utils import (  # noqa: E402
    CHART_DIR,
    FEATURE_COLS,
    N_FOLDS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    display_path,
    ensure_dirs,
    write_json,
)

DEFAULT_INPUT = PROCESSED_DIR / "w07_scored_population.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "capstone_precision_at_k.json"

KS = [20, 50, 100, 200]
N_BOOT = 10_000
BOOT_SEED = 42  # matches w06's bootstrap CI cell
TIEBREAK_TOLERANCE = 0.02  # fold-internal precision@K, raw vs calibrated. Percentile rank
                            # preserves within-fold order exactly, so any real difference
                            # should only come from genuine raw-score ties.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute per-fold precision@K for baseline_rule, oof_rf_score_raw, "
        "and oof_rf_score_calibrated (the deployed score) using the fold split that "
        "already produced oof_rf_score."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def precision_at_k(order_labels: np.ndarray, k: int) -> float:
    return float(np.asarray(order_labels)[:k].mean())


def score_fold(model_df: pd.DataFrame, test_idx: np.ndarray, score_col: str, y: pd.Series) -> np.ndarray:
    """Rank a fold's test rows by score_col (descending), return the ranked y labels."""
    test_slice = model_df.iloc[test_idx]
    order = test_slice[score_col].sort_values(ascending=False).index
    return y.loc[order].values


def bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% bootstrap CI on the mean of fold-level diffs. Same method as w06 Cell 12."""
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    return tuple(np.percentile(boot_means, [2.5, 97.5]))


def main() -> None:
    args = parse_args()
    ensure_dirs()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Scored population not found: {input_path}. Run 01_load_and_score.py first."
        )
    model_df = pd.read_csv(input_path)
    if "oof_rf_score" not in model_df.columns:
        raise ValueError(
            f"{input_path} has no oof_rf_score column — was it written by 01_load_and_score.py, or by an older/partial run?"
        )
    if "oof_rf_score_calibrated" not in model_df.columns:
        raise ValueError(
            f"{input_path} has no oof_rf_score_calibrated column. This CSV predates the "
            "calibration fix — rerun 01_load_and_score.py (current version) first."
        )

    y = model_df["is_declining_proxy"].astype(int)
    base_rate = float(y.mean())

    X = model_df[FEATURE_COLS].astype(float)
    groups = model_df["client_hash_id"]

    print(f"base_rate (is_declining_proxy mean): {base_rate:.3f}  "
          "-- report this next to any precision@K number (template Section 8 checklist)")
    print(f"Reconstructing the same GroupKFold({N_FOLDS}) split used by 01_load_and_score.py...")
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_records = []
    for fold_num, (_, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        scored = model_df.iloc[test_idx].copy()
        scored["baseline_rank_score"] = scored["zero_clicks_at_position"] * 2 + scored["position_worsened"]
        baseline_order = scored.sort_values(
            ["baseline_rank_score", "total_impressions_full"], ascending=[False, False]
        ).index
        baseline_ranked = y.loc[baseline_order].values

        rf_ranked_raw = score_fold(model_df, test_idx, "oof_rf_score", y)
        rf_ranked_calibrated = score_fold(model_df, test_idx, "oof_rf_score_calibrated", y)

        for k in KS:
            fold_records.append({
                "fold": fold_num,
                "k": k,
                "baseline_rule": precision_at_k(baseline_ranked, k),
                "oof_rf_score_raw": precision_at_k(rf_ranked_raw, k),
                "oof_rf_score_calibrated": precision_at_k(rf_ranked_calibrated, k),
            })

    fold_df = pd.DataFrame(fold_records)

    # --- Tie-breaking check: percentile rank preserves order within a fold exactly, so
    # this should show ~0 drift unless real ties in the raw score cause reordering.
    max_diff = (fold_df["oof_rf_score_raw"] - fold_df["oof_rf_score_calibrated"]).abs().max()
    print(f"\nTie-breaking check: max |raw - calibrated| fold-internal precision@K difference = {max_diff:.4f}")
    if max_diff > TIEBREAK_TOLERANCE:
        print(f"  WARNING: exceeds tolerance ({TIEBREAK_TOLERANCE}) — more drift than tie-breaking "
              "alone would typically produce. Investigate before trusting the numbers below.")
    else:
        print("  OK — small drift, consistent with tie-breaking among close raw scores.")

    print("\nPer-fold precision@K (compare baseline_rule / oof_rf_score_raw columns against your own w06 Cell 10 output):")
    print(fold_df.to_string(index=False))

    summary = (
        fold_df.groupby("k")[["baseline_rule", "oof_rf_score_raw", "oof_rf_score_calibrated"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\nMean +/- std across 5 folds:")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(KS))
    width = 0.35
    baseline_vals = [summary.loc[k, ("baseline_rule", "mean")] for k in KS]
    model_vals = [summary.loc[k, ("oof_rf_score_calibrated", "mean")] for k in KS]
    ax.bar([i - width / 2 for i in x], baseline_vals, width, label="baseline_rule", color="#426B69")
    ax.bar([i + width / 2 for i in x], model_vals, width, label="model (calibrated)", color="#6F4E7C")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"K={k}" for k in KS])
    ax.set_ylabel("Precision@K")
    ax.set_title("Model vs. baseline rule")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "precision_at_k.svg")
    plt.close(fig)
    print(f"Wrote {display_path(CHART_DIR / 'precision_at_k.svg')}")

    print(f"\n95% bootstrap CI on fold-level gap vs. baseline_rule (n_boot={N_BOOT}, seed={BOOT_SEED}):")
    ci_records = {}
    for k in [50, 100, 200]:
        fold_k = fold_df[fold_df["k"] == k]

        diff_vs_baseline = (fold_k["oof_rf_score_calibrated"] - fold_k["baseline_rule"]).values
        ci_low, ci_high = bootstrap_ci(diff_vs_baseline)
        print(f"  K={k} oof_rf_score_calibrated - baseline_rule: mean={diff_vs_baseline.mean():.3f}, "
              f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]"
              f"{'  <-- contains zero' if ci_low <= 0 <= ci_high else ''}")

        ci_records[str(k)] = {
            "calibrated_vs_baseline": {"mean_diff": float(diff_vs_baseline.mean()), "ci_low": float(ci_low), "ci_high": float(ci_high)},
        }

    output = {
        "note": "Recomputed from the OOF fold membership that already produced oof_rf_score in "
                "01_load_and_score.py. No retraining. oof_rf_score_calibrated is the score "
                "the deployed queue is ranked by directly — see script docstring.",
        "base_rate": base_rate,
        "n_folds": N_FOLDS,
        "ks": KS,
        "tie_breaking_check_max_diff": float(max_diff),
        "fold_level": fold_df.to_dict(orient="records"),
        "summary_mean_std": {
            str(k): {
                col: {"mean": float(summary.loc[k, (col, "mean")]), "std": float(summary.loc[k, (col, "std")])}
                for col in ["baseline_rule", "oof_rf_score_raw", "oof_rf_score_calibrated"]
            }
            for k in KS
        },
        "bootstrap_ci": ci_records,
    }

    output_path = Path(args.output)
    write_json(output_path, output)
    print(f"\nWrote {display_path(output_path)}")


if __name__ == "__main__":
    main()
