"""Validate combined_score's ranking quality

w06 validated oof_rf_score with GroupKFold precision@K. 
w07 then built combined_score = oof_rf_score + BASELINE_NUDGE_WEIGHT * baseline_score on top of that validated score (see w07_pipeline_utils.py), 
but combined_score itself was never re-run through precision@K — the queue that actually gets deployed was never the thing that got validated. 
This script closes that gap for the capstone's headline K=50 result.

Updated for the calibration fix: 02_build_queue.py now builds combined_score from
oof_rf_score_calibrated, not the raw oof_rf_score. This script validates that calibrated
version, since that's what's actually deployed. It also runs one extra check before
trusting anything else here: calibration is a monotonic transform, so it can never invert
the order of two rows with genuinely different raw scores. It can, however, map a range of
close raw scores to the same calibrated value (isotonic regression produces flat plateaus),
and ties like that can be reordered by the sort. So fold-internal precision@K for
oof_rf_score_calibrated is expected to stay CLOSE to fold-internal precision@K for the
original raw oof_rf_score, not necessarily identical. That's checked directly below —
small drift is expected and fine, a large drift means something other than tie-breaking is
going on and is worth investigating before trusting the numbers that follow.

Method: GroupKFold is deterministic given the same data, row order, and grouping column — no random_state involved — 
so re-splitting the already-scored population (the output of 01_load_and_score.py) reproduces the exact same 5 folds used to produce oof_rf_score in the first place. 
No retraining happens here; this only re-slices existing scores into their original per-fold test membership so precision@K can be computed.

Built-in sanity check: baseline_rule and oof_rf_score (raw) precision@K are recomputed here too, using this same fold reconstruction, and printed plainly. 
Compare these against your own w06 Cell 10 printout before trusting the new combined_score numbers below them — 
if the fold reconstruction were wrong (e.g. row order changed between 01's write and this script's read), 
the oof_rf_score column here would drift from w06's reported values.

ASSUMES: work/data/processed/w07_scored_population.csv exists, is in the same row order 01_load_and_score.py wrote it in (i.e. not re-sorted or filtered since), 
and was produced by the current version of 01_load_and_score.py (i.e. has an oof_rf_score_calibrated column). 
Rerun 01_load_and_score.py fresh if either is ever in doubt.

Usage:
    python work/scripts/03_validate_combined_score.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w07_pipeline_utils import (  # noqa: E402
    BASELINE_NUDGE_WEIGHT,
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
INVARIANCE_TOLERANCE = 0.02  # fold-internal precision@K, raw vs calibrated. Platt scaling is
                              # continuous, so real ties should be rare (only genuine raw-score
                              # duplicates) — tighter than the isotonic version's tolerance,
                              # which had to accommodate large deliberate plateaus.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute per-fold precision@K for baseline_rule, oof_rf_score, and combined_score "
        "(now built from the calibrated score) using the fold split that already produced oof_rf_score."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def precision_at_k(order_labels: np.ndarray, k: int) -> float:
    return float(np.asarray(order_labels)[:k].mean())


def score_fold(
    model_df: pd.DataFrame, test_idx: np.ndarray, score_col: str, y: pd.Series,
    tiebreak_col: str | None = None,
) -> np.ndarray:
    """Rank a fold's test rows by score_col (descending), optionally breaking ties with
    tiebreak_col, return the ranked y labels. combined_score is ranked with
    oof_rf_score_raw as the tiebreak here specifically to match 02_build_queue.py's real
    sort — this function should validate what's actually deployed, not a simpler version of it.
    """
    test_slice = model_df.iloc[test_idx]
    if tiebreak_col:
        order = test_slice.sort_values([score_col, tiebreak_col], ascending=[False, False]).index
    else:
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
    model_df["combined_score"] = model_df["oof_rf_score_calibrated"] + BASELINE_NUDGE_WEIGHT * model_df["baseline_score"]

    X = model_df[FEATURE_COLS].astype(float)
    groups = model_df["client_hash_id"]

    print(f"base_rate (is_declining_proxy mean): {base_rate:.3f}  "
          "-- report this next to any precision@K number (template Section 8 checklist)")
    print(f"Reconstructing the same GroupKFold({N_FOLDS}) split used by 01_load_and_score.py...")
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_records = []
    for fold_num, (_, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        y_test = y.iloc[test_idx]

        scored = model_df.iloc[test_idx].copy()
        scored["baseline_rank_score"] = scored["zero_clicks_at_position"] * 2 + scored["position_worsened"]
        baseline_order = scored.sort_values(
            ["baseline_rank_score", "total_impressions_full"], ascending=[False, False]
        ).index
        baseline_ranked = y.loc[baseline_order].values

        rf_ranked_raw = score_fold(model_df, test_idx, "oof_rf_score", y)
        rf_ranked_calibrated = score_fold(model_df, test_idx, "oof_rf_score_calibrated", y)
        combined_ranked = score_fold(model_df, test_idx, "combined_score", y, tiebreak_col="oof_rf_score_raw")

        for k in KS:
            fold_records.append({
                "fold": fold_num,
                "k": k,
                "baseline_rule": precision_at_k(baseline_ranked, k),
                "oof_rf_score_raw": precision_at_k(rf_ranked_raw, k),
                "oof_rf_score_calibrated": precision_at_k(rf_ranked_calibrated, k),
                "combined_score": precision_at_k(combined_ranked, k),
            })

    fold_df = pd.DataFrame(fold_records)

    # --- Tie-breaking check: calibration is monotonic, so it can't invert the order of two
    # rows with genuinely different raw scores, but it can tie rows with close raw scores
    # together, and ties can then get reordered. Small drift here is expected, not a bug.
    max_diff = (fold_df["oof_rf_score_raw"] - fold_df["oof_rf_score_calibrated"]).abs().max()
    print(f"\nTie-breaking check: max |raw - calibrated| fold-internal precision@K difference = {max_diff:.4f}")
    if max_diff > INVARIANCE_TOLERANCE:
        print(f"  WARNING: exceeds tolerance ({INVARIANCE_TOLERANCE}) — more drift than tie-breaking "
              "alone would typically produce. Investigate before trusting the numbers below.")
    else:
        print("  OK — small drift, consistent with tie-breaking among close raw scores.")

    print("\nPer-fold precision@K (compare baseline_rule / oof_rf_score_raw columns against your own w06 Cell 10 output):")
    print(fold_df.to_string(index=False))

    summary = (
        fold_df.groupby("k")[["baseline_rule", "oof_rf_score_raw", "oof_rf_score_calibrated", "combined_score"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\nMean +/- std across 5 folds:")
    print(summary.to_string())

    print(f"\n95% bootstrap CI on fold-level gaps (n_boot={N_BOOT}, seed={BOOT_SEED}):")
    ci_records = {}
    for k in [100, 200]:
        fold_k = fold_df[fold_df["k"] == k]

        diff_vs_baseline = (fold_k["combined_score"] - fold_k["baseline_rule"]).values
        ci_b_low, ci_b_high = bootstrap_ci(diff_vs_baseline)
        print(f"  K={k} combined_score - baseline_rule: mean={diff_vs_baseline.mean():.3f}, "
              f"95% CI=[{ci_b_low:.3f}, {ci_b_high:.3f}]"
              f"{'  <-- contains zero' if ci_b_low <= 0 <= ci_b_high else ''}")

        diff_vs_oof = (fold_k["combined_score"] - fold_k["oof_rf_score_calibrated"]).values
        ci_o_low, ci_o_high = bootstrap_ci(diff_vs_oof)
        print(f"  K={k} combined_score - oof_rf_score_calibrated: mean={diff_vs_oof.mean():.3f}, "
              f"95% CI=[{ci_o_low:.3f}, {ci_o_high:.3f}]"
              f"{'  <-- contains zero' if ci_o_low <= 0 <= ci_o_high else ''} "
              "(checks whether the 0.03 baseline nudge moved the headline number at all)")

        ci_records[str(k)] = {
            "combined_vs_baseline": {"mean_diff": float(diff_vs_baseline.mean()), "ci_low": float(ci_b_low), "ci_high": float(ci_b_high)},
            "combined_vs_oof_rf_calibrated": {"mean_diff": float(diff_vs_oof.mean()), "ci_low": float(ci_o_low), "ci_high": float(ci_o_high)},
        }

    output = {
        "note": "Recomputed from the OOF fold membership that already produced oof_rf_score in "
                "01_load_and_score.py. No retraining. combined_score is built from the CALIBRATED "
                "score, matching what 02_build_queue.py actually deploys — see script docstring.",
        "base_rate": base_rate,
        "n_folds": N_FOLDS,
        "ks": KS,
        "tie_breaking_check_max_diff": float(max_diff),
        "fold_level": fold_df.to_dict(orient="records"),
        "summary_mean_std": {
            str(k): {
                col: {"mean": float(summary.loc[k, (col, "mean")]), "std": float(summary.loc[k, (col, "std")])}
                for col in ["baseline_rule", "oof_rf_score_raw", "oof_rf_score_calibrated", "combined_score"]
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
