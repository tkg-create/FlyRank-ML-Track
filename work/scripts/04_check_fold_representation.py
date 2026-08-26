"""Check whether the top-K queue is disproportionately drawn from one GroupKFold fold.

w06's error analysis found that pooling raw out-of-fold probabilities across 5 separately
trained fold models does not produce a clean cross-client ranking. One fold's model output
systematically higher raw scores without being more accurate, so a pooled top-50 built by
sorting all folds' scores together came out almost entirely made of that fold's rows.

That check was run on a specific diagnostic sample in w06. This script re-runs the same
check on the real scored population and the real deployed queue score, to find out whether
the same skew shows up here before the capstone's headline K=50 claim depends on it.

Requires fold_id, added to 01_load_and_score.py's output — rerun 01 if the input file
predates that change.

Usage:
    python work/scripts/04_check_fold_representation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w07_pipeline_utils import (  # noqa: E402
    BASELINE_NUDGE_WEIGHT,
    N_FOLDS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    display_path,
    ensure_dirs,
    write_json,
)

DEFAULT_INPUT = PROCESSED_DIR / "w07_scored_population.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "fold_representation_check.json"

KS = [20, 50, 100, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check top-K fold representation and per-fold score calibration.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Scored population not found: {input_path}. Run 01_load_and_score.py first.")

    model_df = pd.read_csv(input_path)
    if "fold_id" not in model_df.columns:
        raise ValueError(
            f"{input_path} has no fold_id column. Rerun 01_load_and_score.py with the "
            "fold_id tracking change before using this check."
        )

    model_df["combined_score"] = model_df["oof_rf_score"] + BASELINE_NUDGE_WEIGHT * model_df["baseline_score"]
    y = model_df["is_declining_proxy"].astype(int)

    # --- 1. Per-fold mean score and precision, on the FULL fold (not top-K) ---
    # If one fold's model is systematically over-confident, its mean oof_rf_score will be
    # noticeably higher than other folds' even though its own precision isn't higher to match.
    print("Per-fold mean oof_rf_score vs. per-fold precision (entire fold, not just top-K):")
    fold_summary = []
    for fold in sorted(model_df["fold_id"].unique()):
        fold_rows = model_df[model_df["fold_id"] == fold]
        mean_score = fold_rows["oof_rf_score"].mean()
        # precision at this fold's own size, i.e. how many of ALL its rows are true positives —
        # not a ranking metric, just "is this fold's average score inflated relative to its own base rate"
        fold_base_rate = fold_rows[y.name].mean() if y.name in fold_rows else fold_rows["is_declining_proxy"].mean()
        fold_summary.append({
            "fold_id": int(fold),
            "n_rows": int(len(fold_rows)),
            "mean_oof_rf_score": round(float(mean_score), 4),
            "fold_base_rate": round(float(fold_base_rate), 4),
            "score_minus_base_rate": round(float(mean_score - fold_base_rate), 4),
        })
        print(f"  Fold {fold}: n={len(fold_rows):,}  mean_score={mean_score:.4f}  "
              f"base_rate={fold_base_rate:.4f}  gap={mean_score - fold_base_rate:+.4f}")

    gaps = [r["score_minus_base_rate"] for r in fold_summary]
    print(f"\n  Gap spread across folds: min={min(gaps):+.4f}, max={max(gaps):+.4f}, "
          f"range={max(gaps) - min(gaps):.4f}")
    print("  A fold whose gap is much larger than the others is producing systematically")
    print("  inflated scores relative to how many of its rows are actually positive.\n")

    # --- 2. Top-K fold representation, expected vs. observed ---
    expected_share = 1.0 / N_FOLDS
    fold_pop_share = model_df["fold_id"].value_counts(normalize=True).sort_index()

    results = {}
    for score_col in ["oof_rf_score", "combined_score"]:
        print(f"Top-K fold representation for {score_col}:")
        col_results = {}
        for k in KS:
            top_k = model_df.sort_values(score_col, ascending=False).head(k)
            observed_counts = top_k["fold_id"].value_counts().reindex(range(1, N_FOLDS + 1), fill_value=0)
            observed_share = observed_counts / k
            max_fold = observed_share.idxmax()
            max_share = observed_share.max()

            print(f"  K={k}: " + ", ".join(f"fold{f}={observed_counts[f]}" for f in range(1, N_FOLDS + 1))
                  + f"   (expected ~{expected_share:.0%} each, most-represented: fold {max_fold} at {max_share:.0%})")

            col_results[str(k)] = {
                "counts_by_fold": {str(f): int(observed_counts[f]) for f in range(1, N_FOLDS + 1)},
                "share_by_fold": {str(f): round(float(observed_share[f]), 3) for f in range(1, N_FOLDS + 1)},
                "expected_share": round(expected_share, 3),
                "most_overrepresented_fold": int(max_fold),
                "most_overrepresented_share": round(float(max_share), 3),
            }
        results[score_col] = col_results
        print()

    output = {
        "note": "Diagnostic for whether pooled OOF scores are skewed toward one fold's model "
                "(w06 finding), checked against the real scored population and real queue "
                "scores rather than w06's diagnostic sample.",
        "n_folds": N_FOLDS,
        "fold_pop_share": {str(int(f)): round(float(s), 3) for f, s in fold_pop_share.items()},
        "per_fold_calibration": fold_summary,
        "top_k_representation": results,
    }

    output_path = Path(args.output)
    write_json(output_path, output)
    print(f"Wrote {display_path(output_path)}")


if __name__ == "__main__":
    main()
