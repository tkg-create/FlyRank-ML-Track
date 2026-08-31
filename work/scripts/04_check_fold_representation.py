"""Check whether the top-K queue is disproportionately drawn from one GroupKFold fold.

Pooling raw out-of-fold scores across 5 separately trained fold models doesn't produce a
comparable cross-fold ranking on its own — one fold's model can score systematically higher
without being more accurate, so a pooled top-K ends up dominated by that fold regardless of
which pages are actually declining. w07_pipeline_utils.calibrate_scores() fixes this, and
02_build_queue.py ranks the deployed queue on the calibrated score.

This script checks raw and calibrated side by side, so the fix's effect is visible in one
run rather than needing a before/after diff across two separate runs.

Requires fold_id and oof_rf_score_calibrated, both added to 01_load_and_score.py's output
as part of this fix. Rerun 01 if the input file predates that change.

Usage:
    python work/scripts/04_check_fold_representation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w07_pipeline_utils import (  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Check top-K fold representation, raw vs calibrated, before and after the fix.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def per_fold_calibration_table(model_df: pd.DataFrame, score_col: str, y: pd.Series) -> list[dict]:
    rows = []
    for fold in sorted(model_df["fold_id"].unique()):
        fold_rows = model_df[model_df["fold_id"] == fold]
        mean_score = fold_rows[score_col].mean()
        fold_base_rate = y.loc[fold_rows.index].mean()
        rows.append({
            "fold_id": int(fold),
            "n_rows": int(len(fold_rows)),
            "mean_score": round(float(mean_score), 4),
            "fold_base_rate": round(float(fold_base_rate), 4),
            "score_minus_base_rate": round(float(mean_score - fold_base_rate), 4),
        })
    return rows


def top_k_representation(model_df: pd.DataFrame, score_col: str) -> dict:
    expected_share = 1.0 / N_FOLDS
    col_results = {}
    for k in KS:
        top_k = model_df.sort_values(score_col, ascending=False).head(k)
        observed_counts = top_k["fold_id"].value_counts().reindex(range(1, N_FOLDS + 1), fill_value=0)
        observed_share = observed_counts / k
        max_fold = observed_share.idxmax()
        max_share = observed_share.max()
        col_results[str(k)] = {
            "counts_by_fold": {str(f): int(observed_counts[f]) for f in range(1, N_FOLDS + 1)},
            "share_by_fold": {str(f): round(float(observed_share[f]), 3) for f in range(1, N_FOLDS + 1)},
            "expected_share": round(expected_share, 3),
            "most_overrepresented_fold": int(max_fold),
            "most_overrepresented_share": round(float(max_share), 3),
        }
    return col_results


def main() -> None:
    args = parse_args()
    ensure_dirs()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Scored population not found: {input_path}. Run 01_load_and_score.py first.")

    model_df = pd.read_csv(input_path)
    for required in ["fold_id", "oof_rf_score", "oof_rf_score_calibrated"]:
        if required not in model_df.columns:
            raise ValueError(
                f"{input_path} has no {required} column. Rerun 01_load_and_score.py "
                "(current version) before using this check."
            )

    y = model_df["is_declining_proxy"].astype(int)

    fold_pop_share = model_df["fold_id"].value_counts(normalize=True).sort_index()

    print("=" * 70)
    print("PER-FOLD CALIBRATION — mean score vs. that fold's own base rate")
    print("A large gap on one fold means systematically inflated or deflated scores")
    print("relative to how many of its rows are actually positive.")
    print("=" * 70)

    calibration_results = {}
    for label, col in [("RAW", "oof_rf_score"), ("CALIBRATED", "oof_rf_score_calibrated")]:
        print(f"\n[{label}] ({col}):")
        rows = per_fold_calibration_table(model_df, col, y)
        for row in rows:
            print(f"  Fold {row['fold_id']}: n={row['n_rows']:,}  mean_score={row['mean_score']:.4f}  "
                  f"base_rate={row['fold_base_rate']:.4f}  gap={row['score_minus_base_rate']:+.4f}")
        gaps = [r["score_minus_base_rate"] for r in rows]
        print(f"  Gap spread: min={min(gaps):+.4f}, max={max(gaps):+.4f}, range={max(gaps) - min(gaps):.4f}")
        calibration_results[label.lower()] = rows

    print("\n" + "=" * 70)
    print("TOP-K FOLD REPRESENTATION — raw vs. calibrated, side by side")
    print("Expected ~20% per fold if scores were properly comparable across folds.")
    print("=" * 70)

    representation_results = {}
    for label, col in [
        ("oof_rf_score (raw)", "oof_rf_score"),
        ("oof_rf_score_calibrated (this is what's deployed)", "oof_rf_score_calibrated"),
    ]:
        print(f"\n[{label}]:")
        col_results = top_k_representation(model_df, col)
        for k in KS:
            r = col_results[str(k)]
            counts_str = ", ".join(f"fold{f}={r['counts_by_fold'][str(f)]}" for f in range(1, N_FOLDS + 1))
            print(f"  K={k}: {counts_str}   (most-represented: fold {r['most_overrepresented_fold']} "
                  f"at {r['most_overrepresented_share']:.0%})")
        representation_results[col] = col_results

    output = {
        "note": "Raw vs. calibrated fold representation, checked side by side to confirm the "
                "calibration fix (w07_pipeline_utils.calibrate_scores) actually resolves the "
                "skew found in the pre-fix run of this script.",
        "n_folds": N_FOLDS,
        "fold_pop_share": {str(int(f)): round(float(s), 3) for f, s in fold_pop_share.items()},
        "per_fold_calibration": calibration_results,
        "top_k_representation": representation_results,
    }

    output_path = Path(args.output)
    write_json(output_path, output)
    print(f"\nWrote {display_path(output_path)}")


if __name__ == "__main__":
    main()
