"""Build the ranked action queue from a scored population and export the paper-facing artifacts.

Consolidates work from previous notebooks (archetype/coverage assignment, queue ranking) and output (metrics JSON, charts, report, local-only queue CSV).

The queue is ranked by oof_rf_score_calibrated alone, no rule-based nudge, and coverage
(formerly "confidence") measures data volume, not outcome reliability — see
w07_pipeline_utils.py for both.

Usage:
    python work/scripts/02_build_queue.py
    python work/scripts/02_build_queue.py --input work/data/processed/custom.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w07_pipeline_utils import (  # noqa: E402
    BOUNDARY_MARGIN_PCT,
    CHART_DIR,
    HIGH_SCORE_PERCENTILE,
    LARGE_SWING_PERCENTILE,
    LOW_DATA_PERCENTILE,
    OUTPUT_DIR,
    PROCESSED_DIR,
    assign_archetype,
    assign_coverage,
    display_path,
    ensure_dirs,
    write_json,
)

DEFAULT_INPUT = PROCESSED_DIR / "w07_scored_population.csv"
DEFAULT_QUEUE_OUTPUT = PROCESSED_DIR / "w07_ranked_queue.csv"  # local only, never committed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ranked queue and export paper artifacts.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--queue-output", default=str(DEFAULT_QUEUE_OUTPUT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def use_calibrated_score(model_df: pd.DataFrame) -> pd.DataFrame:
    """Swaps the calibrated score into the oof_rf_score column that every downstream
    function (assign_archetype, assign_coverage) already reads, so none of that code
    needed to change. Raw score is kept as oof_rf_score_raw for audit and as a tiebreak.
    """
    if "oof_rf_score_calibrated" not in model_df.columns:
        raise ValueError(
            "oof_rf_score_calibrated column not found. This CSV predates the calibration "
            "fix — rerun 01_load_and_score.py (current version) to regenerate it."
        )
    model_df = model_df.copy()
    model_df["oof_rf_score_raw"] = model_df["oof_rf_score"]
    model_df["oof_rf_score"] = model_df["oof_rf_score_calibrated"]
    return model_df


def build_queue(model_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    high_score_cut = model_df.loc[model_df["baseline_score"] == 0, "oof_rf_score"].quantile(HIGH_SCORE_PERCENTILE)
    large_swing_cut = model_df["position_change"].abs().quantile(LARGE_SWING_PERCENTILE)
    low_data_cut = model_df["total_impressions_full"].quantile(LOW_DATA_PERCENTILE)
    boundary_margin = BOUNDARY_MARGIN_PCT * high_score_cut

    print("  Assigning archetypes...", flush=True)
    archetype_info = model_df.apply(
        lambda row: assign_archetype(row, high_score_cut, large_swing_cut),
        axis=1, result_type="expand",
    )
    model_df = model_df.copy()
    model_df[["priority_tier", "archetype", "action"]] = archetype_info
    print("  Assigning coverage...", flush=True)
    model_df["coverage"] = model_df.apply(
        lambda row: assign_coverage(row, low_data_cut, high_score_cut, boundary_margin),
        axis=1,
    )
    print("  Sorting ranked queue...", flush=True)

    # Tiebreak by the raw score, not sort_values' default (row order) — defensive: percentile
    # rank shouldn't create wide ties, but any genuine tie (two rows the forest scored
    # identically) should break on real signal, not incidental order.
    ranked_queue = model_df.sort_values(
        ["oof_rf_score", "oof_rf_score_raw"], ascending=[False, False]
    ).reset_index(drop=True)

    thresholds = {
        "high_score_cut": float(high_score_cut),
        "large_swing_cut": float(large_swing_cut),
        "low_data_cut": float(low_data_cut),
        "boundary_margin": float(boundary_margin),
    }
    return ranked_queue, thresholds


def build_metrics(model_df: pd.DataFrame, thresholds: dict) -> dict:
    rule_only = ["zero_clicks_and_worsened", "zero_clicks_only", "position_worsened_only", "no_flag"]
    rule_agreement_means = (
        model_df.loc[model_df["archetype"].isin(rule_only)]
        .groupby("archetype")["oof_rf_score"].mean().round(3).to_dict()
    )

    caution_count = int((
        (model_df["archetype"] == "model_only_catch")
        & (model_df["action"].str.contains("verify before acting", na=False))
    ).sum())
    model_only_total = int((model_df["archetype"] == "model_only_catch").sum())

    return {
        "population_size": int(len(model_df)),
        "seed": 42,
        "score_used": "oof_rf_score_calibrated (percentile rank within fold) — see w07_pipeline_utils.py",
        "thresholds": thresholds,
        "archetype_counts": model_df["archetype"].value_counts().to_dict(),
        "rule_agreement_mean_scores": rule_agreement_means,
        "coverage_split": model_df["coverage"].value_counts().to_dict(),
        "coverage_low_pct": round(float((model_df["coverage"] == "low").mean()), 3),
        "model_only_catch_caution": {
            "flagged": caution_count,
            "total": model_only_total,
            "rate": round(caution_count / model_only_total, 4) if model_only_total else 0.0,
        },
        "feature_distribution_baseline": model_df[
            ["avg_position_fh", "log_impressions_fh", "log_clicks_fh", "ctr_fh", "position_change", "has_position_trend"]
        ].describe().round(3).to_dict(),
    }


def make_charts(model_df: pd.DataFrame, chart_dir: Path) -> pd.DataFrame:
    chart_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = model_df["archetype"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color="#426B69")
    ax.set_title("Archetype mix")
    ax.set_xlabel("Rows")
    plt.tight_layout()
    plt.savefig(chart_dir / "archetype_mix.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cov_counts = model_df["coverage"].value_counts().reindex(["high", "low"])
    ax.bar(cov_counts.index, cov_counts.values, color="#6F4E7C")
    ax.set_title("Data coverage mix")
    ax.set_ylabel("Rows")
    plt.tight_layout()
    plt.savefig(chart_dir / "coverage_mix.svg")
    plt.close(fig)

    tail_check = model_df.groupby("archetype")["oof_rf_score"].agg(
        mean="mean", p50="median", p95=lambda s: s.quantile(0.95),
        p99=lambda s: s.quantile(0.99), max="max",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    tail_check[["p50", "mean", "p95", "p99", "max"]].plot(
        kind="bar", ax=ax, color=["#426B69", "#6F4E7C", "#9BB88A", "#C9A66B", "#B85C5C"]
    )
    ax.set_ylabel("Calibrated score")
    ax.set_title("Score distribution by archetype")
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(chart_dir / "score_distribution_by_archetype.svg")
    plt.close(fig)

    return tail_check


def write_queue_diagnostics(ranked_queue: pd.DataFrame, tail_check: pd.DataFrame, output_dir: Path) -> None:
    """Coverage-outcome check, K-sweep, and score distribution by archetype — the evidence
    behind two Limitations entries (the coverage rename, the K=50 concentration finding).
    tail_check is passed in rather than recomputed, since make_charts() already built it
    for the matching chart.
    """
    coverage_outcome = (
        ranked_queue.groupby(["archetype", "coverage"])["is_declining_proxy"]
        .agg(["mean", "count"])
        .round(3)
    )
    k_sweep = {
        str(k): ranked_queue.head(k)["archetype"].value_counts().to_dict()
        for k in [50, 100, 500, 1000]
    }

    write_json(output_dir / "queue_diagnostics.json", {
        "note": "Coverage-outcome check, archetype counts at several K, and score "
                "distribution by archetype — see Limitations.",
        "coverage_outcome": {
            f"{archetype}|{coverage}": {"mean": float(row["mean"]), "count": int(row["count"])}
            for (archetype, coverage), row in coverage_outcome.iterrows()
        },
        "k_sweep": k_sweep,
        "score_distribution_by_archetype": tail_check.round(3).to_dict(orient="index"),
    })
    print(f"Wrote {display_path(output_dir / 'queue_diagnostics.json')}")


def write_report(ranked_queue: pd.DataFrame, metrics: dict, report_path: Path) -> None:
    action_by_archetype = ranked_queue.groupby("archetype")["action"].first()

    report = f"""# ML-10 Content Action Playbook Report

Scored population: {metrics['population_size']:,} pages, {metrics.get('label_window', 'March 2026')}, `is_declining_proxy` label.

Queue is ranked by the calibrated model score alone, no rule-based nudge (see
work/outputs/fold_representation_check.json and work/outputs/capstone_precision_at_k.json
for why — every combination method tested reintroduced fold skew, so the queue uses the
model score directly).

## Archetype breakdown

| Archetype | Count | Action |
|---|---:|---|
"""
    for archetype, count in ranked_queue["archetype"].value_counts().items():
        report += f"| `{archetype}` | {count:,} | {action_by_archetype[archetype]} |\n"

    coverage_counts = metrics["coverage_split"]
    report += f"""
## Data coverage split

This tier was originally called "confidence." Renamed after a check found low-tier rows
have a HIGHER real decline rate than high-tier rows in every archetype — it tracks data
volume and score-boundary proximity, not reliability. See Limitations.

- High: {coverage_counts.get('high', 0):,}
- Low: {coverage_counts.get('low', 0):,} ({metrics['coverage_low_pct']:.1%})

## Rule-agreement check (model score vs. rule severity, rule-only archetypes)

| Archetype | Mean oof_rf_score (calibrated) |
|---|---:|
"""
    for archetype, score in metrics["rule_agreement_mean_scores"].items():
        report += f"| `{archetype}` | {score:.3f} |\n"

    caution = metrics["model_only_catch_caution"]
    report += f"""
## model_only_catch caution flag

{caution['flagged']} of {caution['total']} rows ({caution['rate']:.1%}) flagged for large position-swing caution.

## Top 20 queue preview

"""
    top20 = ranked_queue.head(20)[["content_hash_id", "archetype", "action", "coverage", "oof_rf_score"]]
    try:
        report += top20.to_markdown(index=False)
    except ImportError:
        report += top20.to_string(index=False)

    report += """

## Practical use

Use this queue as a reviewer aid, not an automatic action trigger. Every action here is a starting point for review, not an instruction to execute unread. 
See the source notebook (work/notebooks/w07_action_playbook.ipynb) for intended use, limits, human review rules, the no-go list, and monitoring/retrain triggers.

## Generated files

- `work/outputs/w07_metrics.json`
- `work/outputs/charts/archetype_mix.svg`
- `work/outputs/charts/coverage_mix.svg`
- `work/outputs/charts/score_distribution_by_archetype.svg`
- `work/data/processed/w07_ranked_queue.csv` (local only — gitignored, not committed;
  regenerate by rerunning work/scripts/run_all.py rather than expecting this file to
  exist in a fresh clone)
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Scored population not found: {input_path}. Run 01_load_and_score.py first."
        )
    model_df = pd.read_csv(input_path)
    model_df = use_calibrated_score(model_df)

    ranked_queue, thresholds = build_queue(model_df)
    metrics = build_metrics(ranked_queue, thresholds)
    metrics["label_window"] = "impressions before vs after 2026-03-16, month=2026-03"

    output_dir = Path(args.output_dir)
    chart_dir = output_dir / "charts"
    tail_check = make_charts(ranked_queue, chart_dir)

    write_json(output_dir / "w07_metrics.json", metrics)
    write_report(ranked_queue, metrics, output_dir / "w07_report.md")
    write_queue_diagnostics(ranked_queue, tail_check, output_dir)

    queue_output = Path(args.queue_output)
    queue_output.parent.mkdir(parents=True, exist_ok=True)
    ranked_queue.to_csv(queue_output, index=False)

    print(f"Archetype mix:\n{ranked_queue['archetype'].value_counts()}")
    print(f"Wrote {output_dir / 'w07_metrics.json'}")
    print(f"Wrote {output_dir / 'w07_report.md'}")
    print(f"Wrote charts to {chart_dir}")
    print(f"Wrote {queue_output} (local only — gitignored, not committed)")


if __name__ == "__main__":
    main()
