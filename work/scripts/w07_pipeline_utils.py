"""Shared configuration and helpers for the lane's consolidated pipeline.

This plays the same role work/scripts/ that ml_utils.py plays in scripts/ —
paths, constants, and small reusable helpers, imported by every numbered
script and by run_all.py. It holds no modeling logic itself.

Every threshold and design choice below is a *validated, final* decision
from w04-w07 (see each notebook for the full reasoning and the paths that
were tried and rejected). Nothing here is new — this consolidates, it does
not extend.
"""
from __future__ import annotations

import json
import os
from getpass import getpass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- Paths -------------------------------------------------------------
# This file lives at work/scripts/w07_pipeline_utils.py, so parents[2] is
# the repo root (work/scripts -> work -> root).
ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "work"
PROCESSED_DIR = WORK_DIR / "data" / "processed"
OUTPUT_DIR = WORK_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

# All of the above sit under work/, where work/**/*.csv is gitignored
# repo-wide (see work/README.md rule 2). CSVs written by these scripts —
# the scored population, the full ranked queue — are local, regenerable
# artifacts, never committed. Only the JSON, charts, and report under
# OUTPUT_DIR are meant to be committed.


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _json_default(value: Any) -> Any:
    # numpy scalar types (from .describe(), thresholds, etc.) aren't
    # natively JSON-serializable.
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def display_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def get_hf_token() -> str:
    """Returns the Hugging Face token, preferring an already-set env var.

    Lets run_all.py prompt once via getpass() and have the value inherited
    by every subprocess step, or lets a script be run completely standalone
    and prompt for itself. Never hardcode a token — this script is committed
    to a public repo.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    return getpass("Paste your Hugging Face READ token: ")


# --- Data scope ----------------------------------------------------------
# Validated on a single month only (w05/w06/w07). The pipeline logic itself
# is not month-specific, but nothing has been tested on another month, so
# these are hardcoded rather than parameterized.
#
# To make this genuinely month-agnostic (future work, not done here):
#   - accept a --month YYYY-MM argument
#   - replace SCORED_MONTH below with that argument
#   - HALF_SPLIT_DATE / WEEK1_END_DATE would need to be derived from the
#     month's actual day count (e.g. calendar.monthrange) rather than
#     hardcoded — March 2026 has 31 days, half falls on the 16th, and the
#     first-half/second-half trend split falls on the 8th. That derivation
#     was never built or tested here; treat it as a real design task for
#     the capstone, not a trivial swap.
SCORED_MONTH = "2026-03"
HALF_SPLIT_DATE = "2026-03-16"  # first half vs second half of the month (label)
WEEK1_END_DATE = "2026-03-08"   # week 1 vs week 2 of the first half (position trend)

HF_MONTH_PATH = (
    f"hf://datasets/FlyRank/internship-warehouse/"
    f"fact_content_daily_performance/month={SCORED_MONTH}/data_0.parquet"
)

# --- Model config ----------------------------------------------------------
RANDOM_STATE = 42  # matches w05/w06/w07 throughout
N_FOLDS = 5         # GroupKFold, grouped by client_hash_id (w06/w07)

FEATURE_COLS = [
    "avg_position_fh",
    "log_impressions_fh",
    "log_clicks_fh",
    "ctr_fh",
    "position_change",
    "has_position_trend",
]
# Deliberately excluded: zero_clicks_at_position_fh was built (w06) but
# never added here — it only ever backed a w06 diagnostic LR interaction
# term. This is a named, documented limitation (see w07 Section 2), not an
# oversight: the model only sees CTR problems indirectly via ctr_fh /
# log_clicks_fh. Also excluded: the w05 diagnostic LR and its interaction
# feature — that comparison served w05's method selection, it doesn't feed
# the playbook.

# --- Baseline rule (w04) ----------------------------------------------------
# baseline_score = zero_clicks_at_position * 2 + position_worsened * 1
# Weights are a w04 design choice, never independently validated against
# outcomes — see w07 Section 4's "cost/value" note on treating baseline
# tiers as a nudge, not a hard prioritization.

# --- Queue ranking (w07) ----------------------------------------------------
# combined_score = oof_rf_score + BASELINE_NUDGE_WEIGHT * baseline_score
#
# Two alternatives were tried and rejected before landing here:
#   1. Tier-first sort (sort by rule tier, then by oof_rf_score only to
#      break ties within a tier). Rejected: the model's score could never
#      move a row out of its rule-assigned tier, which contradicts the
#      assignment's "the model gives the order" framing and ignores w06's
#      finding that RF beats the rule at K=50+.
#   2. 50/50 percentile blend (0.5 * oof_rf_score.rank(pct=True) +
#      0.5 * baseline_score.rank(pct=True)). Rejected: baseline_score only
#      has 4 discrete values, so percentile-ranking it reconstructs solid,
#      non-overlapping bands nearly identical to the tier-first sort —
#      the "blend" wasn't actually blending anything.
# The additive nudge below avoids both failure modes: the rule can never
# build a wall the model can't cross, but it does move close scores.
BASELINE_NUDGE_WEIGHT = 0.03

# --- Archetype / confidence thresholds (w07) --------------------------------
# All percentile-based off the scored population itself, not fixed numbers.
HIGH_SCORE_PERCENTILE = 0.90     # model_only_catch: top decile of oof_rf_score
                                    # among rows the rule didn't flag
LARGE_SWING_PERCENTILE = 0.90    # caution flag: top decile of |position_change|
                                    # across the whole population
LOW_DATA_PERCENTILE = 0.25       # confidence: bottom quartile of
                                    # total_impressions_full
BOUNDARY_MARGIN_PCT = 0.05       # confidence: how close to high_score_cut
                                    # counts as "too close to call"
# Tuned by hand (see w07) across a few combinations before settling here;
# 0.25/0.05 was chosen because it's the point where both thresholds still
# discriminate meaningfully — higher LOW_DATA_PERCENTILE flags most of the
# queue as low-confidence (useless as a filter); higher BOUNDARY_MARGIN_PCT
# collapses model_only_catch into "basically all low-confidence" (defeats
# the point of the archetype).


def assign_archetype(row: pd.Series, high_score_cut: float, large_swing_cut: float) -> tuple[int, str, str]:
    """Returns (priority_tier, archetype, action). See w07 Section 1.

    priority_tier is descriptive/audit-only — it does NOT drive the queue's
    sort order (that's combined_score, computed by the caller). Kept as its
    own column because value_counts()/groupby() on it are how w07's
    diagnostics were built.
    """
    if row["zero_clicks_at_position"] == 1 and row["position_worsened"] == 1:
        return 1, "zero_clicks_and_worsened", "Refresh content + overhaul title/meta"
    if row["zero_clicks_at_position"] == 1:
        return 2, "zero_clicks_only", "Overhaul title & meta"
    if row["position_worsened"] == 1:
        return 3, "position_worsened_only", "Refresh content"
    if row["baseline_score"] == 0 and row["oof_rf_score"] >= high_score_cut:
        caution = abs(row["position_change"]) >= large_swing_cut
        action = (
            "Flag for manual review (model-only signal — verify before acting)"
            if caution
            else "Flag for manual review"
        )
        return 4, "model_only_catch", action
    return 5, "no_flag", "Monitor"
    # A 5th archetype for "clicks but immediate bounce" was considered and
    # rejected (w07): the only candidate signal, session_rate, was tested
    # in w04 and rejected as confounded with impression volume. There is no
    # validated signal for this behavior anywhere in the pipeline — that's
    # a named gap (w07 Section 2), not something to invent a category for.


def assign_confidence(
    row: pd.Series,
    low_data_cut: float,
    high_score_cut: float,
    boundary_margin: float,
) -> str:
    """Returns 'low' or 'high'. See w07 Section 1 / the confidence tuning pass."""
    if row["total_impressions_full"] < low_data_cut:
        return "low"
    if row["archetype"] == "model_only_catch" and abs(row["oof_rf_score"] - high_score_cut) < boundary_margin:
        return "low"
    return "high"
