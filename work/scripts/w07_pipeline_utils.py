"""Shared configuration and helpers used by every script in this pipeline.

Paths, constants, and small reusable functions imported by 01_load_and_score.py, 02_build_queue.py,
03_validate_combined_score.py, 04_check_fold_representation.py, and run_all.py.

Every threshold and design choice below is a validated decision from w04-w07 — see each notebook for
the full reasoning. The one exception is calibrate_scores(), added during capstone validation to fix a
problem w06 found but never resolved. Details near that function.
"""
from __future__ import annotations

import json
import os
from getpass import getpass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- Paths -------------------------------------------------------------
# This file lives at work/scripts/w07_pipeline_utils.py, so parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "work"
PROCESSED_DIR = WORK_DIR / "data" / "processed"
OUTPUT_DIR = WORK_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

# work/**/*.csv is gitignored repo-wide. CSVs written by these scripts are local and regenerable.
# Only the JSON, charts, and report under OUTPUT_DIR are committed.


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
    # numpy scalar types (from .describe(), thresholds, etc.) aren't natively JSON-serializable.
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

    Lets run_all.py prompt once and share it with every subprocess step, or lets a script prompt for
    itself when run standalone. Never hardcode a token here — this file is committed to a public repo.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    return getpass("Paste your Hugging Face READ token: ")


# --- Data scope ----------------------------------------------------------
# Validated on one month only (w05/w06/w07). Hardcoded rather than parameterized, since nothing has
# been tested on another month. To generalize: accept a --month argument, and derive HALF_SPLIT_DATE /
# WEEK1_END_DATE from that month's day count instead of hardcoding them (not done here).
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
# zero_clicks_at_position_fh was built in w06 but excluded here — it only backed a w06 diagnostic.
# Named limitation (w07 Section 2): the model only sees click-through problems indirectly, via
# ctr_fh / log_clicks_fh. Also excluded: the w05 diagnostic LR and its interaction feature.

# --- Baseline rule (w04) ----------------------------------------------------
# baseline_score = zero_clicks_at_position * 2 + position_worsened * 1
# Weights are a w04 design choice, never independently validated against outcomes.

# --- Cross-fold calibration (added during capstone validation) -------------
# 01_load_and_score.py trains 5 separate RF models, one per fold, and pools their out-of-fold scores
# into one column. Two pooled fixes were tried (isotonic, then pooled Platt scaling) and both failed
# for the same reason: a single curve fit on raw score alone can't tell which fold a row came from, so
# it can't apply an opposite correction to two folds that need opposite corrections. Confirmed on the
# real population — barely moved the skew (see work/outputs/fold_representation_check.json).
#
# calibrate_scores() below fits Platt scaling separately per fold, using only that fold's own
# out-of-fold scores and labels, then applies it only to that fold's own rows. This uses fold identity
# directly, so it can fix bias that's specific to one fold. No leakage: each fold's scores are already
# out-of-fold, so calibrating against that fold's own labels doesn't touch its own training data.
def calibrate_scores(raw_scores: pd.Series, y: pd.Series, fold_id: pd.Series) -> np.ndarray:
    """Platt scaling fit separately per fold. See the note above this function for why."""
    raw = raw_scores.values
    labels = y.values
    folds = fold_id.values
    calibrated = np.zeros(len(raw))
    for fold in np.unique(folds):
        mask = folds == fold
        lr = LogisticRegression()
        lr.fit(raw[mask].reshape(-1, 1), labels[mask])
        calibrated[mask] = lr.predict_proba(raw[mask].reshape(-1, 1))[:, 1]
    return calibrated


# --- Queue ranking (w07, updated during capstone validation) ---------------
# combined_score = oof_rf_score + BASELINE_NUDGE_WEIGHT * baseline_score
#
# "oof_rf_score" here and downstream (assign_archetype, assign_confidence) means the CALIBRATED score
# — 02_build_queue.py swaps it in under that name after loading. Original score kept as
# oof_rf_score_raw for audit.
#
# Two other combination methods were tried and rejected:
#   1. Tier-first sort (rule tier, then oof_rf_score to break ties). Rejected — the model could never
#      move a row out of its rule tier, which ignores w06's finding that RF beats the rule at K=50+.
#   2. 50/50 percentile blend. Rejected — baseline_score only has 4 values, so this just rebuilds the
#      same tier walls as option 1.
# The additive nudge avoids both: the rule can't build a wall the model can't cross, but can still move
# close scores.
BASELINE_NUDGE_WEIGHT = 0.03

# --- Archetype / confidence thresholds (w07) --------------------------------
# All percentile-based off the scored population itself, not fixed numbers. Tuned by hand (w07) —
# 0.25/0.05 is where both thresholds still discriminate meaningfully without flagging most of the
# queue as low-confidence or collapsing model_only_catch into "basically all low-confidence."
HIGH_SCORE_PERCENTILE = 0.90    # model_only_catch: top decile of oof_rf_score among unflagged rows
LARGE_SWING_PERCENTILE = 0.90   # caution flag: top decile of |position_change|
LOW_DATA_PERCENTILE = 0.25      # confidence: bottom quartile of total_impressions_full
BOUNDARY_MARGIN_PCT = 0.05      # confidence: how close to high_score_cut counts as "too close to call"


def assign_archetype(row: pd.Series, high_score_cut: float, large_swing_cut: float) -> tuple[int, str, str]:
    """Returns (priority_tier, archetype, action). See w07 Section 1.

    priority_tier is descriptive only, not the sort order (that's combined_score). row["oof_rf_score"]
    is whatever the caller put there — the calibrated score, as set by 02_build_queue.py.
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
    # A 5th archetype ("clicks but immediate bounce") was considered and rejected in w07 — the only
    # candidate signal, session_rate, was rejected in w04 as confounded. Named gap, not an oversight.


def assign_confidence(
    row: pd.Series,
    low_data_cut: float,
    high_score_cut: float,
    boundary_margin: float,
) -> str:
    """Returns 'low' or 'high'. See w07 Section 1 and the confidence tuning pass.

    row["oof_rf_score"] is the calibrated score, same as in assign_archetype above.
    """
    if row["total_impressions_full"] < low_data_cut:
        return "low"
    if row["archetype"] == "model_only_catch" and abs(row["oof_rf_score"] - high_score_cut) < boundary_margin:
        return "low"
    return "high"
