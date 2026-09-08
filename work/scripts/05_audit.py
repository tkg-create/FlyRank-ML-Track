"""
Pipeline audit, as a plain script.

Same checks as work/notebooks/pipeline_audit.ipynb, minus the narration. 
Clones the repo fresh, snapshots what's currently committed, reruns the full pipeline, 
and compares the two — labels, features, archetypes, coverage, 
and precision@K all get independently re-derived rather than trusted from the pipeline's own output.

Usage:
    python 05_audit.py
    HF_TOKEN=hf_xxx python 05_audit.py   # skips the interactive prompt

Exits 0 if every hard check passes, 1 otherwise. 
WARN-level checks (documented nondeterminism, empirical findings, 
numbers with no committed source) never block the exit code — 
see the RESULTS table printed at the end for what those are.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from getpass import getpass

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

REPO_URL = "https://github.com/tkg-create/FlyRank-ML-Track.git"
REPO_DIR = "FlyRank-ML-Track"
COMMITTED_DIR = os.path.abspath("committed_outputs_snapshot")
KS = [20, 50, 100, 200]

RESULTS = []


def check(name, passed, detail="", severity="fail"):
    """Records one result instead of stopping at the first failure — the point is seeing every problem in one pass."""
    if passed:
        status = "PASS"
    else:
        status = "WARN" if severity == "warn" else "FAIL"
    RESULTS.append({"test": name, "status": status, "detail": detail})
    icon = {"PASS": "\u2705", "WARN": "\u26a0\ufe0f", "FAIL": "\u274c"}[status]
    print(f"{icon} {name}" + (f" \u2014 {detail}" if detail else ""))
    return passed


def close(a, b, tol):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Setup: fresh clone, snapshot committed outputs, rebuild everything.
# ---------------------------------------------------------------------------

def clone_repo():
    """Deletes any stale local copy first, so this always checks what's actually on GitHub right now."""
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    subprocess.run(["git", "clone", REPO_URL], check=True)
    os.chdir(REPO_DIR)
    assert os.path.isfile("work/scripts/01_load_and_score.py"), \
        "01_load_and_score.py not found \u2014 did the clone work?"
    print("Working dir:", os.getcwd())


def snapshot_committed_outputs():
    """Copies work/outputs/ before anything gets regenerated. Everything compared against 'committed' below reads from this snapshot, not the live repo state."""
    if os.path.isdir(COMMITTED_DIR):
        shutil.rmtree(COMMITTED_DIR)
    shutil.copytree("work/outputs", COMMITTED_DIR)
    print("Snapshotted committed work/outputs/ to", COMMITTED_DIR)


def get_hf_token():
    return os.environ.get("HF_TOKEN") or getpass("Paste your Hugging Face READ token: ")


def run_step(script):
    print(f"\n{'=' * 70}\n\u25b6 {script}\n{'=' * 70}", flush=True)
    process = subprocess.Popen(
        [sys.executable, "-u", f"work/scripts/{script}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    assert process.returncode == 0, f"{script} failed with exit code {process.returncode}"


def run_pipeline():
    """01+02 build the live model_df/queue_df and overwrite work/outputs/ with a fresh run. 
    03+04 are audit scripts in their own right, run so their JSON is fresh too."""
    os.environ["HF_TOKEN"] = get_hf_token()
    for script in [
        "01_load_and_score.py",
        "02_build_queue.py",
        "03_validate_deployed_score.py",
        "04_check_fold_representation.py",
    ]:
        run_step(script)


def load_committed(name):
    with open(f"{COMMITTED_DIR}/{name}") as f:
        return json.load(f)


def load_live(name):
    with open(f"work/outputs/{name}") as f:
        return json.load(f)


def load_data():
    sys.path.insert(0, "work/scripts")
    from w07_pipeline_utils import (
        FEATURE_COLS, HALF_SPLIT_DATE, HF_MONTH_PATH, N_FOLDS,
        HIGH_SCORE_PERCENTILE, LARGE_SWING_PERCENTILE, LOW_DATA_PERCENTILE,
        BOUNDARY_MARGIN_PCT,
    )
    model_df = pd.read_csv("work/data/processed/w07_scored_population.csv")
    queue_df = pd.read_csv("work/data/processed/w07_ranked_queue.csv")
    print(f"model_df: {model_df.shape}")
    print(f"queue_df: {queue_df.shape}")
    consts = dict(
        FEATURE_COLS=FEATURE_COLS, HALF_SPLIT_DATE=HALF_SPLIT_DATE,
        HF_MONTH_PATH=HF_MONTH_PATH, N_FOLDS=N_FOLDS,
        HIGH_SCORE_PERCENTILE=HIGH_SCORE_PERCENTILE,
        LARGE_SWING_PERCENTILE=LARGE_SWING_PERCENTILE,
        LOW_DATA_PERCENTILE=LOW_DATA_PERCENTILE,
        BOUNDARY_MARGIN_PCT=BOUNDARY_MARGIN_PCT,
    )
    return model_df, queue_df, consts


def query_raw_rows(consts):
    """Pulls raw daily rows straight from the warehouse, for every independent recompute below. 
    Needs its own HF secret — the one 01_load_and_score.py registers doesn't carry over to a separate connection."""
    print("Querying raw daily rows (this can take a minute)...")
    con = duckdb.connect()
    con.execute(f"CREATE SECRET (TYPE huggingface, TOKEN '{os.environ['HF_TOKEN']}')")
    raw = con.sql(f"""
        SELECT content_hash_id, report_date, gsc_impressions, gsc_clicks, gsc_avg_position
        FROM read_parquet('{consts["HF_MONTH_PATH"]}')
        WHERE gsc_data_available IS TRUE
    """).df()
    print(f"raw: {raw.shape}")
    return raw


# ---------------------------------------------------------------------------
# Section 1: shape sanity.
# ---------------------------------------------------------------------------

def check_shape_sanity(model_df, queue_df):
    check(
        "model_df has no duplicate content_hash_id rows",
        model_df["content_hash_id"].is_unique,
        detail=f"{model_df['content_hash_id'].duplicated().sum()} duplicate id(s)",
    )
    check(
        "queue_df has no duplicate content_hash_id rows",
        queue_df["content_hash_id"].is_unique,
        detail=f"{queue_df['content_hash_id'].duplicated().sum()} duplicate id(s)",
    )
    check(
        "queue_df is the same population as model_df (same content_hash_id set)",
        set(model_df["content_hash_id"]) == set(queue_df["content_hash_id"]),
        detail=f"model_df={len(model_df):,} rows, queue_df={len(queue_df):,} rows",
    )
    check(
        "no missing client_hash_id",
        model_df["client_hash_id"].notna().all(),
        detail=f"{model_df['client_hash_id'].isna().sum()} missing",
    )
    for col in ["oof_rf_score", "oof_rf_score_calibrated", "archetype", "coverage", "is_declining_proxy"]:
        source = queue_df if col in queue_df.columns else model_df
        check(f"no NaNs in {col}", source[col].notna().all())

    pop_size = len(model_df)
    check(
        "population size is on the order of hundreds of thousands (not the 'millions of pages' error caught earlier)",
        100_000 <= pop_size <= 999_999,
        detail=f"population_size={pop_size:,}",
        severity="warn",
    )


# ---------------------------------------------------------------------------
# Section 2: label lineage. Recomputes is_declining_proxy from raw rows in
# pandas, independent of 01's own SQL, to catch drift from the documented spec.
# ---------------------------------------------------------------------------

def check_label_lineage(model_df, raw, consts):
    fh_mask = raw["report_date"] < consts["HALF_SPLIT_DATE"]
    sh_mask = raw["report_date"] >= consts["HALF_SPLIT_DATE"]

    impr_fh = raw[fh_mask].groupby("content_hash_id")["gsc_impressions"].sum()
    impr_sh = raw[sh_mask].groupby("content_hash_id")["gsc_impressions"].sum()

    label_base = impr_fh[impr_fh > 0]
    label_indep = (impr_sh.reindex(label_base.index).fillna(0) < label_base).astype(int)
    label_indep.name = "is_declining_proxy_indep"

    compare = model_df.set_index("content_hash_id")[["is_declining_proxy"]].join(label_indep, how="inner")
    mismatches = compare[compare["is_declining_proxy"] != compare["is_declining_proxy_indep"]]

    check(
        "is_declining_proxy matches an independent pandas recompute, row for row",
        len(mismatches) == 0,
        detail=f"{len(mismatches):,} mismatches out of {len(compare):,} rows compared",
    )
    check(
        "label base rate is not degenerate (not ~0% or ~100%, which would mean a broken comparison)",
        0.05 < model_df["is_declining_proxy"].mean() < 0.95,
        detail=f"base rate = {model_df['is_declining_proxy'].mean():.3f}",
    )
    return fh_mask


# ---------------------------------------------------------------------------
# Section 3: feature leakage. Regression tests for the two historical leaks
# (total_impressions, eligible as features), plus independent recompute of
# every first-half feature.
# ---------------------------------------------------------------------------

def check_feature_leakage(model_df, raw, fh_mask, consts):
    check(
        "FEATURE_COLS does not include total_impressions_full or eligible (the leak found earlier)",
        not any(c in consts["FEATURE_COLS"] for c in ["total_impressions_full", "total_impressions", "eligible"]),
        detail=f"FEATURE_COLS={consts['FEATURE_COLS']}",
    )
    check(
        "'total_impressions' (bare) is not present \u2014 only the qualified _fh/_full variants",
        "total_impressions" not in model_df.columns,
        detail="'total_impressions' would be ambiguous with total_impressions_fh/_full and was the literal name of the original leak",
    )

    fh = raw[fh_mask].copy()
    impr_fh_sum = fh.groupby("content_hash_id")["gsc_impressions"].sum()
    clicks_fh_sum = fh.groupby("content_hash_id")["gsc_clicks"].sum()
    pos_fh_mean = fh[fh["gsc_avg_position"] > 0].groupby("content_hash_id")["gsc_avg_position"].mean()

    feat_indep = pd.DataFrame({
        "log_impressions_fh_indep": np.log1p(impr_fh_sum),
        "log_clicks_fh_indep": np.log1p(clicks_fh_sum),
        "avg_position_fh_indep": pos_fh_mean,
    })
    feat_indep["ctr_fh_indep"] = clicks_fh_sum / impr_fh_sum

    feat_compare = model_df.set_index("content_hash_id")[
        ["avg_position_fh", "log_impressions_fh", "log_clicks_fh", "ctr_fh"]
    ].join(feat_indep, how="inner")

    for col in ["avg_position_fh", "log_impressions_fh", "log_clicks_fh", "ctr_fh"]:
        diffs = (feat_compare[col] - feat_compare[f"{col}_indep"]).abs()
        check(
            f"{col} matches independent first-half-only recompute",
            (diffs < 1e-6).all(),
            detail=f"max abs diff = {diffs.max():.6g}",
        )

    # Catches an accidental revert to full-month aggregation: if that happened,
    # avg_position_fh would equal the full-month average, not just resemble it.
    full_pos_mean = raw[raw["gsc_avg_position"] > 0].groupby("content_hash_id")["gsc_avg_position"].mean()
    full_vs_fh = model_df.set_index("content_hash_id")["avg_position_fh"].to_frame().join(
        full_pos_mean.rename("avg_position_full_check"), how="inner"
    )
    share_identical = (full_vs_fh["avg_position_fh"].round(6) == full_vs_fh["avg_position_full_check"].round(6)).mean()
    check(
        "avg_position_fh is NOT just the full-month average in disguise",
        share_identical < 0.5,
        detail=f"{share_identical:.1%} of rows have fh == full-month position (expect low, not ~100%)",
    )


# ---------------------------------------------------------------------------
# Section 4: split integrity. Confirms GroupKFold actually grouped by client.
# ---------------------------------------------------------------------------

def check_split_integrity(model_df):
    per_client_folds = model_df.groupby("client_hash_id")["fold_id"].nunique()
    check(
        "every client_hash_id appears in exactly one fold (GroupKFold is doing its job)",
        (per_client_folds == 1).all(),
        detail=f"{(per_client_folds != 1).sum()} client(s) split across folds",
    )
    check(
        "every row got a fold_id (none left unscored)",
        (model_df["fold_id"] > 0).all(),
        detail=f"{(model_df['fold_id'] <= 0).sum()} rows without a valid fold_id",
    )
    fold_share = model_df["fold_id"].value_counts(normalize=True)
    check(
        "fold sizes are roughly balanced (~20% each)",
        ((fold_share - 0.2).abs() < 0.03).all(),
        detail=fold_share.round(3).to_dict(),
    )


# ---------------------------------------------------------------------------
# Section 5: calibration. Percentile rank must preserve within-fold order and
# spread the top-K queue roughly evenly across folds.
# ---------------------------------------------------------------------------

def check_calibration(model_df):
    rank_breaks = 0
    for fold in sorted(model_df["fold_id"].unique()):
        sub = model_df[model_df["fold_id"] == fold]
        rho, _ = spearmanr(sub["oof_rf_score"], sub["oof_rf_score_calibrated"])
        if rho < 0.999:
            rank_breaks += 1
    check(
        "oof_rf_score_calibrated preserves within-fold rank order of oof_rf_score exactly",
        rank_breaks == 0,
        detail=f"{rank_breaks} fold(s) with Spearman rho < 0.999 between raw and calibrated",
    )

    cal_bounds = model_df.groupby("fold_id")["oof_rf_score_calibrated"].agg(["min", "max"])
    check(
        "calibrated score spans ~0 to 1 within every fold",
        ((cal_bounds["min"] < 0.05) & (cal_bounds["max"] > 0.95)).all(),
        detail=cal_bounds.round(3).to_dict(orient="index"),
    )

    for k in KS:
        top_k = model_df.sort_values("oof_rf_score_calibrated", ascending=False).head(k)
        max_share = top_k["fold_id"].value_counts(normalize=True).max()
        check(
            f"K={k}: no single fold dominates the top-K queue (calibrated score)",
            max_share <= 0.30,
            detail=f"most-represented fold at {max_share:.0%} (expect ~20%)",
        )


# ---------------------------------------------------------------------------
# Section 6: archetype and coverage logic, re-derived independently from the
# written spec rather than imported from the pipeline's own function.
# ---------------------------------------------------------------------------

def check_archetype_and_coverage(model_df, queue_df, consts):
    high_score_cut = model_df.loc[model_df["baseline_score"] == 0, "oof_rf_score_calibrated"].quantile(
        consts["HIGH_SCORE_PERCENTILE"])
    low_data_cut = model_df["total_impressions_full"].quantile(consts["LOW_DATA_PERCENTILE"])
    boundary_margin = consts["BOUNDARY_MARGIN_PCT"] * high_score_cut

    zc = model_df["zero_clicks_at_position"] == 1
    pw = model_df["position_worsened"] == 1
    model_only_mask = (
        (~zc) & (~pw)
        & (model_df["baseline_score"] == 0)
        & (model_df["oof_rf_score_calibrated"] >= high_score_cut)
    )

    archetype_indep_arr = np.select(
        [zc & pw, zc & ~pw, (~zc) & pw, model_only_mask],
        ["zero_clicks_and_worsened", "zero_clicks_only", "position_worsened_only", "model_only_catch"],
        default="no_flag",
    )
    archetype_indep = pd.Series(archetype_indep_arr, index=model_df["content_hash_id"], name="archetype_indep")

    compare = queue_df.set_index("content_hash_id")[["archetype"]].join(archetype_indep, how="inner")
    archetype_mismatches = (compare["archetype"] != compare["archetype_indep"]).sum()
    check(
        "archetype matches an independently re-derived assignment, row for row",
        archetype_mismatches == 0,
        detail=f"{archetype_mismatches:,} mismatches out of {len(compare):,} rows",
    )
    check(
        "every row gets exactly one of the 5 known archetypes (no unexpected label)",
        set(queue_df["archetype"].unique()) <= {
            "zero_clicks_and_worsened", "zero_clicks_only", "position_worsened_only",
            "model_only_catch", "no_flag",
        },
        detail=f"observed: {sorted(queue_df['archetype'].unique())}",
    )

    low_data_mask = model_df["total_impressions_full"] < low_data_cut
    near_boundary_mask = (model_df["oof_rf_score_calibrated"] - high_score_cut).abs() < boundary_margin
    coverage_indep_arr = np.where(
        low_data_mask, "low",
        np.where(model_only_mask & near_boundary_mask, "low", "high"),
    )
    coverage_indep = pd.Series(coverage_indep_arr, index=model_df["content_hash_id"], name="coverage_indep")

    compare_cov = queue_df.set_index("content_hash_id")[["coverage"]].join(coverage_indep, how="inner")
    coverage_mismatches = (compare_cov["coverage"] != compare_cov["coverage_indep"]).sum()
    check(
        "coverage matches an independently re-derived assignment, row for row",
        coverage_mismatches == 0,
        detail=f"{coverage_mismatches:,} mismatches out of {len(compare_cov):,} rows",
    )


# ---------------------------------------------------------------------------
# Section 7: coverage-honesty regression test, for the finding behind the
# confidence-to-coverage rename.
# ---------------------------------------------------------------------------

def check_coverage_honesty(queue_df):
    outcome_by_cov = queue_df.groupby(["archetype", "coverage"])["is_declining_proxy"].mean().unstack()
    print(outcome_by_cov.round(3))
    if {"low", "high"}.issubset(outcome_by_cov.columns):
        still_inverted = bool((outcome_by_cov["low"] >= outcome_by_cov["high"]).all())
        check(
            "low-coverage rows still show >= decline rate than high-coverage rows in every archetype",
            still_inverted,
            detail="matches the documented finding behind the confidence\u2192coverage rename" if still_inverted
                   else "the inversion no longer holds in every archetype \u2014 recheck the Limitations claim",
            severity="warn",
        )


# ---------------------------------------------------------------------------
# Section 8: baseline rule reproduction.
# ---------------------------------------------------------------------------

def check_baseline_rule(model_df):
    baseline_indep = model_df["zero_clicks_at_position"] * 2 + model_df["position_worsened"]
    check(
        "baseline_score matches zero_clicks_at_position*2 + position_worsened exactly",
        (model_df["baseline_score"] == baseline_indep).all(),
        detail=f"{(model_df['baseline_score'] != baseline_indep).sum()} mismatches",
    )


# ---------------------------------------------------------------------------
# Section 9: precision@K, recomputed with a fresh GroupKFold reconstruction
# and compared against both 03's fresh output and the committed (locked) run.
# ---------------------------------------------------------------------------

def check_precision_at_k(model_df, consts):
    def precision_at_k(labels_sorted, k):
        return float(np.asarray(labels_sorted)[:k].mean())

    X = model_df[consts["FEATURE_COLS"]].astype(float)
    y = model_df["is_declining_proxy"].astype(int)
    groups = model_df["client_hash_id"]
    gkf = GroupKFold(n_splits=consts["N_FOLDS"])

    records = []
    for fold_num, (_, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        fold_rows = model_df.iloc[test_idx]
        order = fold_rows["oof_rf_score_calibrated"].sort_values(ascending=False).index
        ranked_labels = y.loc[order].values
        for k in KS:
            records.append({"fold": fold_num, "k": k, "precision_indep": precision_at_k(ranked_labels, k)})

    precision_indep = pd.DataFrame(records).groupby("k")["precision_indep"].mean().round(3)
    print(precision_indep)

    live_precision = load_live("capstone_precision_at_k.json")
    committed_precision = load_committed("capstone_precision_at_k.json")

    for k in KS:
        live_mean = live_precision["summary_mean_std"][str(k)]["oof_rf_score_calibrated"]["mean"]
        check(
            f"K={k}: independent precision@K recompute matches 03's own fresh output",
            close(precision_indep[k], live_mean, tol=0.01),
            detail=f"independent={precision_indep[k]:.3f}, script={live_mean:.3f}",
        )
    for k in KS:
        committed_mean = committed_precision["summary_mean_std"][str(k)]["oof_rf_score_calibrated"]["mean"]
        live_mean = live_precision["summary_mean_std"][str(k)]["oof_rf_score_calibrated"]["mean"]
        check(
            f"K={k}: fresh precision@K hasn't drifted far from the committed (locked) result",
            close(committed_mean, live_mean, tol=0.03),
            detail=f"committed={committed_mean:.3f}, fresh={live_mean:.3f} "
                   "(small drift expected from DuckDB aggregation-order nondeterminism)",
            severity="warn",
        )
    return live_precision


# ---------------------------------------------------------------------------
# Section 10: cross-file consistency, committed vs. freshly regenerated.
# ---------------------------------------------------------------------------

def check_cross_file_consistency():
    committed_metrics = load_committed("w07_metrics.json")
    live_metrics = load_live("w07_metrics.json")

    check(
        "population_size: committed vs. fresh",
        close(committed_metrics["population_size"], live_metrics["population_size"],
              tol=live_metrics["population_size"] * 0.02),
        detail=f"committed={committed_metrics['population_size']:,}, fresh={live_metrics['population_size']:,}",
        severity="warn",
    )
    for archetype in set(committed_metrics["archetype_counts"]) | set(live_metrics["archetype_counts"]):
        c = committed_metrics["archetype_counts"].get(archetype, 0)
        l = live_metrics["archetype_counts"].get(archetype, 0)
        check(
            f"archetype_counts[{archetype}]: committed vs. fresh",
            close(c, l, tol=max(50, l * 0.03)),
            detail=f"committed={c:,}, fresh={l:,}",
            severity="warn",
        )
    for tier in ["high", "low"]:
        c = committed_metrics["coverage_split"].get(tier, 0)
        l = live_metrics["coverage_split"].get(tier, 0)
        check(
            f"coverage_split[{tier}]: committed vs. fresh",
            close(c, l, tol=max(50, l * 0.03)),
            detail=f"committed={c:,}, fresh={l:,}",
            severity="warn",
        )

    check(
    "base_rate: committed vs. fresh",
    close(committed_metrics["base_rate"], live_metrics["base_rate"], tol=0.005),
    detail=f"committed={committed_metrics['base_rate']}, fresh={live_metrics['base_rate']}",
    severity="warn",
    )

    committed_sentinel = load_committed("sentinel_fill_check.json")
    live_sentinel = load_live("sentinel_fill_check.json")
    check(
        "sentinel_fill_check.json: no_trend_share committed vs. fresh",
        close(committed_sentinel["no_trend_share"], live_sentinel["no_trend_share"], tol=0.02),
        detail=f"committed={committed_sentinel['no_trend_share']:.3f}, fresh={live_sentinel['no_trend_share']:.3f}",
        severity="warn",
    )
    check(
        "sentinel-fill gap (has_position_trend == 0) is still in the ballpark of the documented ~21%",
        0.10 <= live_sentinel["no_trend_share"] <= 0.35,
        detail=f"fresh no_trend_share={live_sentinel['no_trend_share']:.3f}",
        severity="warn",
    )

    expected_files = [
        "work/outputs/feature_importance.json",
        "work/outputs/sentinel_fill_check.json",
        "work/outputs/queue_diagnostics.json",
        "work/outputs/fold_representation_check.json",
        "work/outputs/capstone_precision_at_k.json",
        "work/outputs/w07_metrics.json",
        "work/outputs/w07_report.md",
        "work/outputs/charts/archetype_mix.svg",
        "work/outputs/charts/coverage_mix.svg",
        "work/outputs/charts/feature_importance.svg",
        "work/outputs/charts/precision_at_k.svg",
        "work/outputs/charts/score_distribution_by_archetype.svg",
    ]
    missing = [f for f in expected_files if not os.path.isfile(f)]
    check("all expected output files exist after a full run", len(missing) == 0, detail=f"missing: {missing}")
    check(
        "stale charts/confidence_mix.svg has not reappeared",
        not os.path.isfile("work/outputs/charts/confidence_mix.svg"),
    )


# ---------------------------------------------------------------------------
# Section 11: report cross-check. Pulls hardcoded numbers out of the live
# capstone_report.md and compares them against the JSON they claim to cite.
# ---------------------------------------------------------------------------

def check_report_claims(live_precision):
    with open("work/capstone_report.md") as f:
        report_text = f.read()
    metrics = load_live("w07_metrics.json")
    importances = load_live("feature_importance.json")
    sentinel = load_live("sentinel_fill_check.json")
    with open("work/scripts/01_load_and_score.py") as f:
        pipeline_src = f.read()

    def extract(pattern, text, group=1):
        m = re.search(pattern, text)
        return m.group(group) if m else None

    def close_str(claimed, actual, tol=0.006):
        return claimed is not None and abs(float(claimed) - actual) <= tol

    m = re.search(
        r"([\d.]+) at K=20, ([\d.]+) at K=50, ([\d.]+) at K=100, ([\d.]+) at K=200\.\s*"
        r"The model's corresponding numbers are ([\d.]+), ([\d.]+), ([\d.]+), and ([\d.]+)",
        report_text,
    )
    if m:
        claimed_baseline = dict(zip(KS, m.groups()[:4]))
        claimed_model = dict(zip(KS, m.groups()[4:]))
        for k in KS:
            actual_b = live_precision["summary_mean_std"][str(k)]["baseline_rule"]["mean"]
            actual_m = live_precision["summary_mean_std"][str(k)]["oof_rf_score_calibrated"]["mean"]
            check(f"\u00a73: baseline precision@{k} in report matches capstone_precision_at_k.json",
                  close_str(claimed_baseline[k], actual_b),
                  detail=f"report says {claimed_baseline[k]}, file says {actual_b}")
            check(f"\u00a73: model precision@{k} in report matches capstone_precision_at_k.json",
                  close_str(claimed_model[k], actual_m),
                  detail=f"report says {claimed_model[k]}, file says {actual_m}")
    else:
        check("\u00a73: baseline/model precision@K sentence found in report", False,
              detail="expected phrasing not found \u2014 report wording may have changed")

    words_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    m = re.search(r"the rule wins (\w+) of five folds; at K=200 it wins (\w+)", report_text)
    if m:
        claimed_k20 = words_to_num.get(m.group(1))
        claimed_k200 = words_to_num.get(m.group(2))
        for k, claimed in [(20, claimed_k20), (200, claimed_k200)]:
            fold_rows = [r for r in live_precision["fold_level"] if r["k"] == k]
            actual_rule_wins = sum(1 for r in fold_rows if r["baseline_rule"] > r["oof_rf_score_calibrated"])
            check(f"\u00a75: rule's fold-win count at K={k} matches fold_level data",
                  claimed == actual_rule_wins,
                  detail=f"report says rule wins {claimed}, fold_level data says {actual_rule_wins}")
    else:
        check("\u00a75: fold-win-count sentence found in report", False,
              detail="expected phrasing not found \u2014 report wording may have changed")

    for k in [50, 100]:
        fold_rows = [r for r in live_precision["fold_level"] if r["k"] == k]
        model_wins = sum(1 for r in fold_rows if r["oof_rf_score_calibrated"] > r["baseline_rule"])
        check(f"\u00a73/\u00a75: K={k} really is a clean 5/5 sweep, as both sections claim",
              model_wins == 5, detail=f"model wins {model_wins}/5 folds at K={k}")

    m = re.search(r"(\d+) trees, depth (\d+), a (\d+)-row minimum per leaf", report_text)
    rf_call = extract(r"RandomForestClassifier\(([^)]*)\)", pipeline_src)
    if m and rf_call:
        claimed_trees, claimed_depth, claimed_leaf = m.groups()
        rf_call_clean = rf_call.replace(" ", "")
        check("\u00a74: n_estimators matches RandomForestClassifier(...) in 01_load_and_score.py",
              f"n_estimators={claimed_trees}" in rf_call_clean)
        check("\u00a74: max_depth matches RandomForestClassifier(...) in 01_load_and_score.py",
              f"max_depth={claimed_depth}" in rf_call_clean)
        check("\u00a74: min_samples_leaf matches RandomForestClassifier(...) in 01_load_and_score.py",
              f"min_samples_leaf={claimed_leaf}" in rf_call_clean)
    else:
        check("\u00a74: hyperparameter sentence and/or RandomForestClassifier(...) call found", False)

    claimed_order = ["log_impressions_fh", "avg_position_fh", "ctr_fh",
                      "position_change", "has_position_trend", "log_clicks_fh"]
    actual_order = [k for k, _ in sorted(importances["mean_importance"].items(), key=lambda x: -x[1])]
    check("\u00a76: feature importance ranking described in prose matches feature_importance.json",
          claimed_order == actual_order,
          detail=f"report implies {claimed_order}, file order is {actual_order}")

    caution = metrics["model_only_catch_caution"]
    for m in re.finditer(r"fires on (?:only )?(\d+) of (?:the )?([\d,]+)", report_text):
        claimed_flagged, claimed_total = int(m.group(1)), int(m.group(2).replace(",", ""))
        check("\u00a75/\u00a77: caution-flag count matches w07_metrics.json",
              claimed_flagged == caution["flagged"] and claimed_total == caution["total"],
              detail=f"report says {claimed_flagged} of {claimed_total}, "
                     f"file says {caution['flagged']} of {caution['total']}")

    m = re.search(r"largest group at (\d+\.?\d*) percent", report_text)
    actual_pct = round(100 * metrics["archetype_counts"]["no_flag"] / metrics["population_size"], 1)
    check("\u00a77: no_flag population share matches w07_metrics.json",
          m is not None and abs(float(m.group(1)) - actual_pct) <= 0.1,
          detail=f"report says {m.group(1) if m else None}%, computed {actual_pct}%")

    m = re.search(r"score highest on average of any archetype, (0\.\d+)", report_text)
    actual_score = metrics["rule_agreement_mean_scores"]["zero_clicks_and_worsened"]
    check("\u00a77: zero_clicks_and_worsened mean score matches w07_metrics.json",
          close_str(m.group(1) if m else None, actual_score),
          detail=f"report says {m.group(1) if m else None}, file says {actual_score}")

    check('\u00a77: "about a quarter" of the queue is low-coverage \u2014 consistent with coverage_low_pct',
          0.20 <= metrics["coverage_low_pct"] <= 0.30,
          detail=f"coverage_low_pct = {metrics['coverage_low_pct']}")

    ratio = metrics["archetype_counts"]["position_worsened_only"] / metrics["archetype_counts"]["zero_clicks_and_worsened"]
    check('\u00a76: "roughly five times as populous" (position_worsened_only vs. zero_clicks_and_worsened)',
          4.5 <= ratio <= 5.5, detail=f"actual ratio = {ratio:.2f}")

    check('\u00a76: "about a fifth" of pages have no rule signal \u2014 consistent with sentinel_fill_check.json',
          0.15 <= sentinel["no_trend_share"] <= 0.25,
          detail=f"no_trend_share = {sentinel['no_trend_share']}")

    # Both numbers below have no committed JSON to check against — flagged rather
    # than silently skipped.
    check('\u00a76: session-engagement split (35.4% vs. 43.9%) has no committed JSON source to verify against',
          False, detail="sourced from one-off exploration in an earlier notebook, never saved to a file",
          severity="warn")
    check('\u00a73: first-half-only baseline diagnostic (0.06\u20130.15 drop) has no committed JSON source to verify against',
          False, detail="sourced from a throwaway diagnostic cell earlier in this project, never saved to a file",
          severity="warn")


# ---------------------------------------------------------------------------
# Summary. Exit code is what a CI system or a person running this once cares
# about — nonzero if anything hard failed.
# ---------------------------------------------------------------------------

def print_summary_and_exit():
    summary = pd.DataFrame(RESULTS)
    n_fail = (summary["status"] == "FAIL").sum()
    n_warn = (summary["status"] == "WARN").sum()
    n_pass = (summary["status"] == "PASS").sum()

    print(summary.to_string(index=False))
    print(f"\n{len(summary)} checks \u2014 {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")

    if n_fail:
        print(f"\n{n_fail} check(s) FAILED \u2014 see the table above. Do not trust the "
              "capstone notebook's numbers or the paper until these are resolved.")
        sys.exit(1)
    print("\nNo hard failures. WARNs above are worth a skim before finalizing the "
          "paper, but nothing here blocks moving forward.")
    sys.exit(0)


def main():
    clone_repo()
    snapshot_committed_outputs()
    run_pipeline()

    model_df, queue_df, consts = load_data()
    raw = query_raw_rows(consts)

    check_shape_sanity(model_df, queue_df)
    fh_mask = check_label_lineage(model_df, raw, consts)
    check_feature_leakage(model_df, raw, fh_mask, consts)
    check_split_integrity(model_df)
    check_calibration(model_df)
    check_archetype_and_coverage(model_df, queue_df, consts)
    check_coverage_honesty(queue_df)
    check_baseline_rule(model_df)
    live_precision = check_precision_at_k(model_df, consts)
    check_cross_file_consistency()
    check_report_claims(live_precision)

    print_summary_and_exit()


if __name__ == "__main__":
    main()
