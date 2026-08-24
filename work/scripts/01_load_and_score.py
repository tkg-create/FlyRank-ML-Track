"""Load a month of warehouse data, build features, score with the baseline
rule and an out-of-fold random forest.

Consolidates: w04's baseline rule, w05's model/feature design, w06's
leakage-safe position trend and GroupKFold validation approach, and w07's
Cell A (which itself rebuilt w06's OOF scores at runtime, since work/**/*.csv
is gitignored and nothing from w06 was ever persisted).

Requires an HF_TOKEN — set via getpass() here, or pre-set in the environment
by a caller (e.g. run_all.py) so every step in the pipeline shares one prompt.

Usage:
    python work/scripts/01_load_and_score.py
    python work/scripts/01_load_and_score.py --output work/data/processed/custom.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w07_pipeline_utils import (  # noqa: E402
    FEATURE_COLS,
    HALF_SPLIT_DATE,
    HF_MONTH_PATH,
    N_FOLDS,
    PROCESSED_DIR,
    RANDOM_STATE,
    WEEK1_END_DATE,
    display_path,
    ensure_dirs,
    get_hf_token,
    write_json,
)

DEFAULT_OUTPUT = PROCESSED_DIR / "w07_scored_population.csv"
DEFAULT_METADATA = PROCESSED_DIR / "w07_scoring_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load, feature-build, and OOF-score the content population.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    return parser.parse_args()


def load_model_df(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    month_path = HF_MONTH_PATH

    print("  [1/6] Querying label (impressions first half vs second half)...", flush=True)
    label_df = con.sql(f"""
        WITH halves AS (
            SELECT
                content_hash_id,
                SUM(CASE WHEN report_date < '{HALF_SPLIT_DATE}' THEN gsc_impressions ELSE 0 END) AS impr_first_half,
                SUM(CASE WHEN report_date >= '{HALF_SPLIT_DATE}' THEN gsc_impressions ELSE 0 END) AS impr_second_half
            FROM read_parquet('{month_path}')
            WHERE gsc_data_available IS TRUE
            GROUP BY content_hash_id
        )
        SELECT
            content_hash_id,
            CASE WHEN impr_second_half < impr_first_half THEN 1 ELSE 0 END AS is_declining_proxy
        FROM halves
        WHERE impr_first_half > 0
    """).df()
    print(f"        -> {len(label_df):,} rows", flush=True)

    print("  [2/6] Querying full-month position/click signal (baseline rule)...", flush=True)
    pf = con.sql(f"""
        SELECT
            content_hash_id,
            AVG(CASE WHEN gsc_avg_position > 0 THEN gsc_avg_position END) AS avg_position_full,
            SUM(gsc_impressions) AS total_impressions_full,
            SUM(gsc_clicks) AS total_clicks_full
        FROM read_parquet('{month_path}')
        WHERE gsc_data_available IS TRUE
        GROUP BY content_hash_id
        HAVING SUM(gsc_impressions) > 0
    """).df()
    pf_valid = pf.dropna(subset=["avg_position_full"]).copy()
    pf_valid["eligible"] = pf_valid["total_impressions_full"] >= 10
    pf_valid["zero_clicks_at_position"] = (
        (pf_valid["avg_position_full"] <= 10) & (pf_valid["total_clicks_full"] == 0) & (pf_valid["eligible"])
    ).astype(int)
    print(f"        -> {len(pf_valid):,} rows", flush=True)

    print("  [3/6] Querying first-half-only features (model training data)...", flush=True)
    pf_fh = con.sql(f"""
        SELECT
            content_hash_id,
            AVG(CASE WHEN gsc_avg_position > 0 THEN gsc_avg_position END) AS avg_position_fh,
            SUM(gsc_impressions) AS total_impressions_fh,
            SUM(gsc_clicks) AS total_clicks_fh
        FROM read_parquet('{month_path}')
        WHERE gsc_data_available IS TRUE AND report_date < '{HALF_SPLIT_DATE}'
        GROUP BY content_hash_id
        HAVING SUM(gsc_impressions) > 0
    """).df()
    pf_fh = pf_fh.dropna(subset=["avg_position_fh"]).copy()
    pf_fh["ctr_fh"] = pf_fh["total_clicks_fh"] / pf_fh["total_impressions_fh"]
    print(f"        -> {len(pf_fh):,} rows", flush=True)

    print("  [4/6] Querying leakage-safe position trend (week 1 vs week 2)...", flush=True)
    postrend = con.sql(f"""
        SELECT
            content_hash_id,
            AVG(CASE WHEN report_date < '{WEEK1_END_DATE}' AND gsc_avg_position > 0 THEN gsc_avg_position END) AS avg_position_wk1,
            AVG(CASE WHEN report_date >= '{WEEK1_END_DATE}' AND report_date < '{HALF_SPLIT_DATE}' AND gsc_avg_position > 0 THEN gsc_avg_position END) AS avg_position_wk2
        FROM read_parquet('{month_path}')
        WHERE gsc_data_available IS TRUE AND report_date < '{HALF_SPLIT_DATE}'
        GROUP BY content_hash_id
    """).df()
    postrend = postrend.dropna(subset=["avg_position_wk1", "avg_position_wk2"])
    postrend["position_change"] = postrend["avg_position_wk2"] - postrend["avg_position_wk1"]
    postrend["position_worsened"] = (postrend["position_change"] > 0).astype(int)
    postrend_eligible = postrend.merge(pf_valid[["content_hash_id", "eligible"]], on="content_hash_id", how="left")
    postrend_eligible = postrend_eligible[postrend_eligible["eligible"] == True].copy()  # noqa: E712
    print(f"        -> {len(postrend_eligible):,} rows", flush=True)

    print("  [5/6] Querying client map (grouping key only, never a feature)...", flush=True)
    client_map = con.sql(f"""
        SELECT DISTINCT content_hash_id, client_hash_id
        FROM read_parquet('{month_path}')
    """).df()
    print(f"        -> {len(client_map):,} rows", flush=True)

    print("  [6/6] Merging into model_df...", flush=True)
    model_df = pf_fh.merge(
        pf_valid[["content_hash_id", "total_impressions_full", "eligible", "zero_clicks_at_position"]],
        on="content_hash_id", how="inner",
    )
    model_df = model_df.merge(
        postrend_eligible[["content_hash_id", "position_change", "position_worsened"]],
        on="content_hash_id", how="left",
    )
    model_df["has_position_trend"] = model_df["position_change"].notna().astype(int)
    model_df["position_change"] = model_df["position_change"].fillna(0)
    model_df["position_worsened"] = model_df["position_worsened"].fillna(0).astype(int)
    model_df = model_df.merge(client_map, on="content_hash_id", how="left").dropna(subset=["client_hash_id"])
    model_df = model_df.merge(label_df, on="content_hash_id", how="inner").sort_values("content_hash_id").reset_index(drop=True)

    model_df["log_impressions_fh"] = np.log1p(model_df["total_impressions_fh"])
    model_df["log_clicks_fh"] = np.log1p(model_df["total_clicks_fh"])
    model_df["baseline_score"] = model_df["zero_clicks_at_position"] * 2 + model_df["position_worsened"]

    return model_df


def add_oof_scores(model_df: pd.DataFrame) -> pd.DataFrame:
    X = model_df[FEATURE_COLS].astype(float)
    y = model_df["is_declining_proxy"].astype(int)
    groups = model_df["client_hash_id"]

    oof_rf = np.full(len(model_df), np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_num, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"  Fold {fold_num}/{N_FOLDS}: fitting on {len(train_idx):,} rows, scoring {len(test_idx):,}...", flush=True)
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_rf[test_idx] = rf.predict_proba(X.iloc[test_idx])[:, 1]
        print(f"  Fold {fold_num}/{N_FOLDS}: done", flush=True)

    model_df = model_df.copy()
    model_df["oof_rf_score"] = oof_rf
    return model_df


def main() -> None:
    args = parse_args()
    ensure_dirs()

    con = duckdb.connect()
    hf_token = get_hf_token()
    con.execute(f"CREATE SECRET (TYPE huggingface, TOKEN '{hf_token}')")

    print("Loading and building features from the warehouse...")
    model_df = load_model_df(con)
    print(f"model_df: {model_df.shape}, base rate: {model_df['is_declining_proxy'].mean():.3f}")

    print(f"Running {N_FOLDS}-fold GroupKFold OOF scoring (random_state={RANDOM_STATE})...")
    model_df = add_oof_scores(model_df)
    coverage = model_df["oof_rf_score"].notna().sum()
    print(f"OOF coverage: {coverage} / {len(model_df)} rows")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_path, index=False)

    write_json(Path(args.metadata), {
        "population_size": int(len(model_df)),
        "oof_coverage": int(coverage),
        "base_rate": float(model_df["is_declining_proxy"].mean()),
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "feature_cols": FEATURE_COLS,
        "output": display_path(output_path),
    })

    print(f"Wrote {output_path} (local only — gitignored, not committed)")


if __name__ == "__main__":
    main()
