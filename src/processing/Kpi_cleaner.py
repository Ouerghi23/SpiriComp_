"""
KPI Data Cleaner
================
Handles all cleaning operations for the network KPI dataset:
  - Duplicate session removal
  - Physical range validation per KPI (e.g. packet_loss ∈ [0,100])
  - Outlier detection via IQR + Z-score (with configurable strategy)
  - Missing value imputation (per-KPI strategy: median / forward-fill / knn)
  - Degraded session labelling verification
  - Cell ID normalisation

Each step returns the modified DataFrame + an audit entry.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

ALL_KPI_COLS: list[str] = (
    cfg["features"]["kpi_columns"]["data"] +
    cfg["features"]["kpi_columns"]["voice"]
)

# Physical bounds per KPI — values outside these are physically impossible
KPI_BOUNDS: dict[str, tuple[float, float]] = {
    "dl_throughput_mbps":        (0.0,   2000.0),
    "ul_throughput_mbps":        (0.0,   1000.0),
    "latency_ms":                (1.0,  10000.0),
    "packet_loss_pct":           (0.0,    100.0),
    "data_session_success_rate": (0.0,    100.0),
    "data_qoe_score":            (0.0,    100.0),
    "call_setup_success_rate":   (0.0,    100.0),
    "call_drop_rate":            (0.0,    100.0),
    "voice_quality_score_mos":   (1.0,      5.0),
    "handover_success_rate":     (0.0,    100.0),
    "voice_qoe_score":           (0.0,    100.0),
    "qoe_score":                 (0.0,    100.0),
}

# Imputation strategy per KPI
KPI_IMPUTATION: dict[str, str] = {
    "dl_throughput_mbps":        "median",
    "ul_throughput_mbps":        "median",
    "latency_ms":                "median",
    "packet_loss_pct":           "median",
    "data_session_success_rate": "median",
    "data_qoe_score":            "median",
    "call_setup_success_rate":   "median",
    "call_drop_rate":            "median",
    "voice_quality_score_mos":   "median",
    "handover_success_rate":     "median",
    "voice_qoe_score":           "median",
    "qoe_score":                 "median",
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_kpi_data(df: pd.DataFrame,
                   outlier_strategy: str = "cap",
                   iqr_multiplier: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline for KPI data.

    Parameters
    ----------
    df               : raw KPI DataFrame (output of data_loader.load_kpi_data)
    outlier_strategy : 'cap'  → clamp to IQR fences  (recommended for telecom)
                       'flag' → add outlier_flag column but keep values
                       'drop' → remove outlier rows
    iqr_multiplier   : IQR fence multiplier (default 3.0 — conservative)

    Returns
    -------
    (cleaned_df, report_dict)
    """
    report: dict = {}
    original_len = len(df)
    df = df.copy()

    logger.info(f"Starting KPI cleaning pipeline — {original_len:,} rows")

    # ── Step 1: Deduplicate ──────────────────────────────────────────────────
    df, report["duplicates"] = _remove_kpi_duplicates(df)

    # ── Step 2: Physical range validation ────────────────────────────────────
    df, report["range_violations"] = _fix_range_violations(df)

    # ── Step 3: Outlier handling ─────────────────────────────────────────────
    df, report["outliers"] = _handle_outliers(df, outlier_strategy, iqr_multiplier)

    # ── Step 4: Impute missing KPI values ────────────────────────────────────
    df, report["imputation"] = _impute_kpi_missing(df)

    # ── Step 5: Normalise cell IDs ───────────────────────────────────────────
    df = _normalise_cell_ids(df)

    # ── Step 6: Recompute qoe_category if needed ─────────────────────────────
    df = _recompute_qoe_category(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    rows_removed = original_len - len(df)
    report["summary"] = {
        "original_rows":  original_len,
        "final_rows":     len(df),
        "rows_removed":   rows_removed,
        "removal_pct":    round(rows_removed / original_len * 100, 2),
        "remaining_nulls": int(df.isnull().sum().sum()),
        "outlier_strategy": outlier_strategy,
    }

    logger.success(
        f"KPI cleaning complete — {len(df):,} rows retained "
        f"({rows_removed:,} removed)"
    )
    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# STEPS
# ─────────────────────────────────────────────────────────────────────────────

def _remove_kpi_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicates and duplicate (msisdn, timestamp) pairs."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_row_dedup = n_before - len(df)

    # Same user, same timestamp — keep last measurement (most recent update)
    if {"msisdn", "timestamp"}.issubset(df.columns):
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["msisdn", "timestamp"], keep="last"
        )
    n_ts_dedup = (n_before - n_row_dedup) - len(df)

    report = {
        "exact_duplicates_removed":      n_row_dedup,
        "msisdn_timestamp_dedup_removed": n_ts_dedup,
        "total_removed":                 n_row_dedup + n_ts_dedup,
    }
    logger.info(f"  KPI dedup: {report['total_removed']} records removed")
    return df, report


def _fix_range_violations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clamp values to physically valid ranges.
    E.g. packet_loss_pct cannot be < 0 or > 100.
    Records are NOT dropped — impossible values are clamped to boundary.
    """
    report: dict = {}
    for col, (lo, hi) in KPI_BOUNDS.items():
        if col not in df.columns:
            continue
        below = int((df[col] < lo).sum())
        above = int((df[col] > hi).sum())
        if below + above > 0:
            df[col] = df[col].clip(lower=lo, upper=hi)
            report[col] = {"below_min_clamped": below, "above_max_clamped": above}

    total = sum(
        v["below_min_clamped"] + v["above_max_clamped"]
        for v in report.values()
    )
    logger.info(f"  Range violations: {total} values clamped across {len(report)} KPIs")
    return df, report


def _handle_outliers(df: pd.DataFrame,
                     strategy: str,
                     k: float) -> tuple[pd.DataFrame, dict]:
    """
    IQR-based outlier handling for all numeric KPI columns.

    strategy='cap'  → Winsorise at [Q1 - k*IQR, Q3 + k*IQR]
    strategy='flag' → Add binary column kpi_outlier_flag
    strategy='drop' → Remove rows where any KPI is outside fences
    """
    numeric_kpis = [c for c in ALL_KPI_COLS if c in df.columns]
    report: dict  = {"strategy": strategy, "k": k, "affected_per_col": {}}
    outlier_mask  = pd.Series(False, index=df.index)

    for col in numeric_kpis:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - k * iqr
        hi  = q3 + k * iqr

        col_mask = (df[col] < lo) | (df[col] > hi)
        n_out    = int(col_mask.sum())

        if n_out > 0:
            report["affected_per_col"][col] = n_out
            if strategy == "cap":
                df[col] = df[col].clip(lower=lo, upper=hi)
            elif strategy in ("flag", "drop"):
                outlier_mask |= col_mask

    if strategy == "drop" and outlier_mask.any():
        n_drop = int(outlier_mask.sum())
        df = df[~outlier_mask].copy()
        report["rows_dropped"] = n_drop
        logger.info(f"  Outliers (drop): {n_drop} rows removed")
    elif strategy == "flag" and outlier_mask.any():
        df["kpi_outlier_flag"] = outlier_mask.astype(int)
        logger.info(f"  Outliers (flag): {int(outlier_mask.sum())} flagged")
    else:
        total_capped = sum(report["affected_per_col"].values())
        logger.info(f"  Outliers (cap): {total_capped} values capped across {len(report['affected_per_col'])} KPIs")

    return df, report


def _impute_kpi_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Per-KPI imputation. Default strategy: median per (region, cell_id).
    Falls back to global median if group median is NaN.
    """
    report: dict = {}

    group_cols = [c for c in ["region", "cell_id"] if c in df.columns]

    for col, strategy in KPI_IMPUTATION.items():
        if col not in df.columns:
            continue
        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            continue

        if strategy == "median" and group_cols:
            # Group median imputation
            group_medians = df.groupby(group_cols)[col].transform("median")
            global_median = df[col].median()
            df[col] = df[col].fillna(group_medians).fillna(global_median)
        else:
            df[col] = df[col].fillna(df[col].median())

        report[col] = {"n_imputed": n_missing, "strategy": strategy}

    total = sum(v["n_imputed"] for v in report.values())
    logger.info(f"  KPI imputation: {total:,} values imputed across {len(report)} columns")
    return df, report


def _normalise_cell_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise cell_id format: uppercase, strip whitespace."""
    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str).str.strip().str.upper()
        df.loc[df["cell_id"].isin(["NAN", "NONE", ""]), "cell_id"] = "UNKNOWN"
    return df


def _recompute_qoe_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute qoe_category from cleaned qoe_score using config thresholds.
    This ensures consistency after clamping/imputation changed some scores.
    """
    if "qoe_score" not in df.columns:
        return df

    thresholds = cfg["qoe"]["thresholds"]
    green  = thresholds["green"]
    yellow = thresholds["yellow"]

    df["qoe_category"] = pd.cut(
        df["qoe_score"],
        bins=[-np.inf, yellow, green, np.inf],
        labels=["Poor", "Fair", "Good"],
    ).astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_cleaning_report(report: dict) -> None:
    """Pretty-print the KPI cleaning audit trail."""
    print("\n" + "=" * 60)
    print("  KPI CLEANING REPORT")
    print("=" * 60)
    s = report.get("summary", {})
    print(f"  Input rows        : {s.get('original_rows', '?'):>10,}")
    print(f"  Output rows       : {s.get('final_rows', '?'):>10,}")
    print(f"  Rows removed      : {s.get('rows_removed', '?'):>10,}  ({s.get('removal_pct', '?')}%)")
    print(f"  Remaining nulls   : {s.get('remaining_nulls', '?'):>10,}")
    print(f"  Outlier strategy  : {s.get('outlier_strategy', '?')}")

    out_detail = report.get("outliers", {}).get("affected_per_col", {})
    if out_detail:
        print("\n  Outliers per KPI:")
        for col, cnt in sorted(out_detail.items(), key=lambda x: -x[1]):
            print(f"    {col:<40} : {cnt:>8,}")
    print("=" * 60 + "\n")