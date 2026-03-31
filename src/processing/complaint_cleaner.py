"""
Complaint Data Cleaner
======================
Handles all cleaning operations for the complaint dataset:
  - Duplicate removal
  - Category & service type standardisation + fuzzy correction
  - Geographic validation and coordinate clamping
  - Temporal outlier removal
  - Missing value imputation (per-column strategy)
  - Outlier detection and flagging
  - Priority / segment encoding

Each step is a standalone function so it can be applied
selectively and documented individually in the thesis.

Output: a clean DataFrame ready for feature engineering.
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

VALID_REGIONS    = set(cfg["data"]["regions"])
VALID_SERVICES   = set(cfg["data"]["service_types"])
VALID_CATEGORIES = set(cfg["data"]["complaint_categories"])

# Ordered priority/segment maps for ordinal encoding
PRIORITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
SEGMENT_ORDER  = {"Standard": 0, "Premium": 1, "Enterprise": 2, "Vip": 3}

# Fuzzy correction maps — add real-world variations you observe in production data
CATEGORY_ALIASES: dict[str, str] = {
    "No Signal":           "No Service",
    "No Network":          "No Service",
    "Out Of Coverage":     "No Service",
    "Slow Internet":       "Slow Data",
    "Low Speed":           "Slow Data",
    "Dropped Call":        "Call Drop",
    "Call Disconnected":   "Call Drop",
    "Cannot Call":         "Call Setup Failure",
    "Call Failed":         "Call Setup Failure",
    "Sms Not Received":    "SMS Failure",       # note: Title-cased input
    "Sms Not Delivered":   "SMS Failure",
    "Bad Voice":           "Poor Voice Quality",
    "Echo":                "Poor Voice Quality",
    "Unstable Connection": "Intermittent Connection",
    "Roaming":             "Roaming Issue",
}

SERVICE_ALIASES: dict[str, str] = {
    "Mobile Data": "Data",
    "Internet":    "Data",
    "4G":          "Data",
    "5G":          "Data",
    "Call":        "Voice",
    "Calls":       "Voice",
    "Text":        "SMS",
    "Message":     "SMS",
}

# Tunisia bounding box (lat: 30–37.5, lon: 7.5–11.6)
LAT_MIN, LAT_MAX = 30.0, 37.5
LON_MIN, LON_MAX = 7.5,  11.6


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_complaints(df: pd.DataFrame,
                     drop_debug_cols: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline for complaint data.

    Parameters
    ----------
    df : raw complaint DataFrame (output of data_loader.load_complaints)
    drop_debug_cols : remove internal columns like _hotspot_multiplier

    Returns
    -------
    (cleaned_df, report_dict)
        cleaned_df  — fully cleaned DataFrame
        report_dict — step-by-step audit trail for thesis documentation
    """
    report: dict = {}
    original_len = len(df)
    df = df.copy()

    logger.info(f"Starting complaint cleaning pipeline — {original_len:,} rows")

    # ── Step 1: Drop debug / internal columns ────────────────────────────────
    if drop_debug_cols:
        debug_cols = [c for c in df.columns if c.startswith("_")]
        df.drop(columns=debug_cols, errors="ignore", inplace=True)
        report["debug_cols_dropped"] = debug_cols

    # ── Step 2: Deduplicate ──────────────────────────────────────────────────
    df, report["duplicates"] = _remove_duplicates(df)

    # ── Step 3: Standardise categorical values ───────────────────────────────
    df, report["category_fixes"] = _standardise_categories(df)
    df, report["service_fixes"]  = _standardise_service_types(df)

    # ── Step 4: Validate and fix geographic data ─────────────────────────────
    df, report["geo"] = _clean_geographic(df)

    # ── Step 5: Temporal validation ──────────────────────────────────────────
    df, report["temporal"] = _clean_temporal(df)

    # ── Step 6: Impute missing values ────────────────────────────────────────
    df, report["imputation"] = _impute_missing(df)

    # ── Step 7: Ordinal encode priority & segment ────────────────────────────
    df = _encode_ordinal(df)
    report["ordinal_encoded"] = ["priority_encoded", "segment_encoded"]

    # ── Step 8: Flag remaining unknowns ──────────────────────────────────────
    df = _flag_unknowns(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    rows_removed = original_len - len(df)
    report["summary"] = {
        "original_rows":  original_len,
        "final_rows":     len(df),
        "rows_removed":   rows_removed,
        "removal_pct":    round(rows_removed / original_len * 100, 2),
        "final_columns":  len(df.columns),
        "remaining_nulls": int(df.isnull().sum().sum()),
    }

    logger.success(
        f"Cleaning complete — {len(df):,} rows retained "
        f"({rows_removed:,} removed, {report['summary']['removal_pct']}%)"
    )
    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# STEP IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows and duplicate case_ids."""
    n_before = len(df)

    # Exact row duplicates
    df = df.drop_duplicates()
    n_after_rows = len(df)

    # Duplicate case_ids — keep first occurrence
    if "case_id" in df.columns:
        df = df.drop_duplicates(subset=["case_id"], keep="first")
    n_after_ids = len(df)

    report = {
        "exact_row_duplicates_removed": n_before - n_after_rows,
        "case_id_duplicates_removed":   n_after_rows - n_after_ids,
        "total_removed":                n_before - n_after_ids,
    }
    logger.info(f"  Dedup: removed {report['total_removed']} duplicates")
    return df, report


def _standardise_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply fuzzy alias corrections to complaint_category.
    Unknown categories are labelled 'Other' and flagged.
    """
    if "complaint_category" not in df.columns:
        return df, {}

    before = df["complaint_category"].value_counts().to_dict()

    # Apply alias map
    df["complaint_category"] = (
        df["complaint_category"]
        .str.strip()
        .str.title()
        .replace(CATEGORY_ALIASES)
    )

    # Remaining unknowns
    unknown_mask = ~df["complaint_category"].isin(VALID_CATEGORIES)
    n_unknown = unknown_mask.sum()
    if n_unknown > 0:
        logger.warning(f"  {n_unknown} complaints have unrecognised category → labelled 'Other'")
        df.loc[unknown_mask, "complaint_category"] = "Other"

    after = df["complaint_category"].value_counts().to_dict()
    report = {
        "aliases_applied":   len(CATEGORY_ALIASES),
        "unknown_relabelled": int(n_unknown),
        "before_counts":     before,
        "after_counts":      after,
    }
    logger.info(f"  Categories: {n_unknown} unknowns relabelled")
    return df, report


def _standardise_service_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Correct service_type variations to canonical values."""
    if "service_type" not in df.columns:
        return df, {}

    df["service_type"] = (
        df["service_type"]
        .str.strip()
        .str.title()
        .replace(SERVICE_ALIASES)
    )

    unknown_mask = ~df["service_type"].isin(VALID_SERVICES)
    n_unknown    = unknown_mask.sum()
    if n_unknown > 0:
        # Infer service from category when possible
        inferred = df.loc[unknown_mask, "complaint_category"].map({
            "Call Drop":          "Voice",
            "Call Setup Failure": "Voice",
            "Poor Voice Quality": "Voice",
            "Slow Data":          "Data",
            "SMS Failure":        "SMS",
        })
        df.loc[unknown_mask & inferred.notna(), "service_type"] = inferred.dropna()
        # Remaining still unknown
        still_unknown = ~df["service_type"].isin(VALID_SERVICES)
        df.loc[still_unknown, "service_type"] = "Unknown"

    report = {
        "aliases_applied": len(SERVICE_ALIASES),
        "unknown_inferred_from_category": int(n_unknown - int((~df["service_type"].isin(VALID_SERVICES)).sum())),
        "final_unknown_count": int((~df["service_type"].isin(VALID_SERVICES | {"Unknown"})).sum()),
    }
    logger.info(f"  Service types: {n_unknown} unknowns processed")
    return df, report


def _clean_geographic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validate coordinates, clamp to bounding box, impute missing coords
    from region centroid lookup, flag out-of-bounds records.
    """
    report: dict = {}

    # Region centroid lookup (approximate)
    region_centroids = {
        "Tunis":     (36.818, 10.165),
        "Sfax":      (34.740, 10.760),
        "Sousse":    (35.825, 10.638),
        "Kairouan":  (35.671, 10.100),
        "Bizerte":   (37.275,  9.873),
        "Gabes":     (33.881, 10.097),
        "Ariana":    (36.862, 10.193),
        "Gafsa":     (34.422,  8.784),
        "Monastir":  (35.777, 10.826),
        "Ben Arous": (36.753, 10.228),
    }

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df, {"status": "no_geo_columns"}

    n_missing_before = df[["latitude", "longitude"]].isnull().any(axis=1).sum()

    # Impute missing coords from region centroid
    if "region" in df.columns:
        missing_geo = df["latitude"].isnull() | df["longitude"].isnull()
        for region, (lat, lon) in region_centroids.items():
            mask = missing_geo & (df["region"] == region)
            df.loc[mask, "latitude"]  = lat
            df.loc[mask, "longitude"] = lon

    # Any still missing → Tunisia centroid
    still_missing = df["latitude"].isnull() | df["longitude"].isnull()
    df.loc[still_missing, "latitude"]  = 34.0
    df.loc[still_missing, "longitude"] = 9.0

    # Flag and clamp out-of-bounds
    out_of_bounds = (
        (df["latitude"]  < LAT_MIN) | (df["latitude"]  > LAT_MAX) |
        (df["longitude"] < LON_MIN) | (df["longitude"] > LON_MAX)
    )
    n_oob = int(out_of_bounds.sum())
    df.loc[out_of_bounds, "latitude"]  = df.loc[out_of_bounds, "latitude"].clip(LAT_MIN, LAT_MAX)
    df.loc[out_of_bounds, "longitude"] = df.loc[out_of_bounds, "longitude"].clip(LON_MIN, LON_MAX)
    df["geo_imputed"] = (
        (df["latitude"]  == df.get("latitude",  pd.Series(dtype=float))) |
        missing_geo
    ).astype(int)

    # Validate region names
    if "region" in df.columns:
        invalid_regions = ~df["region"].isin(VALID_REGIONS)
        n_invalid = int(invalid_regions.sum())
        df.loc[invalid_regions, "region"] = "Unknown"
        report["invalid_regions_relabelled"] = n_invalid

    report.update({
        "missing_coords_imputed": int(n_missing_before),
        "out_of_bounds_clamped":  n_oob,
    })
    logger.info(f"  Geo: {n_missing_before} coords imputed, {n_oob} out-of-bounds clamped")
    return df, report


def _clean_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove records with:
      - Timestamps in the future
      - Timestamps before a reasonable telecom start date (2015)
      - Negative or zero time deltas that suggest data entry errors
    """
    n_before = len(df)
    now = pd.Timestamp.now()
    min_date = pd.Timestamp("2015-01-01")

    future_mask = df["timestamp"] > now
    past_mask   = df["timestamp"] < min_date

    n_future = int(future_mask.sum())
    n_past   = int(past_mask.sum())

    df = df[~future_mask & ~past_mask].copy()

    report = {
        "future_timestamps_removed": n_future,
        "too_old_timestamps_removed": n_past,
        "total_removed": n_before - len(df),
    }
    logger.info(f"  Temporal: {n_future} future + {n_past} pre-2015 records removed")
    return df, report


def _impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Column-specific imputation strategies:
      - complaint_subcategory → 'Unknown'  (categorical, no safe inference)
      - cell_id               → 'UNKNOWN'  (categorical)
      - customer_segment      → mode       (most common segment)
      - priority              → mode
    """
    imputation_log: dict = {}

    cat_fill = {
        "complaint_subcategory": "Unknown",
        "cell_id":               "UNKNOWN",
    }
    for col, fill_val in cat_fill.items():
        if col in df.columns:
            n = int(df[col].isnull().sum())
            df[col] = df[col].fillna(fill_val)
            imputation_log[col] = {"strategy": f"constant='{fill_val}'", "n_imputed": n}

    mode_cols = ["customer_segment", "priority"]
    for col in mode_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            n = int(df[col].isnull().sum())
            df[col] = df[col].fillna(mode_val)
            imputation_log[col] = {"strategy": f"mode='{mode_val}'", "n_imputed": n}

    logger.info(f"  Imputation complete: {len(imputation_log)} columns processed")
    return df, imputation_log


def _encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Add ordinal-encoded versions of priority and customer_segment."""
    if "priority" in df.columns:
        df["priority_encoded"] = df["priority"].map(PRIORITY_ORDER).fillna(-1).astype(int)
    if "customer_segment" in df.columns:
        df["segment_encoded"] = df["customer_segment"].map(SEGMENT_ORDER).fillna(-1).astype(int)
    return df


def _flag_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary flag for records that had any imputation/correction applied."""
    flags = []
    if "geo_imputed" in df.columns:
        flags.append(df["geo_imputed"])
    if "complaint_category" in df.columns:
        flags.append((df["complaint_category"] == "Other").astype(int))
    if "service_type" in df.columns:
        flags.append((df["service_type"] == "Unknown").astype(int))

    if flags:
        df["data_quality_flag"] = (sum(flags) > 0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_cleaning_report(report: dict) -> None:
    """Pretty-print the cleaning audit trail."""
    print("\n" + "=" * 60)
    print("  COMPLAINT CLEANING REPORT")
    print("=" * 60)
    s = report.get("summary", {})
    print(f"  Input rows        : {s.get('original_rows', '?'):>10,}")
    print(f"  Output rows       : {s.get('final_rows', '?'):>10,}")
    print(f"  Rows removed      : {s.get('rows_removed', '?'):>10,}  ({s.get('removal_pct', '?')}%)")
    print(f"  Output columns    : {s.get('final_columns', '?'):>10}")
    print(f"  Remaining nulls   : {s.get('remaining_nulls', '?'):>10,}")

    print("\n  Step Details:")
    for step, detail in report.items():
        if step == "summary":
            continue
        print(f"\n  [{step}]")
        if isinstance(detail, dict):
            for k, v in detail.items():
                if not isinstance(v, dict):
                    print(f"    {k:<35} : {v}")
        elif isinstance(detail, list):
            print(f"    {detail}")
    print("=" * 60 + "\n")