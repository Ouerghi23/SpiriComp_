"""
Feature Engineering Module
===========================
Transforms clean complaint and KPI DataFrames into ML-ready feature matrices.

Organised into four groups:

  A. Temporal features
     — Hour, day-of-week, week, month, is_weekend, is_peak_hour
     — Cyclical encoding (sin/cos) for periodic features

  B. Complaint aggregation features
     — Daily/hourly complaint counts per (region, service_type, category)
     — Lag features: complaint counts at t-1, t-3, t-7, t-14 days
     — Rolling averages: 3d, 7d, 14d, 30d windows

  C. KPI aggregation features
     — Cell/region-level daily KPI aggregates (mean, min, p10, std)
     — Rolling KPI trends (7-day average)
     — KPI degradation flags (below threshold)

  D. Join & merge
     — Joins complaint aggregates with KPI aggregates on (region, date)
     — Produces a single analysis-ready DataFrame for ML

All functions accept and return DataFrames with an audit trail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# ── Config 
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

LAG_WINDOWS     = cfg["features"]["lag_windows"]       # [1, 3, 7, 14]
ROLLING_WINDOWS = cfg["features"]["rolling_windows"]   # [3, 7, 14, 30]

DATA_KPI_COLS  = cfg["features"]["kpi_columns"]["data"]
VOICE_KPI_COLS = cfg["features"]["kpi_columns"]["voice"]
ALL_KPI_COLS   = DATA_KPI_COLS + VOICE_KPI_COLS

# KPI degradation thresholds (below/above = degraded)
KPI_DEGRADATION_THRESHOLDS: dict[str, tuple[str, float]] = {
    "dl_throughput_mbps":        ("below", 1.0),    # < 1 Mbps = poor
    "latency_ms":                ("above", 300.0),   # > 300 ms = high latency
    "packet_loss_pct":           ("above", 5.0),     # > 5% = significant loss
    "data_session_success_rate": ("below", 90.0),    # < 90% = degraded
    "call_setup_success_rate":   ("below", 92.0),
    "call_drop_rate":            ("above", 3.0),     # > 3% drop rate
    "voice_quality_score_mos":   ("below", 3.0),     # MOS < 3 = poor
    "qoe_score":                 ("below", 60.0),    # composite QoE < 60
}

PEAK_HOURS = {8, 9, 12, 13, 17, 18, 19, 20}


# ─────────────────────────────────────────────────────────────────────────────
# A. TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame,
                          ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Add a comprehensive set of temporal features to any DataFrame
    that has a datetime column.

    Features added
    --------------
    hour, day_of_week_num, week, month, year,
    is_weekend, is_peak_hour, is_business_hour,
    hour_sin, hour_cos,          ← cyclical
    dow_sin,  dow_cos,           ← cyclical
    month_sin, month_cos         ← cyclical
    """
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])

    df["hour"]           = ts.dt.hour
    df["day_of_week_num"]= ts.dt.dayofweek          # 0=Mon … 6=Sun
    df["week"]           = ts.dt.isocalendar().week.astype(int)
    df["month"]          = ts.dt.month
    df["year"]           = ts.dt.year
    df["day_of_year"]    = ts.dt.dayofyear
    df["quarter"]        = ts.dt.quarter

    df["is_weekend"]      = (df["day_of_week_num"] >= 5).astype(int)
    df["is_peak_hour"]    = df["hour"].isin(PEAK_HOURS).astype(int)
    df["is_business_hour"]= df["hour"].between(8, 18).astype(int)
    df["is_night"]        = df["hour"].between(0, 6).astype(int)

    # Cyclical encoding (preserves periodicity for ML models)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]            / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]            / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week_num"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]           / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]           / 12)

    logger.info("  Temporal features added (16 new columns)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B. COMPLAINT AGGREGATION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_complaint_daily_agg(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate complaints to (region, date) granularity.

    Produces:
      - total_complaints
      - complaints_data / _voice / _sms
      - complaints per category (pivoted)
      - complaints by priority (high + critical count)
      - Lag features: t-1, t-3, t-7, t-14
      - Rolling averages: 3d, 7d, 14d, 30d
    """
    df = complaints.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    logger.info("  Building daily complaint aggregates ...")

    # ── Base count ────────────────────────────────────────────────────────────
    base = (
        df.groupby(["region", "date"])
          .size()
          .reset_index(name="total_complaints")
    )

    # ── By service type ───────────────────────────────────────────────────────
    svc = (
        df.groupby(["region", "date", "service_type"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    svc.columns = (
        ["region", "date"] +
        [f"complaints_{c.lower()}" for c in svc.columns[2:]]
    )

    # ── By category (top 8) ───────────────────────────────────────────────────
    cat = (
        df.groupby(["region", "date", "complaint_category"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    cat.columns = (
        ["region", "date"] +
        [f"cat_{c.lower().replace(' ', '_')}" for c in cat.columns[2:]]
    )

    # ── High priority complaints ──────────────────────────────────────────────
    high_prio = (
        df[df["priority"].isin(["High", "Critical"])]
          .groupby(["region", "date"])
          .size()
          .reset_index(name="high_priority_complaints")
    )

    # ── VIP / Enterprise customers ────────────────────────────────────────────
    vip = (
        df[df["customer_segment"].isin(["Vip", "Enterprise"])]
          .groupby(["region", "date"])
          .size()
          .reset_index(name="vip_complaints")
    )

    # ── Merge all ─────────────────────────────────────────────────────────────
    agg = base.copy()
    for part in [svc, cat, high_prio, vip]:
        agg = agg.merge(part, on=["region", "date"], how="left")
    agg = agg.fillna(0)

    # ── Fill complete date × region grid (no gaps for time series) ────────────
    agg = _fill_date_region_grid(agg)

    # ── Lag features ──────────────────────────────────────────────────────────
    agg = _add_lag_features(agg, "total_complaints", LAG_WINDOWS)

    # ── Rolling features ──────────────────────────────────────────────────────
    agg = _add_rolling_features(agg, "total_complaints", ROLLING_WINDOWS)

    # ── Complaint spike flag (> mean + 2*std per region) ─────────────────────
    agg = _add_spike_flag(agg)

    logger.info(f"  Complaint daily agg: {agg.shape[0]:,} rows × {agg.shape[1]} columns")
    return agg


def _fill_date_region_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every (region, date) combination exists.
    Missing days = 0 complaints (not missing data, just truly zero).
    """
    df["date"] = pd.to_datetime(df["date"])
    all_dates  = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    all_regions = df["region"].unique()
    full_grid = pd.MultiIndex.from_product(
        [all_regions, all_dates], names=["region", "date"]
    ).to_frame(index=False)

    df = full_grid.merge(df, on=["region", "date"], how="left").fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["region", "date"]).reset_index(drop=True)


def _add_lag_features(df: pd.DataFrame,
                      col: str,
                      windows: list[int]) -> pd.DataFrame:
    """Add lag-N columns per region for the given column."""
    df = df.sort_values(["region", "date"])
    for lag in windows:
        df[f"{col}_lag_{lag}d"] = (
            df.groupby("region")[col]
              .shift(lag)
              .fillna(0)
        )
    return df


def _add_rolling_features(df: pd.DataFrame,
                          col: str,
                          windows: list[int]) -> pd.DataFrame:
    """Add rolling mean and std columns per region."""
    df = df.sort_values(["region", "date"])
    for w in windows:
        df[f"{col}_roll_mean_{w}d"] = (
            df.groupby("region")[col]
              .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
              .fillna(0)
        )
        df[f"{col}_roll_std_{w}d"] = (
            df.groupby("region")[col]
              .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
              .fillna(0)
        )
    return df


def _add_spike_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag days where complaints exceed mean + 2*std for that region."""
    stats = (
        df.groupby("region")["total_complaints"]
          .agg(["mean", "std"])
          .reset_index()
    )
    df = df.merge(stats, on="region", how="left")
    df["complaint_spike_flag"] = (
        df["total_complaints"] > (df["mean"] + 2 * df["std"].fillna(0))
    ).astype(int)
    df = df.drop(columns=["mean", "std"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# C. KPI AGGREGATION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_kpi_daily_agg(kpi_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-session KPI data to (region, date) granularity.

    Produces for each KPI:
      - mean, median, min, max, std
      - 10th percentile (captures worst 10% of users)
      - degradation rate (% of sessions below/above threshold)

    Also produces:
      - session_count per region-day
      - degraded_session_rate
      - Rolling 7-day KPI trend
    """
    df = kpi_data.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    logger.info("  Building daily KPI aggregates ...")

    present_kpis = [c for c in ALL_KPI_COLS if c in df.columns]

    # ── Percentile aggregations ───────────────────────────────────────────────
    agg_dict: dict = {"timestamp": "count"}
    for col in present_kpis:
        agg_dict[col] = ["mean", "median", "min", "max", "std",
                         lambda x: x.quantile(0.10)]

    agg = df.groupby(["region", "date"]).agg(agg_dict).reset_index()

    # Flatten multi-level columns
    new_cols = ["region", "date", "session_count"]
    for col in present_kpis:
        for stat in ["mean", "median", "min", "max", "std", "p10"]:
            new_cols.append(f"{col}_{stat}")
    agg.columns = new_cols

    # ── Degradation rate per KPI ──────────────────────────────────────────────
    for kpi, (direction, threshold) in KPI_DEGRADATION_THRESHOLDS.items():
        if kpi not in df.columns:
            continue
        if direction == "below":
            deg = df.groupby(["region", "date"])[kpi].apply(
                lambda x: (x < threshold).mean() * 100
            ).reset_index(name=f"{kpi}_degradation_rate")
        else:
            deg = df.groupby(["region", "date"])[kpi].apply(
                lambda x: (x > threshold).mean() * 100
            ).reset_index(name=f"{kpi}_degradation_rate")
        agg = agg.merge(deg, on=["region", "date"], how="left")

    # ── Degraded session rate ─────────────────────────────────────────────────
    if "is_degraded_session" in df.columns:
        deg_rate = (
            df.groupby(["region", "date"])["is_degraded_session"]
              .mean()
              .mul(100)
              .reset_index(name="degraded_session_rate_pct")
        )
        agg = agg.merge(deg_rate, on=["region", "date"], how="left")

    # ── Fill date grid ────────────────────────────────────────────────────────
    agg = _fill_date_region_grid(agg)

    # ── 7-day rolling mean for key KPIs (trend signal) ────────────────────────
    for col in [f"{k}_mean" for k in present_kpis]:
        if col in agg.columns:
            agg = _add_rolling_features(agg, col, [7])

    agg["date"] = pd.to_datetime(agg["date"])
    logger.info(f"  KPI daily agg: {agg.shape[0]:,} rows × {agg.shape[1]} columns")
    return agg


def add_kpi_degradation_flags(kpi_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flag columns indicating whether a region-day
    crossed degradation thresholds for key KPIs.
    """
    for kpi, (direction, threshold) in KPI_DEGRADATION_THRESHOLDS.items():
        rate_col = f"{kpi}_degradation_rate"
        if rate_col in kpi_agg.columns:
            kpi_agg[f"{kpi}_degraded_flag"] = (
                kpi_agg[rate_col] > 20  # >20% of sessions degraded
            ).astype(int)
    return kpi_agg


# ─────────────────────────────────────────────────────────────────────────────
# D. JOIN: Complaints + KPI → Unified Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(complaint_agg: pd.DataFrame,
                         kpi_agg:       pd.DataFrame,
                         join_strategy: str = "left") -> pd.DataFrame:
    """
    Join complaint and KPI aggregates on (region, date).

    Parameters
    ----------
    complaint_agg  : output of build_complaint_daily_agg
    kpi_agg        : output of build_kpi_daily_agg
    join_strategy  : 'left' keeps all complaint days (recommended)
                     'inner' keeps only days with both data sources

    Returns
    -------
    Unified feature matrix ready for ML model ingestion.
    """
    logger.info(f"  Joining complaint + KPI aggregates (strategy='{join_strategy}') ...")

    # Normalise date types
    complaint_agg["date"] = pd.to_datetime(complaint_agg["date"])
    kpi_agg["date"]       = pd.to_datetime(kpi_agg["date"])

    merged = complaint_agg.merge(
        kpi_agg,
        on=["region", "date"],
        how=join_strategy,
        suffixes=("_complaint", "_kpi"),
    )

    # Add temporal features to the joined matrix
    merged["timestamp_proxy"] = pd.to_datetime(merged["date"])
    merged = add_temporal_features(merged, ts_col="timestamp_proxy")
    merged = merged.drop(columns=["timestamp_proxy"], errors="ignore")

    # Fill any KPI nulls that appeared from the join
    kpi_cols_in_merged = [c for c in merged.columns
                          if any(k in c for k in ALL_KPI_COLS)]
    merged[kpi_cols_in_merged] = merged[kpi_cols_in_merged].fillna(
        merged[kpi_cols_in_merged].median()
    )

    # Region one-hot encoding (for models that need it)
    region_dummies = pd.get_dummies(merged["region"], prefix="region", drop_first=False)
    merged = pd.concat([merged, region_dummies], axis=1)

    logger.info(
        f"  Feature matrix: {merged.shape[0]:,} rows × {merged.shape[1]} features"
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# E. TRAIN / TEST SPLIT (time-aware)
# ─────────────────────────────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame,
                      date_col:  str   = "date",
                      test_size: float = 0.20,
                      target:    str   = "total_complaints"
                      ) -> tuple[pd.DataFrame, pd.DataFrame,
                                 pd.Series,    pd.Series]:
    """
    Chronological train/test split.
    NEVER shuffles — preserves temporal ordering to avoid data leakage.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    drop_cols = [date_col, target, "region",
                 "day_of_week",     # string version — use numeric
                 "date",
                 "complaint_spike_flag",   # leakage risk if used as feature
                 ]
    feature_cols = [c for c in df.columns
                    if c not in drop_cols
                    and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, bool]]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        f"  Train/test split: {len(X_train):,} train | {len(X_test):,} test "
        f"| {len(feature_cols)} features"
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# F. SAVE PROCESSED DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame,
                   name: str,
                   fmt:  str = "parquet") -> Path:
    """
    Save a processed DataFrame to data/processed/.
    Parquet preferred (columnar, fast, preserves dtypes).
    """
    out_dir = Path(cfg["paths"]["processed_data"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)

    size_mb = path.stat().st_size / 1_048_576
    logger.success(f"  Saved {name} → {path}  ({size_mb:.1f} MB, {len(df):,} rows)")
    return path