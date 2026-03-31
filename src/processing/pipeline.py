"""
Processing Pipeline Orchestrator
==================================
Runs the full Phase 2 pipeline in sequence:

  1. Load raw data
  2. Clean complaints  → cleaned_complaints
  3. Clean KPI data    → cleaned_kpi
  4. Feature engineer  → complaint_daily_agg
                       → kpi_daily_agg
                       → feature_matrix (joined)
  5. Time-series split → X_train, X_test, y_train, y_test
  6. Save all outputs  → data/processed/

Run directly:
    python -m src.processing.pipeline

Or import and call run_pipeline() from a notebook.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from loguru import logger

from src.ingestion.data_loader     import load_complaints, load_kpi_data
from src.processing.complaint_cleaner import (
    clean_complaints, print_cleaning_report as print_complaint_report
)
from src.processing.Kpi_cleaner import (
    clean_kpi_data, print_cleaning_report as print_kpi_report
)
from src.processing.feature_engineering import (
    build_complaint_daily_agg,
    build_kpi_daily_agg,
    add_kpi_degradation_flags,
    build_feature_matrix,
    time_series_split,
    save_processed,
)


def run_pipeline(outlier_strategy: str = "cap",
                 verbose: bool = True) -> dict:
    """
    Execute the full Phase 2 pipeline.

    Parameters
    ----------
    outlier_strategy : 'cap' | 'flag' | 'drop'
    verbose          : print cleaning reports

    Returns
    -------
    dict with keys:
        cleaned_complaints, cleaned_kpi,
        complaint_agg, kpi_agg, feature_matrix,
        X_train, X_test, y_train, y_test
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 — Data Cleaning & Feature Engineering Pipeline")
    logger.info("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading raw data ...")
    complaints_raw = load_complaints()
    kpi_raw        = load_kpi_data()

    # ── 2. Clean complaints ───────────────────────────────────────────────────
    logger.info("\n[2/5] Cleaning complaint data ...")
    complaints_clean, complaint_report = clean_complaints(complaints_raw)
    if verbose:
        print_complaint_report(complaint_report)
    save_processed(complaints_clean, "complaints_clean")

    # ── 3. Clean KPI data ─────────────────────────────────────────────────────
    logger.info("\n[3/5] Cleaning KPI data ...")
    kpi_clean, kpi_report = clean_kpi_data(kpi_raw, outlier_strategy=outlier_strategy)
    if verbose:
        print_kpi_report(kpi_report)
    save_processed(kpi_clean, "kpi_clean")

    # ── 4. Feature engineering ────────────────────────────────────────────────
    logger.info("\n[4/5] Engineering features ...")

    complaint_agg = build_complaint_daily_agg(complaints_clean)
    save_processed(complaint_agg, "complaint_daily_agg")

    kpi_agg = build_kpi_daily_agg(kpi_clean)
    kpi_agg = add_kpi_degradation_flags(kpi_agg)
    save_processed(kpi_agg, "kpi_daily_agg")

    feature_matrix = build_feature_matrix(complaint_agg, kpi_agg)
    save_processed(feature_matrix, "feature_matrix")

    # ── 5. Train/test split ───────────────────────────────────────────────────
    logger.info("\n[5/5] Creating time-series train/test split ...")
    X_train, X_test, y_train, y_test = time_series_split(feature_matrix)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE — Summary")
    logger.info("=" * 60)
    logger.info(f"  Cleaned complaints  : {len(complaints_clean):>8,} rows × {complaints_clean.shape[1]} cols")
    logger.info(f"  Cleaned KPI data    : {len(kpi_clean):>8,} rows × {kpi_clean.shape[1]} cols")
    logger.info(f"  Complaint daily agg : {len(complaint_agg):>8,} rows × {complaint_agg.shape[1]} cols")
    logger.info(f"  KPI daily agg       : {len(kpi_agg):>8,} rows × {kpi_agg.shape[1]} cols")
    logger.info(f"  Feature matrix      : {len(feature_matrix):>8,} rows × {feature_matrix.shape[1]} cols")
    logger.info(f"  X_train             : {len(X_train):>8,} rows")
    logger.info(f"  X_test              : {len(X_test):>8,} rows")
    logger.info("=" * 60)

    return {
        "cleaned_complaints": complaints_clean,
        "cleaned_kpi":        kpi_clean,
        "complaint_agg":      complaint_agg,
        "kpi_agg":            kpi_agg,
        "feature_matrix":     feature_matrix,
        "X_train":            X_train,
        "X_test":             X_test,
        "y_train":            y_train,
        "y_test":             y_test,
        "reports": {
            "complaint_cleaning": complaint_report,
            "kpi_cleaning":       kpi_report,
        },
    }


if __name__ == "__main__":
    results = run_pipeline(verbose=True)