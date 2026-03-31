"""
Dashboard Data Loader
=====================
Centralised, cached data loading for the Streamlit NOC dashboard.
All heavy I/O happens once at startup; Streamlit caches results in memory.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
import yaml
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROCESSED   = Path(cfg["paths"]["processed_data"])
MODELS_ROOT = Path(cfg["paths"]["models"])


# ── Streamlit cache-compatible loaders ────────────────────────────────────────
# (import streamlit here so the module can also be used outside Streamlit)

def load_all():
    """Load and return all dashboard datasets as a dict."""
    data = {}

    # ── Processed datasets ────────────────────────────────────────────────────
    data["complaint_agg"]    = _load_parquet(PROCESSED / "complaint_daily_agg.parquet")
    data["kpi_agg"]          = _load_parquet(PROCESSED / "kpi_daily_agg.parquet")
    data["complaints_clean"] = _load_parquet(PROCESSED / "complaints_clean.parquet")
    data["feature_matrix"]   = _load_parquet(PROCESSED / "feature_matrix.parquet")

    # ── Model outputs ─────────────────────────────────────────────────────────
    data["anomaly_results"]  = _load_parquet(MODELS_ROOT / "anomaly" / "anomaly_results.parquet")
    data["forecasts"]        = _load_parquet(MODELS_ROOT / "prediction" / "forecasts.parquet")
    data["kmeans_users"]     = _load_parquet(MODELS_ROOT / "clustering" / "kmeans_users.parquet")

    # ── Prediction scores ─────────────────────────────────────────────────────
    try:
        scores_raw = joblib.load(MODELS_ROOT / "prediction" / "scores.pkl")
        # Flatten to DataFrame
        rows = []
        for region, models in scores_raw.items():
            for model_name, metrics in models.items():
                rows.append({"region": region, "model": model_name, **metrics})
        data["prediction_scores"] = pd.DataFrame(rows)
    except Exception:
        data["prediction_scores"] = pd.DataFrame()

    # ── Cluster profiles ──────────────────────────────────────────────────────
    try:
        data["cluster_profiles"] = _build_cluster_profiles(data["kmeans_users"])
    except Exception:
        data["cluster_profiles"] = pd.DataFrame()

    # ── Derive composite qoe_score_mean if missing ───────────────────────────
    ka = data["kpi_agg"]
    if "qoe_score_mean" not in ka.columns:
        if "data_qoe_score_mean" in ka.columns and "voice_qoe_score_mean" in ka.columns:
            ka["qoe_score_mean"] = (
                0.55 * ka["data_qoe_score_mean"] +
                0.45 * ka["voice_qoe_score_mean"]
            ).round(2)
        elif "data_qoe_score_mean" in ka.columns:
            ka["qoe_score_mean"] = ka["data_qoe_score_mean"]
        data["kpi_agg"] = ka

    # ── Date normalisation ────────────────────────────────────────────────────
    for key in ["complaint_agg", "kpi_agg", "anomaly_results", "forecasts", "feature_matrix"]:
        if key in data and "date" in data[key].columns:
            data[key]["date"] = pd.to_datetime(data[key]["date"])

    if "complaints_clean" in data and "timestamp" in data["complaints_clean"].columns:
        data["complaints_clean"]["timestamp"] = pd.to_datetime(data["complaints_clean"]["timestamp"])

    logger.info(f"Dashboard data loaded: {len(data)} datasets")
    return data


def _load_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    logger.warning(f"Missing file: {path}")
    return pd.DataFrame()


def _build_cluster_profiles(kmeans_users: pd.DataFrame) -> pd.DataFrame:
    """Build summary profile per cluster."""
    if kmeans_users.empty or "kmeans_cluster" not in kmeans_users.columns:
        return pd.DataFrame()
    kpi_mean_cols = [c for c in kmeans_users.columns if c.endswith("_mean")]
    agg = (
        kmeans_users.groupby("kmeans_cluster")
        .agg(n_users=("msisdn", "count"),
             **{c: (c, "mean") for c in kpi_mean_cols if c in kmeans_users.columns})
        .reset_index()
    )
    agg["pct"] = (agg["n_users"] / agg["n_users"].sum() * 100).round(1)
    return agg


# ── KPI display helpers ───────────────────────────────────────────────────────

KPI_META = {
    "qoe_score_mean":                  {"label": "QoE Score",            "unit": "",    "good": "high", "fmt": ".1f"},
    "dl_throughput_mbps_mean":         {"label": "DL Throughput",         "unit": "Mbps","good": "high", "fmt": ".1f"},
    "latency_ms_mean":                 {"label": "Latency",               "unit": "ms",  "good": "low",  "fmt": ".0f"},
    "packet_loss_pct_mean":            {"label": "Packet Loss",           "unit": "%",   "good": "low",  "fmt": ".2f"},
    "call_drop_rate_mean":             {"label": "Call Drop Rate",        "unit": "%",   "good": "low",  "fmt": ".2f"},
    "voice_quality_score_mos_mean":    {"label": "Voice MOS",             "unit": "",    "good": "high", "fmt": ".2f"},
    "data_session_success_rate_mean":  {"label": "Data Session SR",       "unit": "%",   "good": "high", "fmt": ".1f"},
    "call_setup_success_rate_mean":    {"label": "Call Setup SR",         "unit": "%",   "good": "high", "fmt": ".1f"},
}

REGIONS = cfg["data"]["regions"]
QOE_GREEN  = cfg["qoe"]["thresholds"]["green"]   # 80
QOE_YELLOW = cfg["qoe"]["thresholds"]["yellow"]  # 60


def qoe_color(score: float) -> str:
    if score >= QOE_GREEN:
        return "#2ecc71"
    elif score >= QOE_YELLOW:
        return "#f39c12"
    return "#e74c3c"


def delta_arrow(current: float, previous: float, good: str) -> str:
    if previous == 0:
        return ""
    pct = (current - previous) / abs(previous) * 100
    if good == "high":
        return f"{'▲' if pct >= 0 else '▼'} {abs(pct):.1f}%"
    else:
        return f"{'▼' if pct <= 0 else '▲'} {abs(pct):.1f}%"