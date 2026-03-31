"""
Anomaly Detection Module
========================
Detects abnormal KPI drops and complaint surge events using:

  Model 1 — Isolation Forest (multivariate, unsupervised)
            Best for detecting unusual combinations of KPI values
            that individually look normal but together signal a problem.

  Model 2 — Statistical Control Charts (Z-score + CUSUM)
            Best for univariate time-series monitoring of a single
            KPI or complaint count over time. Simple and explainable
            — preferred by NOC operators.

Both models produce:
  - anomaly_flag   (0 = normal, 1 = anomaly)
  - anomaly_score  (continuous severity score)
  - A trained artifact saved to models/anomaly/

Usage (from notebook):
    from src.models.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    results  = detector.run(feature_matrix, kpi_agg)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MODELS_DIR = Path(cfg["paths"]["models"]) / "anomaly"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CONTAMINATION   = cfg["models"]["anomaly_contamination"]   # 0.05
RANDOM_STATE    = cfg["models"]["random_state"]

# KPI features to use for Isolation Forest
ANOMALY_FEATURES = [
    "qoe_score_mean", "qoe_score_p10",
    "dl_throughput_mbps_mean", "dl_throughput_mbps_p10",
    "latency_ms_mean", "latency_ms_max",
    "packet_loss_pct_mean",
    "call_drop_rate_mean",
    "voice_quality_score_mos_mean",
    "data_session_success_rate_mean",
    "call_setup_success_rate_mean",
    "degraded_session_rate_pct",
]


class AnomalyDetector:
    """
    Unified anomaly detection wrapper.
    Trains Isolation Forest and Z-score/CUSUM detectors,
    evaluates them, and returns a combined anomaly report.
    """

    def __init__(self):
        self.iso_forest  : IsolationForest | None = None
        self.scaler      : StandardScaler  | None = None
        self.results_if  : pd.DataFrame   | None = None
        self.results_stat: pd.DataFrame   | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, kpi_agg: pd.DataFrame) -> dict:
        """
        Full anomaly detection pipeline.

        Parameters
        ----------
        kpi_agg : daily KPI aggregates (output of build_kpi_daily_agg)

        Returns
        -------
        dict with keys: isolation_forest, statistical, combined, metrics
        """
        logger.info("=" * 55)
        logger.info("ANOMALY DETECTION")
        logger.info("=" * 55)

        # ── 1. Isolation Forest ───────────────────────────────────────────────
        logger.info("\n[1/3] Training Isolation Forest ...")
        self.results_if = self._run_isolation_forest(kpi_agg)

        # ── 2. Statistical (Z-score + CUSUM) ─────────────────────────────────
        logger.info("\n[2/3] Running statistical control charts ...")
        self.results_stat = self._run_statistical(kpi_agg)

        # ── 3. Combine & evaluate ─────────────────────────────────────────────
        logger.info("\n[3/3] Combining results ...")
        combined, metrics = self._combine_results(self.results_if, self.results_stat)

        # ── Save ──────────────────────────────────────────────────────────────
        self._save()
        combined.to_parquet(MODELS_DIR / "anomaly_results.parquet", index=False)

        logger.success(
            f"\nAnomaly detection complete:\n"
            f"  Isolation Forest anomalies : {self.results_if['if_anomaly'].sum():>5}\n"
            f"  Statistical anomalies      : {self.results_stat['stat_anomaly'].sum():>5}\n"
            f"  Combined (either)          : {combined['anomaly_flag'].sum():>5}\n"
            f"  Combined (both agree)      : {combined['anomaly_consensus'].sum():>5}"
        )

        return {
            "isolation_forest": self.results_if,
            "statistical":      self.results_stat,
            "combined":         combined,
            "metrics":          metrics,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1 — ISOLATION FOREST
    # ─────────────────────────────────────────────────────────────────────────

    def _run_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        # Select available features
        feat_cols = [c for c in ANOMALY_FEATURES if c in df.columns]
        if not feat_cols:
            raise ValueError("No anomaly feature columns found in kpi_agg.")

        X = df[feat_cols].fillna(df[feat_cols].median())

        # Scale
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        # Train
        self.iso_forest = IsolationForest(
            n_estimators  = 200,
            contamination = CONTAMINATION,
            random_state  = RANDOM_STATE,
            n_jobs        = -1,
        )
        preds  = self.iso_forest.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
        scores = self.iso_forest.decision_function(X_scaled)  # lower = more anomalous

        result = df[["region", "date"]].copy()
        result["if_anomaly"]      = (preds == -1).astype(int)
        result["if_score"]        = -scores          # invert: higher = more anomalous
        result["if_score_norm"]   = _minmax(result["if_score"])
        result["if_severity"]     = pd.cut(
            result["if_score_norm"],
            bins=[-np.inf, 0.33, 0.66, np.inf],
            labels=["Low", "Medium", "High"]
        )

        # Which features drove the anomaly most (feature contribution proxy)
        result["top_anomaly_driver"] = _top_driver(X, feat_cols, self.iso_forest)

        logger.info(
            f"  IF: {result['if_anomaly'].sum()} anomalies detected "
            f"({result['if_anomaly'].mean()*100:.1f}% of records)"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2 — STATISTICAL CONTROL CHARTS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_statistical(self, df: pd.DataFrame,
                         zscore_threshold: float = 3.0,
                         cusum_threshold:  float = 5.0) -> pd.DataFrame:
        """
        For each region independently:
          - Z-score: flag days where qoe_score_mean deviates > 3σ from rolling mean
          - CUSUM:   detect sustained cumulative drifts in complaint-relevant KPIs
        """
        result_rows = []

        monitoring_col = "qoe_score_mean" if "qoe_score_mean" in df.columns else \
                         next((c for c in df.columns if "qoe" in c), None)
        if monitoring_col is None:
            logger.warning("  No QoE column found for statistical detection.")
            return df[["region", "date"]].assign(stat_anomaly=0, zscore=0.0, cusum=0.0)

        for region, grp in df.groupby("region"):
            grp = grp.sort_values("date").copy()
            series = grp[monitoring_col].fillna(grp[monitoring_col].median())

            # ── Z-score ───────────────────────────────────────────────────────
            roll_mean = series.rolling(14, min_periods=3).mean()
            roll_std  = series.rolling(14, min_periods=3).std().replace(0, 1e-6)
            zscore    = ((series - roll_mean) / roll_std).abs()

            # ── CUSUM ─────────────────────────────────────────────────────────
            mean_global = series.mean()
            std_global  = series.std() if series.std() > 0 else 1e-6
            cusum_pos   = _cusum(series, mean_global, std_global)
            cusum_neg   = _cusum(-series, -mean_global, std_global)
            cusum_max   = np.maximum(cusum_pos, cusum_neg)

            # ── Combine ───────────────────────────────────────────────────────
            stat_anomaly = (
                (zscore > zscore_threshold) |
                (cusum_max > cusum_threshold)
            ).astype(int)

            tmp = grp[["region", "date"]].copy()
            tmp["zscore"]       = zscore.values
            tmp["cusum"]        = cusum_max
            tmp["stat_anomaly"] = stat_anomaly.values
            tmp["stat_score"]   = (zscore / zscore_threshold).clip(0, 3).values
            result_rows.append(tmp)

        result = pd.concat(result_rows, ignore_index=True)
        logger.info(
            f"  Statistical: {result['stat_anomaly'].sum()} anomalies detected "
            f"({result['stat_anomaly'].mean()*100:.1f}% of records)"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINE
    # ─────────────────────────────────────────────────────────────────────────

    def _combine_results(self,
                         if_res:   pd.DataFrame,
                         stat_res: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        combined = if_res.merge(
            stat_res[["region", "date", "stat_anomaly", "zscore", "cusum", "stat_score"]],
            on=["region", "date"], how="left"
        )
        combined["anomaly_flag"]      = (
            (combined["if_anomaly"] == 1) | (combined["stat_anomaly"] == 1)
        ).astype(int)
        combined["anomaly_consensus"] = (
            (combined["if_anomaly"] == 1) & (combined["stat_anomaly"] == 1)
        ).astype(int)
        combined["combined_score"]    = (
            0.6 * combined["if_score_norm"] +
            0.4 * combined["stat_score"].fillna(0).clip(0, 1)
        )

        metrics = {
            "total_records":            len(combined),
            "if_anomalies":             int(combined["if_anomaly"].sum()),
            "stat_anomalies":           int(combined["stat_anomaly"].sum()),
            "union_anomalies":          int(combined["anomaly_flag"].sum()),
            "consensus_anomalies":      int(combined["anomaly_consensus"].sum()),
            "anomaly_rate_pct":         round(combined["anomaly_flag"].mean() * 100, 2),
            "top_anomaly_regions":      (
                combined[combined["anomaly_flag"] == 1]["region"]
                .value_counts().head(5).to_dict()
            ),
        }
        return combined, metrics

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        joblib.dump(self.iso_forest, MODELS_DIR / "isolation_forest.pkl")
        joblib.dump(self.scaler,     MODELS_DIR / "if_scaler.pkl")
        logger.info(f"  Models saved → {MODELS_DIR}")

    @classmethod
    def load(cls) -> "AnomalyDetector":
        obj = cls()
        obj.iso_forest = joblib.load(MODELS_DIR / "isolation_forest.pkl")
        obj.scaler     = joblib.load(MODELS_DIR / "if_scaler.pkl")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


def _cusum(series: pd.Series, mu: float, sigma: float,
           k: float = 0.5) -> np.ndarray:
    """One-sided CUSUM statistic."""
    s = np.zeros(len(series))
    for i in range(1, len(series)):
        s[i] = max(0, s[i-1] + (series.iloc[i] - mu) / sigma - k)
    return s


def _top_driver(X: pd.DataFrame, cols: list[str],
                model: IsolationForest) -> pd.Series:
    """
    Approximate feature importance per sample:
    find the column with the highest absolute deviation from median.
    (Exact SHAP for Isolation Forest requires shap TreeExplainer — 
     done separately in the notebook for performance.)
    """
    medians  = X.median()
    stds     = X.std().replace(0, 1e-9)
    deviations = ((X - medians) / stds).abs()
    return deviations.idxmax(axis=1)