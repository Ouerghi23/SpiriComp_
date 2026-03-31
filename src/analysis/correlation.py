"""
Correlation & Root Cause Analysis Module
==========================================
Deliverable D3 — Statistical linkage between network KPIs
and customer complaint patterns.

Five analysis sections:

  1. Pearson & Spearman correlation matrices
     — KPI means vs complaint counts at (region, date) level
     — Identifies which KPIs are most strongly associated with complaints

  2. KPI threshold detection
     — For each KPI, finds the breakpoint value beyond which
       complaint spikes become significantly more likely
     — Uses decision tree stump (depth=1) for interpretable thresholds

  3. Granger causality testing
     — Does KPI degradation *precede* complaint spikes?
     — Tests lags 1–7 days per region to find lead time

  4. QoE degradation event analysis
     — Links sessions where QoE < threshold to complaint surges
       on the same day / region
     — Quantifies: when QoE drops, how much do complaints increase?

  5. KPI–complaint cross-correlation (CCF)
     — Time-lagged correlation between each KPI and complaint volume
     — Shows peak lag (how many days after KPI drops do complaints peak)

Usage (from notebook):
    from src.analysis.correlation import CorrelationAnalyser
    analyser = CorrelationAnalyser()
    results  = analyser.run(complaint_agg, kpi_agg, feature_matrix)
"""

from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.tree import DecisionTreeClassifier

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels not available — Granger causality skipped")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

REPORTS_DIR = Path(cfg["paths"]["reports"]) / "exports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

QOE_THRESHOLD = cfg["qoe"]["thresholds"]["yellow"]   # 60

KPI_COLS = (
    cfg["features"]["kpi_columns"]["data"] +
    cfg["features"]["kpi_columns"]["voice"]
)

# KPI display names for readable output
KPI_LABELS = {
    "dl_throughput_mbps":        "DL Throughput (Mbps)",
    "ul_throughput_mbps":        "UL Throughput (Mbps)",
    "latency_ms":                "Latency (ms)",
    "packet_loss_pct":           "Packet Loss (%)",
    "data_session_success_rate": "Data Session SR (%)",
    "data_qoe_score":            "Data QoE Score",
    "call_setup_success_rate":   "Call Setup SR (%)",
    "call_drop_rate":            "Call Drop Rate (%)",
    "voice_quality_score_mos":   "Voice MOS",
    "handover_success_rate":     "Handover SR (%)",
    "voice_qoe_score":           "Voice QoE Score",
}


class CorrelationAnalyser:
    """
    Runs the full correlation & root cause analysis pipeline.
    All results are returned as DataFrames suitable for
    plotting in the notebook and documenting in the thesis.
    """

    def run(self,
            complaint_agg:  pd.DataFrame,
            kpi_agg:        pd.DataFrame,
            feature_matrix: pd.DataFrame) -> dict:
        """
        Run all five analysis sections.

        Returns
        -------
        dict with keys:
            pearson_matrix, spearman_matrix,
            top_correlations,
            thresholds,
            granger_results,
            qoe_event_analysis,
            ccf_results,
            summary
        """
        logger.info("=" * 60)
        logger.info("CORRELATION & ROOT CAUSE ANALYSIS  (D3)")
        logger.info("=" * 60)

        # ── Build joined daily dataset ─────────────────────────────────────────
        logger.info("\n[0/5] Joining complaint + KPI aggregates ...")
        joined = self._build_joined(complaint_agg, kpi_agg)
        logger.info(f"  Joined dataset: {joined.shape[0]:,} rows")

        # ── 1. Correlation matrices ────────────────────────────────────────────
        logger.info("\n[1/5] Computing Pearson & Spearman correlation matrices ...")
        pearson_mat, spearman_mat, top_corr = self._correlation_matrices(joined)

        # ── 2. KPI threshold detection ─────────────────────────────────────────
        logger.info("\n[2/5] Detecting KPI complaint-spike thresholds ...")
        thresholds = self._threshold_detection(joined)

        # ── 3. Granger causality ───────────────────────────────────────────────
        logger.info("\n[3/5] Running Granger causality tests ...")
        granger_results = self._granger_causality(complaint_agg, kpi_agg)

        # ── 4. QoE degradation event analysis ─────────────────────────────────
        logger.info("\n[4/5] Analysing QoE degradation events ...")
        qoe_events = self._qoe_event_analysis(joined)

        # ── 5. Cross-correlation functions ────────────────────────────────────
        logger.info("\n[5/5] Computing KPI–complaint cross-correlations ...")
        ccf_results = self._cross_correlation(complaint_agg, kpi_agg)

        # ── Summary ────────────────────────────────────────────────────────────
        summary = self._build_summary(top_corr, thresholds, granger_results, qoe_events)
        self._print_summary(summary)

        # ── Save D3 report ─────────────────────────────────────────────────────
        self._save_report(top_corr, thresholds, granger_results,
                          qoe_events, ccf_results)

        return {
            "pearson_matrix":    pearson_mat,
            "spearman_matrix":   spearman_mat,
            "top_correlations":  top_corr,
            "thresholds":        thresholds,
            "granger_results":   granger_results,
            "qoe_event_analysis":qoe_events,
            "ccf_results":       ccf_results,
            "joined":            joined,
            "summary":           summary,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 0. JOIN
    # ─────────────────────────────────────────────────────────────────────────

    def _build_joined(self,
                      complaint_agg: pd.DataFrame,
                      kpi_agg:       pd.DataFrame) -> pd.DataFrame:
        """
        Inner join complaints + KPI at (region, date).
        Adds composite qoe_score_mean if not present.
        """
        ca = complaint_agg.copy()
        ka = kpi_agg.copy()

        ca["date"] = pd.to_datetime(ca["date"])
        ka["date"] = pd.to_datetime(ka["date"])

        # Keep only _mean KPI columns for clarity
        kpi_mean_cols = [c for c in ka.columns
                         if c.endswith("_mean") and "roll" not in c]
        ka_slim = ka[["region", "date"] + kpi_mean_cols].copy()

        # Derive composite QoE if missing
        if "qoe_score_mean" not in ka_slim.columns:
            if "data_qoe_score_mean" in ka_slim.columns and \
               "voice_qoe_score_mean" in ka_slim.columns:
                ka_slim["qoe_score_mean"] = (
                    0.55 * ka_slim["data_qoe_score_mean"] +
                    0.45 * ka_slim["voice_qoe_score_mean"]
                ).round(2)

        # Complaint columns to keep
        complaint_cols = ["region", "date", "total_complaints",
                          "complaints_data", "complaints_voice",
                          "high_priority_complaints", "complaint_spike_flag"]
        cat_cols = [c for c in ca.columns if c.startswith("cat_")]
        ca_slim = ca[complaint_cols + cat_cols].copy()

        joined = ca_slim.merge(ka_slim, on=["region", "date"], how="inner")
        return joined.sort_values(["region", "date"]).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CORRELATION MATRICES
    # ─────────────────────────────────────────────────────────────────────────

    def _correlation_matrices(self,
                              joined: pd.DataFrame
                              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute Pearson and Spearman correlations between
        all KPI _mean columns and total_complaints.
        Returns both full matrices and a ranked top-correlation table.
        """
        kpi_mean_cols = [c for c in joined.columns
                         if c.endswith("_mean") and "roll" not in c]
        target = "total_complaints"

        # Full matrices (KPI × KPI + target)
        analysis_cols = kpi_mean_cols + [target]
        df_num = joined[analysis_cols].dropna()

        pearson_mat  = df_num.corr(method="pearson")
        spearman_mat = df_num.corr(method="spearman")

        # Top correlations with target — ranked table
        rows = []
        for kpi in kpi_mean_cols:
            pair = joined[[kpi, target]].dropna()
            if len(pair) < 10:
                continue
            p_r,  p_p  = pearsonr(pair[kpi], pair[target])
            sp_r, sp_p = spearmanr(pair[kpi], pair[target])
            rows.append({
                "kpi":             kpi,
                "kpi_label":       KPI_LABELS.get(
                                       kpi.replace("_mean", ""), kpi),
                "pearson_r":       round(p_r,  4),
                "pearson_p":       round(p_p,  4),
                "spearman_r":      round(sp_r, 4),
                "spearman_p":      round(sp_p, 4),
                "pearson_sig":     "✓" if p_p  < 0.05 else "✗",
                "spearman_sig":    "✓" if sp_p < 0.05 else "✗",
                "abs_pearson":     abs(p_r),
            })

        top_corr = (pd.DataFrame(rows)
                      .sort_values("abs_pearson", ascending=False)
                      .reset_index(drop=True)
                      .drop(columns=["abs_pearson"]))

        # Log top 5
        logger.info("  Top 5 KPIs correlated with total_complaints (Pearson):")
        for _, row in top_corr.head(5).iterrows():
            logger.info(
                f"    {row['kpi_label']:<35} r={row['pearson_r']:+.3f}  "
                f"p={row['pearson_p']:.4f}  {row['pearson_sig']}"
            )

        return pearson_mat, spearman_mat, top_corr

    # ─────────────────────────────────────────────────────────────────────────
    # 2. THRESHOLD DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _threshold_detection(self, joined: pd.DataFrame) -> pd.DataFrame:
        """
        For each KPI, find the value threshold that best separates
        spike days from normal days using a depth-1 decision tree.

        Returns a table with:
          kpi, threshold_value, direction (above/below),
          spike_rate_below, spike_rate_above, gini_improvement
        """
        kpi_mean_cols = [c for c in joined.columns
                         if c.endswith("_mean") and "roll" not in c]
        spike_col = "complaint_spike_flag"
        if spike_col not in joined.columns:
            logger.warning("  complaint_spike_flag not found — skipping thresholds")
            return pd.DataFrame()

        rows = []
        for kpi in kpi_mean_cols:
            pair = joined[[kpi, spike_col]].dropna()
            if len(pair) < 20 or pair[spike_col].sum() < 5:
                continue

            X = pair[[kpi]].values
            y = pair[spike_col].values.astype(int)

            tree = DecisionTreeClassifier(max_depth=1, random_state=42)
            tree.fit(X, y)

            threshold  = float(tree.tree_.threshold[0])
            gini_improv= float(tree.tree_.impurity[0] -
                               (tree.tree_.n_node_samples[1] / len(y)) *
                                tree.tree_.impurity[1] -
                               (tree.tree_.n_node_samples[2] / len(y)) *
                                tree.tree_.impurity[2])

            # Spike rates on each side of threshold
            below = pair[pair[kpi] <= threshold][spike_col]
            above = pair[pair[kpi] >  threshold][spike_col]
            rate_below = float(below.mean()) if len(below) > 0 else 0.0
            rate_above = float(above.mean()) if len(above) > 0 else 0.0

            # Direction: which side has higher spike rate?
            direction = "above" if rate_above > rate_below else "below"

            rows.append({
                "kpi":              kpi,
                "kpi_label":        KPI_LABELS.get(kpi.replace("_mean",""), kpi),
                "threshold_value":  round(threshold, 3),
                "direction":        direction,
                "spike_rate_below": round(rate_below, 3),
                "spike_rate_above": round(rate_above, 3),
                "gini_improvement": round(gini_improv, 5),
                "n_samples":        len(pair),
            })

        thresholds = (pd.DataFrame(rows)
                        .sort_values("gini_improvement", ascending=False)
                        .reset_index(drop=True))

        logger.info("  Top 5 KPI thresholds by Gini improvement:")
        for _, row in thresholds.head(5).iterrows():
            logger.info(
                f"    {row['kpi_label']:<35} "
                f"threshold={row['threshold_value']:.2f}  "
                f"direction={row['direction']}  "
                f"spike_rate_above={row['spike_rate_above']:.1%}"
            )
        return thresholds

    # ─────────────────────────────────────────────────────────────────────────
    # 3. GRANGER CAUSALITY
    # ─────────────────────────────────────────────────────────────────────────

    def _granger_causality(self,
                           complaint_agg: pd.DataFrame,
                           kpi_agg:       pd.DataFrame,
                           max_lag:       int = 7) -> pd.DataFrame:
        """
        For each (region, KPI) pair, test whether past KPI values
        Granger-cause future complaint counts.

        Uses a single representative region (Tunis) for speed,
        then optionally all regions.

        Returns table: region, kpi, best_lag, min_p_value, is_significant
        """
        if not STATSMODELS_OK:
            logger.warning("  statsmodels not available — returning empty results")
            return pd.DataFrame()

        ca = complaint_agg.copy()
        ka = kpi_agg.copy()
        ca["date"] = pd.to_datetime(ca["date"])
        ka["date"] = pd.to_datetime(ka["date"])

        kpi_mean_cols = [c for c in ka.columns
                         if c.endswith("_mean") and "roll" not in c][:6]  # top 6 for speed

        rows = []
        regions_to_test = sorted(ca["region"].unique())

        for region in regions_to_test:
            ca_r = ca[ca["region"] == region].sort_values("date")
            ka_r = ka[ka["region"] == region].sort_values("date")

            merged = ca_r[["date", "total_complaints"]].merge(
                ka_r[["date"] + kpi_mean_cols], on="date", how="inner"
            ).dropna()

            if len(merged) < max_lag * 4:
                continue

            for kpi in kpi_mean_cols:
                try:
                    # Granger test needs [target, cause] array
                    data = merged[["total_complaints", kpi]].values
                    gc   = grangercausalitytests(data, maxlag=max_lag, verbose=False)

                    # Find lag with lowest p-value (F-test)
                    p_vals = {
                        lag: gc[lag][0]["ssr_ftest"][1]
                        for lag in range(1, max_lag + 1)
                    }
                    best_lag = min(p_vals, key=p_vals.get)
                    min_p    = p_vals[best_lag]

                    rows.append({
                        "region":         region,
                        "kpi":            kpi,
                        "kpi_label":      KPI_LABELS.get(kpi.replace("_mean",""), kpi),
                        "best_lag_days":  best_lag,
                        "min_p_value":    round(min_p, 5),
                        "is_significant": min_p < 0.05,
                        "interpretation": (
                            f"{kpi.replace('_mean','').replace('_',' ').title()} "
                            f"Granger-causes complaints with {best_lag}-day lag"
                            if min_p < 0.05 else "Not significant"
                        ),
                    })
                except Exception:
                    continue

        granger_df = (pd.DataFrame(rows)
                        .sort_values("min_p_value")
                        .reset_index(drop=True))

        n_sig = int(granger_df["is_significant"].sum()) if not granger_df.empty else 0
        logger.info(f"  Granger: {n_sig} significant KPI→complaint causal links found")
        if not granger_df.empty:
            for _, row in granger_df[granger_df["is_significant"]].head(5).iterrows():
                logger.info(
                    f"    [{row['region']}] {row['kpi_label']:<35} "
                    f"lag={row['best_lag_days']}d  p={row['min_p_value']:.4f}"
                )
        return granger_df

    # ─────────────────────────────────────────────────────────────────────────
    # 4. QoE DEGRADATION EVENT ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def _qoe_event_analysis(self, joined: pd.DataFrame) -> pd.DataFrame:
        """
        Split region-days into:
          - Degraded: QoE score mean < QOE_THRESHOLD
          - Normal:   QoE score mean >= QOE_THRESHOLD

        Compare complaint counts between the two groups.
        Also compute % complaint increase during degraded periods.
        """
        qoe_col = next(
            (c for c in ["qoe_score_mean", "data_qoe_score_mean"]
             if c in joined.columns), None
        )
        if qoe_col is None:
            logger.warning("  No QoE column found — skipping QoE event analysis")
            return pd.DataFrame()

        # Adaptive threshold: use config value, but fall back to
        # 25th percentile if no days fall below the config threshold
        threshold = QOE_THRESHOLD
        global_p25 = joined[qoe_col].quantile(0.25)
        if (joined[qoe_col] < threshold).sum() < 10:
            threshold = round(float(global_p25), 1)
            logger.info(
                f"  No days below QoE={QOE_THRESHOLD} in data — "
                f"using adaptive threshold (p25={threshold})"
            )

        rows = []
        for region, grp in joined.groupby("region"):
            grp = grp.copy()
            degraded = grp[grp[qoe_col] < threshold]
            normal   = grp[grp[qoe_col] >= threshold]

            if len(degraded) < 3 or len(normal) < 3:
                continue

            mean_deg  = degraded["total_complaints"].mean()
            mean_norm = normal["total_complaints"].mean()
            pct_incr  = ((mean_deg - mean_norm) / (mean_norm + 1e-9)) * 100

            # Mann-Whitney U test (non-parametric)
            stat, p_val = stats.mannwhitneyu(
                degraded["total_complaints"],
                normal["total_complaints"],
                alternative="greater"
            )

            rows.append({
                "region":              region,
                "n_degraded_days":     len(degraded),
                "n_normal_days":       len(normal),
                "mean_complaints_degraded": round(mean_deg, 2),
                "mean_complaints_normal":   round(mean_norm, 2),
                "pct_increase":        round(pct_incr, 1),
                "mannwhitney_stat":    round(stat, 2),
                "p_value":             round(p_val, 5),
                "is_significant":      p_val < 0.05,
                "avg_qoe_degraded":    round(degraded[qoe_col].mean(), 1),
                "avg_qoe_normal":      round(normal[qoe_col].mean(), 1),
            })

        result = (pd.DataFrame(rows)
                    .sort_values("pct_increase", ascending=False)
                    .reset_index(drop=True))

        logger.info(f"  QoE events: {int(result['is_significant'].sum())} "
                    f"regions show significant complaint increase during QoE degradation")
        for _, row in result.head(5).iterrows():
            logger.info(
                f"    {row['region']:<12} "
                f"+{row['pct_increase']:.1f}% complaints when QoE < {QOE_THRESHOLD}  "
                f"p={row['p_value']:.4f}  {'✓' if row['is_significant'] else '✗'}"
            )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CROSS-CORRELATION FUNCTION (CCF)
    # ─────────────────────────────────────────────────────────────────────────

    def _cross_correlation(self,
                           complaint_agg: pd.DataFrame,
                           kpi_agg:       pd.DataFrame,
                           max_lag:       int = 14) -> pd.DataFrame:
        """
        Compute time-lagged cross-correlation between KPI means
        and total complaint counts per region.

        For each KPI, finds the lag (0–14 days) at which the
        correlation with future complaints is highest.

        This answers: "How many days after a KPI drops do complaints peak?"
        """
        ca = complaint_agg.copy()
        ka = kpi_agg.copy()
        ca["date"] = pd.to_datetime(ca["date"])
        ka["date"] = pd.to_datetime(ka["date"])

        kpi_mean_cols = [c for c in ka.columns
                         if c.endswith("_mean") and "roll" not in c]

        rows = []
        for region in sorted(ca["region"].unique()):
            ca_r = ca[ca["region"] == region].sort_values("date")
            ka_r = ka[ka["region"] == region].sort_values("date")
            merged = ca_r[["date","total_complaints"]].merge(
                ka_r[["date"] + kpi_mean_cols], on="date", how="inner"
            ).dropna().reset_index(drop=True)

            if len(merged) < max_lag + 10:
                continue

            complaints = merged["total_complaints"].values

            for kpi in kpi_mean_cols:
                kpi_series = merged[kpi].values
                best_lag, best_corr = 0, 0.0

                for lag in range(0, max_lag + 1):
                    if lag == 0:
                        x, y = kpi_series, complaints
                    else:
                        x = kpi_series[:-lag]
                        y = complaints[lag:]

                    if len(x) < 10:
                        continue
                    r, _ = pearsonr(x, y)
                    if abs(r) > abs(best_corr):
                        best_corr = r
                        best_lag  = lag

                rows.append({
                    "region":          region,
                    "kpi":             kpi,
                    "kpi_label":       KPI_LABELS.get(kpi.replace("_mean",""), kpi),
                    "best_lag_days":   best_lag,
                    "peak_correlation":round(best_corr, 4),
                    "abs_correlation": abs(best_corr),
                    "direction":       "inverse" if best_corr < 0 else "direct",
                })

        ccf_df = (pd.DataFrame(rows)
                    .sort_values("abs_correlation", ascending=False)
                    .reset_index(drop=True))

        logger.info("  Top CCF results (KPI → complaint, peak lag):")
        for _, row in ccf_df.head(5).iterrows():
            logger.info(
                f"    [{row['region']:<10}] {row['kpi_label']:<35} "
                f"lag={row['best_lag_days']}d  r={row['peak_correlation']:+.3f} ({row['direction']})"
            )
        return ccf_df

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY & SAVE
    # ─────────────────────────────────────────────────────────────────────────

    def _build_summary(self, top_corr, thresholds,
                       granger, qoe_events) -> dict:
        summary = {}

        if not top_corr.empty:
            top3 = top_corr.head(3)
            summary["top_correlated_kpis"] = top3["kpi_label"].tolist()
            summary["top_pearson_r"]        = top3["pearson_r"].tolist()

        if not thresholds.empty:
            top_thresh = thresholds.iloc[0]
            summary["most_predictive_threshold"] = {
                "kpi":       top_thresh["kpi_label"],
                "threshold": top_thresh["threshold_value"],
                "direction": top_thresh["direction"],
            }

        if not granger.empty:
            sig = granger[granger["is_significant"]]
            summary["granger_significant_pairs"] = len(sig)
            if not sig.empty:
                best = sig.iloc[0]
                summary["strongest_granger_cause"] = {
                    "kpi":     best["kpi_label"],
                    "lag":     best["best_lag_days"],
                    "p_value": best["min_p_value"],
                }

        if not qoe_events.empty:
            sig_qoe = qoe_events[qoe_events["is_significant"]]
            summary["qoe_degradation_impact"] = {
                "significant_regions": len(sig_qoe),
                "max_pct_increase":    float(qoe_events["pct_increase"].max()),
                "avg_pct_increase":    float(qoe_events["pct_increase"].mean().round(1)),
            }

        return summary

    def _print_summary(self, summary: dict):
        logger.info("\n" + "=" * 60)
        logger.info("  D3 CORRELATION STUDY — KEY FINDINGS")
        logger.info("=" * 60)

        if "top_correlated_kpis" in summary:
            logger.info("  Most correlated KPIs with complaints:")
            for kpi, r in zip(summary["top_correlated_kpis"],
                               summary["top_pearson_r"]):
                logger.info(f"    {kpi:<40} r={r:+.3f}")

        if "most_predictive_threshold" in summary:
            t = summary["most_predictive_threshold"]
            logger.info(f"\n  Best threshold: {t['kpi']} "
                        f"{t['direction']} {t['threshold']}")

        if "granger_significant_pairs" in summary:
            logger.info(f"\n  Granger causality: "
                        f"{summary['granger_significant_pairs']} significant KPI→complaint pairs")
            if "strongest_granger_cause" in summary:
                g = summary["strongest_granger_cause"]
                logger.info(f"    Strongest: {g['kpi']} "
                            f"(lag={g['lag']}d, p={g['p_value']:.4f})")

        if "qoe_degradation_impact" in summary:
            q = summary["qoe_degradation_impact"]
            logger.info(f"\n  QoE degradation impact:")
            logger.info(f"    Significant regions : {q['significant_regions']}")
            logger.info(f"    Max complaint increase: +{q['max_pct_increase']:.1f}%")
            logger.info(f"    Avg complaint increase: +{q['avg_pct_increase']:.1f}%")
        logger.info("=" * 60)

    def _save_report(self, top_corr, thresholds,
                     granger, qoe_events, ccf_results):
        """Save all result tables to reports/exports/ as CSV."""
        saves = {
            "d3_correlation_rankings.csv":   top_corr,
            "d3_kpi_thresholds.csv":         thresholds,
            "d3_granger_causality.csv":      granger,
            "d3_qoe_event_analysis.csv":     qoe_events,
            "d3_cross_correlation.csv":      ccf_results,
        }
        for fname, df in saves.items():
            if not df.empty:
                path = REPORTS_DIR / fname
                df.to_csv(path, index=False)
        logger.info(f"  D3 report tables saved → {REPORTS_DIR}")