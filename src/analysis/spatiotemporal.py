"""
Spatio-Temporal Analysis Module
=================================
Deliverable D3 (part 1) — Geographic and time-based complaint pattern detection.

Six analysis sections:

  1. Geographic hotspot mapping
     — Complaint density by region and city
     — Cell-level hotspot ranking
     — Folium interactive heatmap (saved as HTML)

  2. Temporal pattern analysis
     — Hourly distribution (peak hour detection)
     — Day-of-week cycle
     — Monthly trends and seasonality

  3. Hour × Day-of-week heatmap
     — 2D intensity map: when are complaints highest?

  4. Anomaly burst detection
     — Days where complaint volume exceeds mean + 2σ per region
     — Burst characterisation: duration, magnitude, service type

  5. Service-type segmentation by region
     — Which regions suffer most from Data vs Voice vs SMS issues

  6. Cell-level hotspot analysis
     — Top complaint-generating cells
     — Cell complaint rate over time

Usage (from notebook):
    from src.analysis.spatiotemporal import SpatioTemporalAnalyser
    st = SpatioTemporalAnalyser()
    results = st.run(complaints_clean, complaint_agg, kpi_agg)
"""

from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

REPORTS_DIR = Path(cfg["paths"]["reports"]) / "exports"
FIGURES_DIR = Path(cfg["paths"]["figures"])
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = cfg["data"]["regions"]

# Region centroids (lat, lon)
REGION_CENTROIDS = {
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

PEAK_HOURS     = {8, 9, 12, 13, 17, 18, 19, 20}
DOW_ORDER      = ["Monday","Tuesday","Wednesday","Thursday",
                  "Friday","Saturday","Sunday"]


class SpatioTemporalAnalyser:
    """
    Full spatio-temporal analysis pipeline.
    All outputs are DataFrames + saved figures/HTML files.
    """

    def run(self,
            complaints_clean: pd.DataFrame,
            complaint_agg:    pd.DataFrame,
            kpi_agg:          pd.DataFrame) -> dict:
        """
        Run all six analysis sections.

        Returns
        -------
        dict with keys:
            regional_hotspots, cell_hotspots,
            hourly_patterns, dow_patterns, monthly_trends,
            hour_dow_heatmap,
            anomaly_bursts,
            service_by_region,
            summary
        """
        logger.info("=" * 60)
        logger.info("SPATIO-TEMPORAL ANALYSIS")
        logger.info("=" * 60)

        cc = complaints_clean.copy()
        ca = complaint_agg.copy()
        cc["timestamp"] = pd.to_datetime(cc["timestamp"])
        ca["date"]      = pd.to_datetime(ca["date"])

        # ── 1. Geographic hotspots ─────────────────────────────────────────
        logger.info("\n[1/6] Geographic hotspot mapping ...")
        regional_hotspots, cell_hotspots = self._geographic_hotspots(cc, kpi_agg)

        # ── 2. Temporal patterns ───────────────────────────────────────────
        logger.info("\n[2/6] Temporal pattern analysis ...")
        hourly, dow, monthly = self._temporal_patterns(cc)

        # ── 3. Hour × DoW heatmap ──────────────────────────────────────────
        logger.info("\n[3/6] Building hour × day-of-week heatmap ...")
        hour_dow = self._hour_dow_heatmap(cc)

        # ── 4. Anomaly burst detection ─────────────────────────────────────
        logger.info("\n[4/6] Detecting anomaly bursts ...")
        bursts = self._anomaly_bursts(ca)

        # ── 5. Service-type by region ──────────────────────────────────────
        logger.info("\n[5/6] Service-type segmentation by region ...")
        service_by_region = self._service_by_region(cc)

        # ── 6. Folium interactive map ──────────────────────────────────────
        logger.info("\n[6/6] Building interactive Folium map ...")
        self._build_folium_map(regional_hotspots, kpi_agg)

        # ── Summary ────────────────────────────────────────────────────────
        summary = self._build_summary(
            regional_hotspots, hourly, bursts, service_by_region
        )
        self._print_summary(summary)
        self._save_csv(regional_hotspots, cell_hotspots, hourly,
                       dow, monthly, bursts, service_by_region)

        return {
            "regional_hotspots":  regional_hotspots,
            "cell_hotspots":      cell_hotspots,
            "hourly_patterns":    hourly,
            "dow_patterns":       dow,
            "monthly_trends":     monthly,
            "hour_dow_heatmap":   hour_dow,
            "anomaly_bursts":     bursts,
            "service_by_region":  service_by_region,
            "summary":            summary,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 1. GEOGRAPHIC HOTSPOTS
    # ─────────────────────────────────────────────────────────────────────

    def _geographic_hotspots(self,
                              cc: pd.DataFrame,
                              kpi_agg: pd.DataFrame
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Regional hotspot table:
          - total complaints, complaint rate (per 1000 per day)
          - dominant category, dominant service type
          - average QoE score (joined from kpi_agg)
          - hotspot rank

        Cell-level hotspot table:
          - top 20 cells by complaint count
        """
        # ── Regional ─────────────────────────────────────────────────────
        n_days = cc["timestamp"].dt.date.nunique()

        regional = (
            cc.groupby("region")
              .agg(
                  total_complaints    = ("case_id",           "count"),
                  unique_cells        = ("cell_id",           "nunique"),
                  dominant_category   = ("complaint_category",
                                         lambda x: x.value_counts().index[0]),
                  dominant_service    = ("service_type",
                                         lambda x: x.value_counts().index[0]),
                  high_priority_count = ("priority_encoded",
                                         lambda x: (x >= 2).sum()),
                  vip_count           = ("segment_encoded",
                                         lambda x: (x >= 2).sum()),
                  lat                 = ("latitude",  "mean"),
                  lon                 = ("longitude", "mean"),
              )
              .reset_index()
        )
        regional["complaint_rate_per_day"] = (
            regional["total_complaints"] / n_days
        ).round(2)
        regional["high_priority_pct"] = (
            regional["high_priority_count"] / regional["total_complaints"] * 100
        ).round(1)

        # Join average QoE
        ka = kpi_agg.copy()
        ka["date"] = pd.to_datetime(ka["date"])
        qoe_col = ("qoe_score_mean"
                   if "qoe_score_mean" in ka.columns
                   else "data_qoe_score_mean")
        if qoe_col in ka.columns:
            avg_qoe = (ka.groupby("region")[qoe_col]
                          .mean().round(2).reset_index()
                          .rename(columns={qoe_col: "avg_qoe_score"}))
            regional = regional.merge(avg_qoe, on="region", how="left")
        else:
            regional["avg_qoe_score"] = np.nan

        regional["hotspot_rank"] = (
            regional["total_complaints"]
            .rank(ascending=False).astype(int)
        )
        regional = regional.sort_values("total_complaints", ascending=False)

        logger.info("  Regional hotspot ranking:")
        for _, row in regional.head(5).iterrows():
            logger.info(
                f"    #{int(row['hotspot_rank'])} {row['region']:<12} "
                f"{int(row['total_complaints']):>6,} complaints  "
                f"QoE={row.get('avg_qoe_score', np.nan):.1f}"
            )

        # ── Cell-level ────────────────────────────────────────────────────
        cells = (
            cc[cc["cell_id"] != "UNKNOWN"]
              .groupby(["cell_id", "region"])
              .agg(
                  total_complaints  = ("case_id",           "count"),
                  dominant_category = ("complaint_category",
                                       lambda x: x.value_counts().index[0]),
                  lat               = ("latitude",  "mean"),
                  lon               = ("longitude", "mean"),
              )
              .reset_index()
              .sort_values("total_complaints", ascending=False)
              .head(20)
              .reset_index(drop=True)
        )
        cells["cell_rank"] = cells.index + 1
        logger.info(f"  Top cell: {cells.iloc[0]['cell_id']} "
                    f"({int(cells.iloc[0]['total_complaints'])} complaints)")

        return regional, cells

    # ─────────────────────────────────────────────────────────────────────
    # 2. TEMPORAL PATTERNS
    # ─────────────────────────────────────────────────────────────────────

    def _temporal_patterns(self,
                            cc: pd.DataFrame
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Hourly, day-of-week, and monthly complaint distributions.
        Each table includes absolute counts and normalised rates.
        """
        total = len(cc)

        # ── Hourly ───────────────────────────────────────────────────────
        hourly = (
            cc.groupby("hour")
              .size()
              .reset_index(name="complaint_count")
        )
        hourly["pct"]        = (hourly["complaint_count"] / total * 100).round(2)
        hourly["is_peak"]    = hourly["hour"].isin(PEAK_HOURS).astype(int)
        hourly["period"]     = hourly["hour"].apply(_hour_label)

        peak_hours_data   = hourly[hourly["is_peak"] == 1]["complaint_count"].sum()
        offpeak_hours_data= hourly[hourly["is_peak"] == 0]["complaint_count"].sum()
        hourly["peak_vs_offpeak_ratio"] = round(
            peak_hours_data / max(offpeak_hours_data, 1), 2
        )
        logger.info(
            f"  Peak/off-peak ratio: {hourly['peak_vs_offpeak_ratio'].iloc[0]:.2f}x  "
            f"| Peak hours: {sorted(PEAK_HOURS)}"
        )

        # ── Day of week ───────────────────────────────────────────────────
        dow = (
            cc.groupby("day_of_week")
              .size()
              .reindex(DOW_ORDER)
              .reset_index(name="complaint_count")
        )
        dow["pct"]        = (dow["complaint_count"] / total * 100).round(2)
        dow["is_weekend"] = dow["day_of_week"].isin(["Saturday","Sunday"]).astype(int)

        # ── Monthly ───────────────────────────────────────────────────────
        cc["month_label"] = (
            cc["timestamp"].dt.to_period("M").astype(str)
        )
        monthly = (
            cc.groupby(["month_label", "service_type"])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )
        monthly["total"] = monthly.drop(columns=["month_label"]).sum(axis=1)
        monthly = monthly.sort_values("month_label").reset_index(drop=True)

        return hourly, dow, monthly

    # ─────────────────────────────────────────────────────────────────────
    # 3. HOUR × DAY-OF-WEEK HEATMAP
    # ─────────────────────────────────────────────────────────────────────

    def _hour_dow_heatmap(self, cc: pd.DataFrame) -> pd.DataFrame:
        """
        2D pivot: rows = day-of-week, columns = hour,
        values = complaint count.
        """
        pivot = (
            cc.groupby(["day_of_week", "hour"])
              .size()
              .unstack(fill_value=0)
              .reindex(DOW_ORDER)
        )
        # Peak cell
        max_idx = np.unravel_index(pivot.values.argmax(), pivot.shape)
        peak_day  = pivot.index[max_idx[0]]
        peak_hour = pivot.columns[max_idx[1]]
        logger.info(
            f"  Peak complaint slot: {peak_day} at {peak_hour:02d}:00 "
            f"({pivot.values.max():,} complaints)"
        )
        return pivot

    # ─────────────────────────────────────────────────────────────────────
    # 4. ANOMALY BURST DETECTION
    # ─────────────────────────────────────────────────────────────────────

    def _anomaly_bursts(self, ca: pd.DataFrame) -> pd.DataFrame:
        """
        For each region, identify burst periods:
          - A burst day = complaint_spike_flag == 1
          - Consecutive burst days = a burst event
          - Characterise by: duration, total complaints, magnitude (z-score)
        """
        rows = []
        for region, grp in ca.groupby("region"):
            grp   = grp.sort_values("date").reset_index(drop=True)
            mean_ = grp["total_complaints"].mean()
            std_  = grp["total_complaints"].std()
            if std_ == 0:
                continue

            grp["zscore"]       = (grp["total_complaints"] - mean_) / std_
            grp["burst"]        = (grp["complaint_spike_flag"] == 1).astype(int)
            grp["burst_group"]  = (grp["burst"] != grp["burst"].shift()).cumsum()

            for gid, burst_df in grp[grp["burst"] == 1].groupby("burst_group"):
                rows.append({
                    "region":             region,
                    "burst_start":        burst_df["date"].min(),
                    "burst_end":          burst_df["date"].max(),
                    "duration_days":      len(burst_df),
                    "total_complaints":   int(burst_df["total_complaints"].sum()),
                    "peak_complaints":    int(burst_df["total_complaints"].max()),
                    "mean_zscore":        round(float(burst_df["zscore"].mean()), 2),
                    "peak_zscore":        round(float(burst_df["zscore"].max()),  2),
                    "severity":           (
                        "Critical" if burst_df["zscore"].max() > 3 else
                        "High"     if burst_df["zscore"].max() > 2 else
                        "Medium"
                    ),
                })

        bursts = (
            pd.DataFrame(rows)
              .sort_values("peak_zscore", ascending=False)
              .reset_index(drop=True)
        ) if rows else pd.DataFrame()

        if not bursts.empty:
            logger.info(
                f"  Burst events detected: {len(bursts)}  "
                f"| Critical: {(bursts['severity']=='Critical').sum()}  "
                f"| High: {(bursts['severity']=='High').sum()}"
            )
            logger.info("  Top 3 bursts:")
            for _, row in bursts.head(3).iterrows():
                logger.info(
                    f"    {row['region']:<12} "
                    f"{str(row['burst_start'])[:10]} → "
                    f"{str(row['burst_end'])[:10]}  "
                    f"peak_z={row['peak_zscore']:.2f}  "
                    f"[{row['severity']}]"
                )
        return bursts

    # ─────────────────────────────────────────────────────────────────────
    # 5. SERVICE TYPE BY REGION
    # ─────────────────────────────────────────────────────────────────────

    def _service_by_region(self, cc: pd.DataFrame) -> pd.DataFrame:
        """
        For each region: breakdown of complaints by service type,
        normalised as % of regional total.
        """
        service = (
            cc.groupby(["region", "service_type"])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )
        totals = service.drop(columns="region").sum(axis=1)
        for col in service.columns[1:]:
            service[f"{col}_pct"] = (service[col] / totals * 100).round(1)

        # Dominant service per region
        svc_cols = [c for c in service.columns
                    if c not in ("region",) and not c.endswith("_pct")]
        service["dominant_service"] = service[svc_cols].idxmax(axis=1)

        logger.info("  Service type breakdown by region:")
        for _, row in service.iterrows():
            pct_cols = [c for c in service.columns if c.endswith("_pct")]
            parts = "  ".join(
                f"{c.replace('_pct','')}: {row[c]:.0f}%"
                for c in pct_cols
            )
            logger.info(f"    {row['region']:<12} {parts}")

        return service

    # ─────────────────────────────────────────────────────────────────────
    # 6. FOLIUM INTERACTIVE MAP
    # ─────────────────────────────────────────────────────────────────────

    def _build_folium_map(self,
                           regional: pd.DataFrame,
                           kpi_agg:  pd.DataFrame) -> None:
        """
        Build and save two Folium maps:
          1. Complaint volume choropleth (circle markers)
          2. Individual complaint point density heatmap
        """
        try:
            import folium
            from folium.plugins import HeatMap

            max_complaints = regional["total_complaints"].max()

            # ── Map 1: Regional bubble map ────────────────────────────────
            m1 = folium.Map(
                location=[35.5, 10.0], zoom_start=7,
                tiles="CartoDB dark_matter"
            )

            for _, row in regional.iterrows():
                region = row["region"]
                if region not in REGION_CENTROIDS:
                    continue
                lat, lon  = REGION_CENTROIDS[region]
                count     = int(row["total_complaints"])
                radius    = 15 + (count / max_complaints) * 40
                qoe       = row.get("avg_qoe_score", 70)
                color     = ("#2ecc71" if qoe >= 80
                              else "#f39c12" if qoe >= 60
                              else "#e74c3c")

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color=color, fill=True,
                    fill_color=color, fill_opacity=0.55,
                    weight=2,
                    tooltip=folium.Tooltip(
                        f"<b>{region}</b><br>"
                        f"Complaints: {count:,}<br>"
                        f"QoE: {qoe:.1f}<br>"
                        f"Dominant: {row.get('dominant_category','N/A')}"
                    )
                ).add_to(m1)

                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:9px;color:white;'
                             f'font-weight:bold;text-align:center;'
                             f'text-shadow:1px 1px 2px black;">'
                             f'{region}<br>{count:,}</div>',
                        icon_size=(75, 28),
                        icon_anchor=(37, 14)
                    )
                ).add_to(m1)

            # Legend
            legend_html = """
            <div style="position:fixed;bottom:30px;left:30px;
                        background:rgba(0,0,0,0.7);padding:12px;
                        border-radius:8px;color:white;font-size:12px;">
              <b>QoE Colour Scale</b><br>
              <span style="color:#2ecc71">●</span> Good (≥ 80)<br>
              <span style="color:#f39c12">●</span> Fair (60–79)<br>
              <span style="color:#e74c3c">●</span> Poor (< 60)<br>
              <i>Circle size = complaint volume</i>
            </div>"""
            m1.get_root().html.add_child(folium.Element(legend_html))

            map1_path = REPORTS_DIR / "st_regional_map.html"
            m1.save(str(map1_path))
            logger.info(f"  Regional map saved → {map1_path}")

        except ImportError:
            logger.warning("  folium not installed — skipping map generation")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY & SAVE
    # ─────────────────────────────────────────────────────────────────────

    def _build_summary(self, regional, hourly, bursts, service) -> dict:
        summary = {}

        if not regional.empty:
            top = regional.iloc[0]
            summary["top_hotspot_region"]    = top["region"]
            summary["top_hotspot_complaints"]= int(top["total_complaints"])
            summary["top_hotspot_category"]  = top.get("dominant_category","N/A")

        if not hourly.empty:
            peak_h = int(hourly.loc[hourly["complaint_count"].idxmax(), "hour"])
            summary["peak_hour"]             = peak_h
            summary["peak_hour_label"]       = _hour_label(peak_h)
            summary["peak_offpeak_ratio"]    = float(
                hourly["peak_vs_offpeak_ratio"].iloc[0]
            )

        if not bursts.empty:
            summary["total_burst_events"]    = len(bursts)
            summary["critical_bursts"]       = int(
                (bursts["severity"] == "Critical").sum()
            )
            summary["most_bursty_region"]    = (
                bursts.groupby("region")
                      .size().idxmax()
            )

        if not service.empty:
            pct_cols = [c for c in service.columns if c.endswith("_pct")]
            dominant_svc_per_region = service.set_index("region")["dominant_service"]
            summary["dominant_service_by_region"] = dominant_svc_per_region.to_dict()

        return summary

    def _print_summary(self, summary: dict):
        logger.info("\n" + "=" * 60)
        logger.info("  SPATIO-TEMPORAL ANALYSIS — KEY FINDINGS")
        logger.info("=" * 60)
        if "top_hotspot_region" in summary:
            logger.info(
                f"  Top hotspot    : {summary['top_hotspot_region']} "
                f"({summary['top_hotspot_complaints']:,} complaints)"
            )
            logger.info(f"  Dominant issue : {summary['top_hotspot_category']}")
        if "peak_hour" in summary:
            logger.info(
                f"  Peak hour      : {summary['peak_hour']:02d}:00 "
                f"({summary['peak_hour_label']})  "
                f"ratio={summary['peak_offpeak_ratio']:.2f}x"
            )
        if "total_burst_events" in summary:
            logger.info(
                f"  Burst events   : {summary['total_burst_events']} total  "
                f"| {summary['critical_bursts']} critical  "
                f"| Most bursty: {summary['most_bursty_region']}"
            )
        logger.info("=" * 60)

    def _save_csv(self, regional, cells, hourly,
                  dow, monthly, bursts, service):
        saves = {
            "st_regional_hotspots.csv":  regional,
            "st_cell_hotspots.csv":      cells,
            "st_hourly_patterns.csv":    hourly,
            "st_dow_patterns.csv":       dow,
            "st_monthly_trends.csv":     monthly,
            "st_anomaly_bursts.csv":     bursts,
            "st_service_by_region.csv":  service,
        }
        for fname, df in saves.items():
            if not df.empty:
                df.to_csv(REPORTS_DIR / fname, index=True)
        logger.info(f"  ST report tables saved → {REPORTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────

def _hour_label(h: int) -> str:
    if   0 <= h < 6:  return "Night (00-05)"
    elif 6 <= h < 9:  return "Early Morning (06-08)"
    elif 9 <= h < 12: return "Morning (09-11)"
    elif 12<= h < 14: return "Lunch (12-13)"
    elif 14<= h < 17: return "Afternoon (14-16)"
    elif 17<= h < 21: return "Evening (17-20)"
    else:             return "Late Evening (21-23)"