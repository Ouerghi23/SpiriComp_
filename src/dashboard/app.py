"""
Huawei NOC Intelligence Dashboard
===================================
Multi-page Streamlit application for telecom quality management.

Pages:
  1. 🏠 Overview         — KPI tiles, complaint trend, QoE heatmap
  2. 🗺️  Complaint Map    — Geographic hotspot visualisation
  3. 🚨 Anomaly Feed     — Live anomaly events with severity badges
  4. 📈 Forecasting      — 7-day complaint volume predictions
  5. 👥 User Segments    — Customer cluster profiles

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.data_loader import (
    load_all, KPI_META, qoe_color, delta_arrow,
    QOE_GREEN, QOE_YELLOW, REGIONS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Huawei NOC Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* KPI metric cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        margin: 4px 0;
    }
    .kpi-label  { color: #8892b0; font-size: 12px; font-weight: 600;
                  letter-spacing: 1px; text-transform: uppercase; }
    .kpi-value  { color: #e6f1ff; font-size: 28px; font-weight: 700;
                  margin: 6px 0; }
    .kpi-unit   { color: #8892b0; font-size: 13px; }
    .kpi-delta  { font-size: 12px; margin-top: 4px; }

    /* Severity badges */
    .badge-high   { background:#e74c3c; color:white; padding:3px 10px;
                    border-radius:12px; font-size:11px; font-weight:700; }
    .badge-medium { background:#f39c12; color:white; padding:3px 10px;
                    border-radius:12px; font-size:11px; font-weight:700; }
    .badge-low    { background:#27ae60; color:white; padding:3px 10px;
                    border-radius:12px; font-size:11px; font-weight:700; }

    /* Section headers */
    .section-header {
        color: #64ffda; font-size: 16px; font-weight: 700;
        border-bottom: 1px solid #2d3561;
        padding-bottom: 6px; margin: 20px 0 14px 0;
        letter-spacing: 0.5px;
    }
    /* Sidebar */
    .css-1d391kg { background-color: #0a0e1a; }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading NOC data...")
def get_data():
    return load_all()

data = get_data()
complaint_agg   = data["complaint_agg"]
kpi_agg         = data["kpi_agg"]
complaints_clean= data["complaints_clean"]
anomaly_results = data["anomaly_results"]
forecasts       = data["forecasts"]
kmeans_users    = data["kmeans_users"]
cluster_profiles= data["cluster_profiles"]
pred_scores     = data["prediction_scores"]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 NOC Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🗺️ Complaint Map", "🚨 Anomaly Feed",
         "📈 Forecasting", "👥 User Segments"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Global filters
    st.markdown("### Filters")
    all_regions = sorted(complaint_agg["region"].unique().tolist()) \
                  if not complaint_agg.empty else REGIONS
    selected_regions = st.multiselect(
        "Regions", all_regions, default=all_regions
    )

    if not complaint_agg.empty:
        date_min = complaint_agg["date"].min().date()
        date_max = complaint_agg["date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(date_min, date_max),
            min_value=date_min, max_value=date_max
        )
    else:
        date_range = (None, None)

    st.markdown("---")

    # Quick stats
    if not complaint_agg.empty:
        total_c = int(complaint_agg["total_complaints"].sum())
        total_a = int(anomaly_results["anomaly_flag"].sum()) \
                  if not anomaly_results.empty else 0
        st.metric("Total Complaints", f"{total_c:,}")
        st.metric("Anomalies Detected", f"{total_a:,}")

    st.markdown("---")
    st.caption("Huawei PFE · NOC Dashboard v1.0")


# ── Apply filters ─────────────────────────────────────────────────────────────
def apply_filters(df, date_col="date"):
    if df.empty:
        return df
    if selected_regions:
        df = df[df["region"].isin(selected_regions)]
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start = pd.Timestamp(date_range[0])
        end   = pd.Timestamp(date_range[1])
        df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    return df

ca_f  = apply_filters(complaint_agg)
ka_f  = apply_filters(kpi_agg)
an_f  = apply_filters(anomaly_results)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Network Operations Centre — Overview")

    if ca_f.empty or ka_f.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # ── KPI Metric Tiles ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Network KPIs — Current Period</div>',
                unsafe_allow_html=True)

    # Compare last 7d vs previous 7d
    ka_sorted  = ka_f.sort_values("date")
    last7      = ka_sorted.tail(7 * len(selected_regions))
    prev7      = ka_sorted.iloc[-(14 * len(selected_regions)):-(7 * len(selected_regions))]

    cols = st.columns(4)
    kpi_keys = list(KPI_META.keys())[:8]
    for i, kpi_key in enumerate(kpi_keys):
        if kpi_key not in ka_f.columns:
            continue
        meta    = KPI_META[kpi_key]
        current = last7[kpi_key].mean() if not last7.empty else 0
        previous= prev7[kpi_key].mean() if not prev7.empty else 0
        delta   = delta_arrow(current, previous, meta["good"])
        color   = qoe_color(current) if "qoe" in kpi_key else "#64ffda"

        delta_color = "#2ecc71" if "▲" in delta else "#e74c3c" if "▼" in delta else "#8892b0"
        # Flip for "low is good" KPIs
        if meta["good"] == "low" and "▼" in delta:
            delta_color = "#2ecc71"
        elif meta["good"] == "low" and "▲" in delta:
            delta_color = "#e74c3c"

        with cols[i % 4]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{meta['label']}</div>
                <div class="kpi-value" style="color:{color}">
                    {current:{meta['fmt']}}
                    <span class="kpi-unit">{meta['unit']}</span>
                </div>
                <div class="kpi-delta" style="color:{delta_color}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Complaint Volume Trend ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Complaint Volume Trend</div>',
                unsafe_allow_html=True)

    daily = (ca_f.groupby("date")["total_complaints"]
               .sum().reset_index())
    spikes = daily[daily["total_complaints"] >
                   daily["total_complaints"].mean() +
                   2 * daily["total_complaints"].std()]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily["date"], y=daily["total_complaints"],
        mode="lines", name="Daily Complaints",
        line=dict(color="#64ffda", width=2),
        fill="tozeroy", fillcolor="rgba(100,255,218,0.07)"
    ))
    fig_trend.add_trace(go.Scatter(
        x=spikes["date"], y=spikes["total_complaints"],
        mode="markers", name="Spike",
        marker=dict(color="#e74c3c", size=8, symbol="diamond")
    ))
    # 7-day rolling average
    daily["roll7"] = daily["total_complaints"].rolling(7, min_periods=1).mean()
    fig_trend.add_trace(go.Scatter(
        x=daily["date"], y=daily["roll7"],
        mode="lines", name="7-day MA",
        line=dict(color="#f39c12", width=1.5, dash="dot")
    ))
    fig_trend.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── QoE Heatmap + Category Breakdown ─────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">🌡️ QoE Score Heatmap — Region × Month</div>',
                    unsafe_allow_html=True)
        if "qoe_score_mean" in ka_f.columns:
            ka_f2 = ka_f.copy()
            ka_f2["month_label"] = ka_f2["date"].dt.strftime("%Y-%m")
            pivot = (ka_f2.groupby(["region", "month_label"])["qoe_score_mean"]
                          .mean().unstack(fill_value=np.nan))
            fig_heat = px.imshow(
                pivot, color_continuous_scale="RdYlGn",
                zmin=40, zmax=100,
                labels=dict(color="QoE Score"),
                aspect="auto",
            )
            fig_heat.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(thickness=12),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">🗂️ Complaints by Category</div>',
                    unsafe_allow_html=True)
        cat_cols = [c for c in ca_f.columns if c.startswith("cat_")]
        if cat_cols:
            cat_totals = ca_f[cat_cols].sum().sort_values(ascending=False)
            cat_names  = [c.replace("cat_", "").replace("_", " ").title()
                          for c in cat_totals.index]
            fig_pie = go.Figure(go.Pie(
                labels=cat_names,
                values=cat_totals.values,
                hole=0.55,
                textinfo="percent",
                hovertemplate="%{label}<br>%{value:,} complaints<extra></extra>",
            ))
            fig_pie.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=0, r=10, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Regional Complaint Bar ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📍 Complaints by Region</div>',
                unsafe_allow_html=True)
    region_totals = (ca_f.groupby("region")["total_complaints"]
                         .sum().sort_values(ascending=True).reset_index())
    fig_bar = px.bar(
        region_totals, x="total_complaints", y="region",
        orientation="h", color="total_complaints",
        color_continuous_scale="Teal",
        labels={"total_complaints": "Total Complaints", "region": ""},
    )
    fig_bar.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPLAINT MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Complaint Map":
    st.title("🗺️ Geographic Complaint Analysis")

    try:
        import folium
        from streamlit_folium import st_folium

        col1, col2 = st.columns([2, 1])

        with col1:
            map_metric = st.selectbox(
                "Map Metric",
                ["Total Complaints", "High Priority Complaints", "VIP Complaints",
                 "Anomaly Count"]
            )

        with col2:
            map_period = st.selectbox("Period", ["All Time", "Last 30 days", "Last 7 days"])

        # Filter by period
        ca_map = ca_f.copy()
        if map_period == "Last 30 days":
            cutoff = ca_map["date"].max() - pd.Timedelta(days=30)
            ca_map = ca_map[ca_map["date"] >= cutoff]
        elif map_period == "Last 7 days":
            cutoff = ca_map["date"].max() - pd.Timedelta(days=7)
            ca_map = ca_map[ca_map["date"] >= cutoff]

        # Region centroids
        region_coords = {
            "Tunis":     (36.818, 10.165), "Sfax":     (34.740, 10.760),
            "Sousse":    (35.825, 10.638), "Kairouan": (35.671, 10.100),
            "Bizerte":   (37.275,  9.873), "Gabes":    (33.881, 10.097),
            "Ariana":    (36.862, 10.193), "Gafsa":    (34.422,  8.784),
            "Monastir":  (35.777, 10.826), "Ben Arous":(36.753, 10.228),
        }

        # Build metric per region
        metric_col_map = {
            "Total Complaints":      "total_complaints",
            "High Priority Complaints": "high_priority_complaints",
            "VIP Complaints":        "vip_complaints",
        }
        if map_metric in metric_col_map:
            col_name = metric_col_map[map_metric]
            region_metric = (
                ca_map.groupby("region")[col_name].sum().reset_index()
                if col_name in ca_map.columns
                else ca_map.groupby("region")["total_complaints"].sum().reset_index()
            )
        else:
          col_name = "anomaly_count"    
          an_agg = (an_f.groupby("region")["anomaly_flag"].sum().reset_index()
          .rename(columns={"anomaly_flag": col_name}))
          region_metric = an_agg

        region_metric.columns = ["region", "value"]
        max_val = region_metric["value"].max() or 1

        # Build Folium map
        m = folium.Map(location=[35.5, 10.0], zoom_start=7,
                       tiles="CartoDB dark_matter")

        for _, row in region_metric.iterrows():
            if row["region"] not in region_coords:
                continue
            lat, lon = region_coords[row["region"]]
            val      = row["value"]
            radius   = 15 + (val / max_val) * 35

            # Colour by QoE if available
            qoe_val = 70.0
            if "qoe_score_mean" in ka_f.columns and not ka_f.empty:
                reg_qoe = ka_f[ka_f["region"] == row["region"]]["qoe_score_mean"]
                if not reg_qoe.empty:
                    qoe_val = reg_qoe.mean()
            color = qoe_color(qoe_val)

            folium.CircleMarker(
                location=[lat, lon], radius=radius,
                color=color, fill=True, fill_color=color, fill_opacity=0.6,
                tooltip=folium.Tooltip(
                    f"<b>{row['region']}</b><br>"
                    f"{map_metric}: {int(val):,}<br>"
                    f"QoE: {qoe_val:.1f}"
                )
            ).add_to(m)

            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;color:white;'
                         f'font-weight:bold;text-align:center;">'
                         f'{row["region"]}<br>{int(val):,}</div>',
                    icon_size=(80, 30), icon_anchor=(40, 15)
                )
            ).add_to(m)

        st_folium(m, width="100%", height=500)

        # Region table below map
        st.markdown('<div class="section-header">📋 Regional Summary Table</div>',
                    unsafe_allow_html=True)
        if not ca_map.empty:
            summary = ca_map.groupby("region").agg(
                total_complaints=("total_complaints", "sum"),
                spike_days=("complaint_spike_flag", "sum"),
                high_priority=("high_priority_complaints", "sum"),
            ).reset_index().sort_values("total_complaints", ascending=False)

            if "qoe_score_mean" in ka_f.columns:
                qoe_summary = (ka_f.groupby("region")["qoe_score_mean"]
                                   .mean().reset_index()
                                   .rename(columns={"qoe_score_mean": "avg_qoe"}))
                summary = summary.merge(qoe_summary, on="region", how="left")
                summary["avg_qoe"] = summary["avg_qoe"].round(1)

            st.dataframe(summary, use_container_width=True, hide_index=True)

    except ImportError:
        st.warning("Install `streamlit-folium` and `folium` to enable the map: "
                   "`pip install folium streamlit-folium`")
        st.info("Showing tabular summary instead.")
        if not ca_f.empty:
            st.dataframe(
                ca_f.groupby("region")["total_complaints"].sum()
                    .sort_values(ascending=False).reset_index(),
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY FEED
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Feed":
    st.title("🚨 Anomaly Detection Feed")

    if an_f.empty:
        st.warning("No anomaly data available. Run Phase 3 models first.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_anomalies  = int(an_f["anomaly_flag"].sum())
    consensus        = int(an_f["anomaly_consensus"].sum()) if "anomaly_consensus" in an_f.columns else 0
    high_sev         = int((an_f["if_severity"] == "High").sum()) if "if_severity" in an_f.columns else 0
    rate             = an_f["anomaly_flag"].mean() * 100

    c1.metric("Total Anomalies",   f"{total_anomalies:,}")
    c2.metric("High Severity",     f"{high_sev:,}", delta=None)
    c3.metric("Consensus (Both)",  f"{consensus:,}")
    c4.metric("Anomaly Rate",      f"{rate:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # ── Anomaly timeline ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Anomaly Score Timeline</div>',
                    unsafe_allow_html=True)
        sel_region = st.selectbox("Select Region", sorted(an_f["region"].unique()))
        region_an  = an_f[an_f["region"] == sel_region].sort_values("date")

        fig_al = go.Figure()
        fig_al.add_trace(go.Scatter(
            x=region_an["date"], y=region_an["combined_score"],
            mode="lines", fill="tozeroy",
            line=dict(color="#9b59b6", width=1.5),
            fillcolor="rgba(155,89,182,0.1)",
            name="Combined Score"
        ))
        # Highlight anomalies
        anom_pts = region_an[region_an["anomaly_flag"] == 1]
        fig_al.add_trace(go.Scatter(
            x=anom_pts["date"], y=anom_pts["combined_score"],
            mode="markers", name="Anomaly",
            marker=dict(color="#e74c3c", size=8, symbol="x")
        ))
        fig_al.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_al, use_container_width=True)

        # ── Anomaly events table ──────────────────────────────────────────────
        st.markdown('<div class="section-header">🗒️ Recent Anomaly Events</div>',
                    unsafe_allow_html=True)

        severity_filter = st.multiselect(
            "Filter by Severity",
            ["High", "Medium", "Low"],
            default=["High", "Medium"]
        )

        events = an_f[an_f["anomaly_flag"] == 1].copy()
        if "if_severity" in events.columns and severity_filter:
            events = events[events["if_severity"].isin(severity_filter)]
        events = events.sort_values("date", ascending=False).head(50)

        if not events.empty:
            display_cols = ["date", "region", "if_severity",
                            "combined_score", "top_anomaly_driver",
                            "anomaly_consensus"]
            display_cols = [c for c in display_cols if c in events.columns]
            events_display = events[display_cols].copy()
            if "combined_score" in events_display.columns:
                events_display["combined_score"] = events_display["combined_score"].round(3)
            if "date" in events_display.columns:
                events_display["date"] = events_display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(events_display, use_container_width=True, hide_index=True)
        else:
            st.info("No anomaly events match the selected severity filter.")

    with col_right:
        # ── Anomalies by region donut ─────────────────────────────────────────
        st.markdown('<div class="section-header">🌍 By Region</div>',
                    unsafe_allow_html=True)
        by_region = (an_f[an_f["anomaly_flag"] == 1]
                     .groupby("region")["anomaly_flag"].sum()
                     .sort_values(ascending=False).reset_index())
        fig_reg = go.Figure(go.Pie(
            labels=by_region["region"], values=by_region["anomaly_flag"],
            hole=0.55, textinfo="percent+label",
        ))
        fig_reg.update_layout(
            template="plotly_dark", height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # ── Top anomaly drivers ───────────────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Top KPI Drivers</div>',
                    unsafe_allow_html=True)
        if "top_anomaly_driver" in an_f.columns:
            drivers = (an_f[an_f["anomaly_flag"] == 1]["top_anomaly_driver"]
                       .value_counts().head(8).reset_index())
            drivers.columns = ["KPI", "Count"]
            drivers["KPI"] = drivers["KPI"].str.replace("_", " ").str.title()
            fig_drv = px.bar(
                drivers, x="Count", y="KPI", orientation="h",
                color="Count", color_continuous_scale="Reds",
            )
            fig_drv.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_drv, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecasting":
    st.title("📈 Complaint Volume Forecasting")

    if forecasts.empty:
        st.warning("No forecast data available. Run Phase 3 spike predictor first.")
        st.stop()

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Model Performance by Region</div>',
                unsafe_allow_html=True)

    if not pred_scores.empty:
        pivot_mae = pred_scores.pivot(
            index="region", columns="model", values="mae"
        ).round(2)
        # Highlight minimum per row (best model)
        st.dataframe(
            pivot_mae.style.highlight_min(axis=1, color="#1a472a"),
            use_container_width=True
        )

    # ── Forecast charts ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 7-Day Forecast by Region</div>',
                unsafe_allow_html=True)

    fc_regions = sorted(forecasts["region"].unique())
    sel_regions_fc = st.multiselect(
        "Select Regions to Display",
        fc_regions, default=fc_regions[:4]
    )

    if sel_regions_fc:
        n_plots = len(sel_regions_fc)
        cols_per_row = 2
        rows = (n_plots + cols_per_row - 1) // cols_per_row

        for row_i in range(rows):
            cols = st.columns(cols_per_row)
            for col_i in range(cols_per_row):
                idx = row_i * cols_per_row + col_i
                if idx >= n_plots:
                    break
                region = sel_regions_fc[idx]

                # Historical (last 45 days)
                hist = (complaint_agg[complaint_agg["region"] == region]
                        .sort_values("date").tail(45))
                fc   = forecasts[forecasts["region"] == region]
                model_used = fc["model_used"].iloc[0] if not fc.empty else "N/A"

                with cols[col_i]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist["date"], y=hist["total_complaints"],
                        mode="lines+markers",
                        marker=dict(size=3),
                        line=dict(color="#64ffda", width=1.5),
                        name="Historical",
                    ))
                    if not fc.empty:
                        fig.add_trace(go.Scatter(
                            x=fc["date"], y=fc["forecast"],
                            mode="lines+markers",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                            marker=dict(size=6, symbol="diamond"),
                            name=f"Forecast ({model_used.upper()})",
                        ))
                        # Shade forecast area
                        fig.add_vrect(
                            x0=fc["date"].min(), x1=fc["date"].max(),
                            fillcolor="rgba(231,76,60,0.07)",
                            layer="below", line_width=0,
                        )
                    fig.update_layout(
                        title=dict(text=region, font=dict(size=13)),
                        template="plotly_dark", height=240,
                        margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(font=dict(size=9), orientation="h"),
                        showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast summary table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Forecast Summary — Next 7 Days</div>',
                unsafe_allow_html=True)
    if not forecasts.empty:
        fc_summary = (
            forecasts[forecasts["region"].isin(selected_regions)]
            .groupby("region")
            .agg(
                total_forecast=("forecast", "sum"),
                avg_daily=("forecast", "mean"),
                peak_day=("forecast", "max"),
                model=("model_used", "first"),
            )
            .reset_index()
            .sort_values("total_forecast", ascending=False)
        )
        fc_summary = fc_summary.round(1)
        fc_summary.columns = ["Region", "Total (7d)", "Avg Daily",
                               "Peak Day", "Model Used"]
        st.dataframe(fc_summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — USER SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 User Segments":
    st.title("👥 Customer Experience Segmentation")

    if kmeans_users.empty:
        st.warning("No clustering data available. Run Phase 3 clustering first.")
        st.stop()

    optimal_k = int(kmeans_users["kmeans_cluster"].nunique())

    # ── Cluster summary cards ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Cluster Profiles</div>',
                unsafe_allow_html=True)

    CLUSTER_COLORS = ["#64ffda", "#f39c12", "#e74c3c", "#9b59b6",
                      "#3498db", "#2ecc71", "#e67e22", "#1abc9c"]

    if not cluster_profiles.empty:
        cols = st.columns(min(optimal_k, 4))
        for i, (_, row) in enumerate(cluster_profiles.iterrows()):
            cluster_id = int(row["kmeans_cluster"])
            n_users    = int(row["n_users"])
            pct        = row["pct"]
            color      = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

            qoe_col = "qoe_score_mean" if "qoe_score_mean" in row.index else None
            qoe_val = f"{row[qoe_col]:.1f}" if qoe_col else "N/A"

            with cols[i % 4]:
                st.markdown(f"""
                <div class="kpi-card" style="border-color:{color}40;">
                    <div class="kpi-label" style="color:{color}">
                        CLUSTER {cluster_id}
                    </div>
                    <div class="kpi-value" style="color:{color};font-size:22px;">
                        {n_users:,} users
                    </div>
                    <div class="kpi-unit">{pct}% of base</div>
                    <div class="kpi-delta" style="color:#8892b0;margin-top:6px;">
                        QoE: {qoe_val}
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_scatter, col_kpi = st.columns([3, 2])

    with col_scatter:
        # ── PCA scatter ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔵 Cluster Scatter (PCA)</div>',
                    unsafe_allow_html=True)
        if "pca_x" in kmeans_users.columns and "pca_y" in kmeans_users.columns:
            # Sample for performance
            sample = kmeans_users.sample(min(3000, len(kmeans_users)),
                                         random_state=42)
            fig_scatter = px.scatter(
                sample, x="pca_x", y="pca_y",
                color=sample["kmeans_cluster"].astype(str),
                color_discrete_sequence=CLUSTER_COLORS,
                labels={"pca_x": "PC1", "pca_y": "PC2",
                        "color": "Cluster"},
                opacity=0.5,
                hover_data={"pca_x": False, "pca_y": False},
            )
            fig_scatter.update_traces(marker=dict(size=3))
            fig_scatter.update_layout(
                template="plotly_dark", height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(title="Cluster"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with col_kpi:
        # ── KPI radar / bar per cluster ───────────────────────────────────────
        st.markdown('<div class="section-header">📡 KPI Profile per Cluster</div>',
                    unsafe_allow_html=True)
        if not cluster_profiles.empty:
            kpi_mean_cols = [c for c in cluster_profiles.columns
                             if c.endswith("_mean") and "n_users" not in c][:6]
            if kpi_mean_cols:
                radar_df = cluster_profiles[["kmeans_cluster"] + kpi_mean_cols].copy()
                # Normalise 0-1 per KPI
                for c in kpi_mean_cols:
                    mn, mx = radar_df[c].min(), radar_df[c].max()
                    radar_df[c] = (radar_df[c] - mn) / (mx - mn + 1e-9)

                fig_radar = go.Figure()
                theta = [c.replace("_mean", "").replace("_", " ").title()
                         for c in kpi_mean_cols]
                for i, (_, row) in enumerate(radar_df.iterrows()):
                    vals = [row[c] for c in kpi_mean_cols]
                    vals += [vals[0]]  # close polygon
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=theta + [theta[0]],
                        name=f"Cluster {int(row['kmeans_cluster'])}",
                        line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                                  width=2),
                        fill="toself",
                        fillcolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                        opacity=0.15,
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1],
                                        gridcolor="#2d3561"),
                        angularaxis=dict(gridcolor="#2d3561"),
                    ),
                    template="plotly_dark", height=360,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    # ── Cluster region distribution ───────────────────────────────────────────
    st.markdown('<div class="section-header">🌍 Cluster Distribution by Region</div>',
                unsafe_allow_html=True)
    if "region" in kmeans_users.columns:
        cross = pd.crosstab(
            kmeans_users["region"],
            kmeans_users["kmeans_cluster"],
            normalize="index"
        ).mul(100).round(1)
        cross.columns = [f"Cluster {c}" for c in cross.columns]

        fig_cross = px.bar(
            cross.reset_index(), x="region",
            y=cross.columns.tolist(),
            barmode="stack",
            color_discrete_sequence=CLUSTER_COLORS,
            labels={"value": "% of Users", "region": "Region",
                    "variable": "Cluster"},
        )
        fig_cross.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cross, use_container_width=True)