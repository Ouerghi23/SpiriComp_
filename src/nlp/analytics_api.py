"""
src/api/analytics_api.py
=========================
SpiriComp — FastAPI analytics backend (main entry point).

Run from project root:
    uvicorn src.api.analytics_api:app --reload --port 8000
                ^^^^^^^^^^^^^^^^^^^^^^^^^
                NOTE: src.API not src.NLP  ← this was the root cause of all 404s
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from fastapi import FastAPI, APIRouter, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("analytics_api")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA   = Path("data/processed")
MODELS = Path("models")

PATHS: dict[str, Path] = {
    "complaints_clean":    DATA   / "complaints_clean.parquet",
    "complaint_daily_agg": DATA   / "complaint_daily_agg.parquet",
    "kpi_daily_agg":       DATA   / "kpi_daily_agg.parquet",
    "feature_matrix":      DATA   / "feature_matrix.parquet",
    "anomaly_results":     MODELS / "anomaly/anomaly_results.parquet",
    "forecasts":           MODELS / "prediction/forecasts.parquet",
    "kmeans_users":        MODELS / "clustering/kmeans_users.parquet",
    "dbscan_users":        MODELS / "clustering/dbscan_users.parquet",
}

TN_COORDS: dict[str, tuple[float, float]] = {
    "Tunis":      (36.8065, 10.1815), "Sfax":        (34.7406, 10.7603),
    "Sousse":     (35.8256, 10.6411), "Kairouan":    (35.6781, 10.0963),
    "Bizerte":    (37.2746,  9.8739), "Gabès":       (33.8815, 10.0982),
    "Ariana":     (36.8625, 10.1956), "Gafsa":       (34.4250,  8.7842),
    "Monastir":   (35.7780, 10.8262), "Mahdia":      (35.5047, 11.0622),
    "Médenine":   (33.3548, 10.5055), "Nabeul":      (36.4561, 10.7376),
    "Béja":       (36.7256,  9.1817), "Jendouba":    (36.5028,  8.7803),
    "Le Kef":     (36.1675,  8.7050), "Siliana":     (36.0844,  9.3708),
    "Kasserine":  (35.1675,  8.8364), "Sidi Bouzid": (35.0381,  9.4858),
    "Tozeur":     (33.9197,  8.1336), "Tataouine":   (32.9297, 10.4517),
    "Kébili":     (33.7050,  8.9692), "Manouba":     (36.8104, 10.0863),
    "Ben Arous":  (36.7531, 10.2189), "Zaghouan":    (36.4022, 10.1429),
    "La Marsa":   (36.8765, 10.3253), "Carthage":    (36.8527, 10.3300),
    "Hammamet":   (36.4000, 10.6167), "Djerba":      (33.7833, 10.8833),
    "Zarzis":     (33.5000, 11.1167), "El Kram":     (36.8333, 10.3167),
}

CACHE_TTL = 120
_cache: dict[str, tuple[pd.DataFrame, float]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load(key: str, refresh: bool = False) -> pd.DataFrame:
    now = time.monotonic()
    if not refresh and key in _cache:
        df, ts = _cache[key]
        if now - ts < CACHE_TTL:
            return df
    p = PATHS.get(key)
    if p and p.exists():
        df = pd.read_parquet(p)
        logger.info("loaded %s (%d rows)", key, len(df))
    else:
        logger.warning("data file not found: %s", p)
        df = pd.DataFrame()
    _cache[key] = (df, now)
    return df


_DATE_COL_CANDIDATES = [
    "date", "Date", "timestamp", "Timestamp",
    "datetime", "DateTime", "date_time", "period",
    "day", "obs_date", "report_date",
]


def _find_date_col(df: pd.DataFrame) -> str | None:
    for candidate in _DATE_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "datetime" in str(df[col].dtype).lower():
            return col
    for col in df.columns:
        if any(kw in col.lower() for kw in ("date", "time", "day", "period")):
            return col
    return None


def safe_dict(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")

    def _cast(v):
        if isinstance(v, np.integer):  return int(v)
        if isinstance(v, np.floating): return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):    return bool(v)
        return v

    return [{k: _cast(v) for k, v in row.items()} for row in records]


# ── Analytics Router ──────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/overview")
async def overview(refresh: bool = Query(False)):
    cc  = load("complaints_clean", refresh)
    kpi = load("kpi_daily_agg",    refresh)
    out: dict = {}
    if not cc.empty:
        out["total_complaints"] = int(len(cc))
        out["unique_msisdns"]   = int(cc["msisdn"].nunique()) if "msisdn" in cc.columns else 0
        out["unique_cities"]    = int(cc["city"].nunique())   if "city"   in cc.columns else 0
        out["unique_regions"]   = int(cc["region"].nunique()) if "region" in cc.columns else 0
        date_col_cc = _find_date_col(cc)
        if date_col_cc:
            ts = pd.to_datetime(cc[date_col_cc], errors="coerce")
            out["date_min"] = str(ts.min())[:10]
            out["date_max"] = str(ts.max())[:10]
        for col, key in [("region", "top_region"), ("complaint_subcategory", "top_subcategory")]:
            if col in cc.columns:
                out[key] = cc[col].value_counts().index[0]
        for col, key in [("service_type", "by_service"), ("complaint_typology", "by_typology"),
                          ("customer_segment", "by_segment"), ("priority", "by_priority")]:
            if col in cc.columns:
                out[key] = cc[col].value_counts().to_dict()
    if not kpi.empty:
        date_col_kpi = _find_date_col(kpi)
        if date_col_kpi:
            kpi = kpi.copy()
            kpi[date_col_kpi] = pd.to_datetime(kpi[date_col_kpi], errors="coerce")
            last   = kpi[date_col_kpi].max()
            recent = kpi[kpi[date_col_kpi] >= last - pd.Timedelta(days=30)]
            prev   = kpi[(kpi[date_col_kpi] >= last - pd.Timedelta(days=60)) &
                         (kpi[date_col_kpi] <  last - pd.Timedelta(days=30))]
            kpi_cols = ["dl_throughput_mbps_mean", "latency_ms_mean", "packet_loss_pct_mean",
                        "call_drop_rate_mean", "data_qoe_score_mean", "voice_qoe_score_mean",
                        "data_session_success_rate_mean", "voice_quality_score_mos_mean"]
            avgs = {}
            for c in kpi_cols:
                if c not in kpi.columns: continue
                cur = float(recent[c].mean()) if not recent.empty else 0.0
                prv = float(prev[c].mean())   if not prev.empty  else cur
                d   = (cur - prv) / prv * 100 if prv and not np.isnan(prv) and not np.isinf(prv) else 0.0
                avgs[c] = {"value": round(cur, 2), "delta": round(d, 2),
                           "delta_str": f"{'+' if d >= 0 else ''}{d:.1f}%"}
            out["kpi_averages"] = avgs
    return out


@router.get("/complaints/trend")
async def complaints_trend(refresh: bool = Query(False)):
    agg = load("complaint_daily_agg", refresh)
    if agg.empty:
        return {"trend": []}
    date_col = _find_date_col(agg)
    if not date_col or "total_complaints" not in agg.columns:
        return {"trend": [], "error": f"Required columns missing. Found: {list(agg.columns)}"}
    agg = agg.copy()
    agg[date_col] = pd.to_datetime(agg[date_col], errors="coerce")
    daily = agg.groupby(date_col)["total_complaints"].sum().reset_index().sort_values(date_col)
    daily[date_col]   = daily[date_col].dt.strftime("%Y-%m-%d")
    daily["roll7"]    = daily["total_complaints"].rolling(7, min_periods=1).mean().round(2)
    mu, sigma         = daily["total_complaints"].mean(), daily["total_complaints"].std()
    daily["is_spike"] = (daily["total_complaints"] > mu + 2 * sigma).astype(int)
    daily = daily.rename(columns={date_col: "date"})
    return {"trend": safe_dict(daily)}


@router.get("/complaints/by-region")
async def complaints_by_region(refresh: bool = Query(False)):
    agg = load("complaint_daily_agg", refresh)
    kpi = load("kpi_daily_agg",       refresh)
    if agg.empty or "region" not in agg.columns:
        return {"regions": []}
    totals = (agg.groupby("region")["total_complaints"].sum()
              .reset_index().sort_values("total_complaints", ascending=False))
    qoe_col = next((c for c in ["qoe_score_mean", "data_qoe_score_mean"] if c in kpi.columns), None)
    if qoe_col and not kpi.empty and "region" in kpi.columns:
        q = kpi.groupby("region")[qoe_col].mean().reset_index()
        q.columns = ["region", "qoe"]
        totals = totals.merge(q, on="region", how="left")
        totals["qoe"] = totals["qoe"].round(1)
    return {"regions": safe_dict(totals)}


@router.get("/complaints/by-city")
async def complaints_by_city(refresh: bool = Query(False)):
    cc  = load("complaints_clean", refresh)
    kpi = load("kpi_daily_agg",    refresh)
    if cc.empty:
        return {"cities": []}
    if "city" not in cc.columns:
        return {"cities": [], "error": "No 'city' column in complaints_clean"}
    gcols   = ["city"] + (["region"] if "region" in cc.columns else [])
    grouped = cc.groupby(gcols).size().reset_index(name="complaints").sort_values("complaints", ascending=False)
    qoe_col = next((c for c in ["qoe_score_mean", "data_qoe_score_mean", "voice_qoe_score_mean"]
                    if c in kpi.columns), None)
    if qoe_col and not kpi.empty:
        by = "city" if "city" in kpi.columns else ("region" if "region" in kpi.columns else None)
        if by and by in grouped.columns:
            q = kpi.groupby(by)[qoe_col].mean().reset_index()
            q.columns = [by, "qoe"]
            grouped = grouped.merge(q, on=by, how="left")
    if "qoe" not in grouped.columns:
        grouped["qoe"] = 50.0
    grouped["qoe"] = grouped["qoe"].fillna(50.0).clip(0, 100).round(1)
    if "service_type" in cc.columns:
        svc = cc.groupby(["city", "service_type"]).size().unstack(fill_value=0)
        for col in svc.columns:
            k = col.lower().replace(" ", "_").replace("-", "_")
            k = "4g" if ("4g" in k or "data" in k) else "voice" if "voice" in k else "sms" if "sms" in k else k
            grouped[k] = grouped["city"].map(svc[col].to_dict()).fillna(0).astype(int)
    grouped["lat"] = grouped["city"].map(lambda x: TN_COORDS.get(x, (None, None))[0])
    grouped["lng"] = grouped["city"].map(lambda x: TN_COORDS.get(x, (None, None))[1])
    grouped = grouped.dropna(subset=["lat", "lng"])
    return {"cities": safe_dict(grouped)}


@router.get("/kpi/tiles")
async def kpi_tiles(refresh: bool = Query(False)):
    kpi = load("kpi_daily_agg", refresh)
    if kpi.empty:
        return {"tiles": []}
    date_col = _find_date_col(kpi)
    if not date_col:
        return {"tiles": [], "error": f"No date column found. Columns: {list(kpi.columns)}"}
    META = [
        {"key": "dl_throughput_mbps_mean",        "label": "DL Throughput",   "unit": "Mbps", "good": "high"},
        {"key": "latency_ms_mean",                 "label": "Latency",         "unit": "ms",   "good": "low"},
        {"key": "packet_loss_pct_mean",            "label": "Packet Loss",     "unit": "%",    "good": "low"},
        {"key": "call_drop_rate_mean",             "label": "Call Drop Rate",  "unit": "%",    "good": "low"},
        {"key": "data_qoe_score_mean",             "label": "Data QoE",        "unit": "/100", "good": "high"},
        {"key": "voice_qoe_score_mean",            "label": "Voice QoE",       "unit": "/100", "good": "high"},
        {"key": "data_session_success_rate_mean",  "label": "Session Success", "unit": "%",    "good": "high"},
        {"key": "voice_quality_score_mos_mean",    "label": "MOS Score",       "unit": "/5",   "good": "high"},
    ]
    kpi = kpi.copy()
    kpi[date_col] = pd.to_datetime(kpi[date_col], errors="coerce")
    kpi = kpi.dropna(subset=[date_col])
    last  = kpi[date_col].max()
    last7 = kpi[kpi[date_col] >= last - pd.Timedelta(days=7)]
    prev7 = kpi[(kpi[date_col] >= last - pd.Timedelta(days=14)) &
                (kpi[date_col] <  last - pd.Timedelta(days=7))]
    tiles = []
    for m in META:
        if m["key"] not in kpi.columns: continue
        cur = float(last7[m["key"]].mean()) if not last7.empty else 0.0
        prv = float(prev7[m["key"]].mean()) if not prev7.empty else cur
        if any(map(lambda v: np.isnan(v) or np.isinf(v), [cur, prv])):
            cur, prv, delta = 0.0, 0.0, 0.0
        elif prv != 0:
            delta = (cur - prv) / prv * 100
        else:
            delta = 0.0
        tiles.append({"label": m["label"], "value": round(cur, 2), "unit": m["unit"],
                      "delta": round(delta, 2),
                      "good": (delta >= 0) if m["good"] == "high" else (delta <= 0)})
    return {"tiles": tiles}


@router.get("/kpi/heatmap")
async def kpi_heatmap(refresh: bool = Query(False)):
    kpi = load("kpi_daily_agg", refresh)
    if kpi.empty:
        return {"series": []}
    qoe_col = next((c for c in ["qoe_score_mean", "data_qoe_score_mean"] if c in kpi.columns), None)
    if not qoe_col:
        return {"series": [], "error": f"No QoE column. Columns: {list(kpi.columns)}"}
    if "region" not in kpi.columns:
        return {"series": [], "error": "'region' column missing"}
    date_col = _find_date_col(kpi)
    if not date_col:
        return {"series": [], "error": f"No date column. Columns: {list(kpi.columns)}"}
    kpi = kpi.copy()
    kpi[date_col] = pd.to_datetime(kpi[date_col], errors="coerce")
    kpi = kpi.dropna(subset=[date_col])
    kpi["month"] = kpi[date_col].dt.strftime("%b %Y")
    pivot = kpi.groupby(["region", "month"])[qoe_col].mean().reset_index()
    pivot.columns = ["region", "month", "qoe"]
    pivot["qoe"] = pivot["qoe"].round(1).replace([np.inf, -np.inf], np.nan)
    months = sorted(pivot["month"].unique().tolist())
    series = []
    for region in pivot["region"].unique():
        rd   = pivot[pivot["region"] == region]
        data = []
        for m in months:
            row = rd[rd["month"] == m]
            val = float(row["qoe"].values[0]) if not row.empty and pd.notna(row["qoe"].values[0]) else None
            data.append({"x": m, "y": val})
        series.append({"name": region.replace(" Gouvernorat", ""), "data": data})
    return {"series": series, "months": months}


@router.get("/anomalies/summary")
async def anomalies_summary(refresh: bool = Query(False)):
    an = load("anomaly_results", refresh)
    if an.empty:
        return {"summary": {}}
    total     = int(an["anomaly_flag"].sum())      if "anomaly_flag"      in an.columns else 0
    consensus = int(an["anomaly_consensus"].sum()) if "anomaly_consensus" in an.columns else 0
    if_count  = int(an["if_anomaly"].sum())        if "if_anomaly"        in an.columns else 0
    stat_cnt  = int(an["stat_anomaly"].sum())      if "stat_anomaly"      in an.columns else 0
    rate      = round(an["anomaly_flag"].mean() * 100, 1) if "anomaly_flag" in an.columns else 0
    top_regions: list = []
    if "region" in an.columns and "anomaly_flag" in an.columns:
        top_regions = (an[an["anomaly_flag"] == 1].groupby("region")["anomaly_flag"].sum()
                       .sort_values(ascending=False).head(5).reset_index()
                       .rename(columns={"anomaly_flag": "count"}).to_dict(orient="records"))
    consensus_events: list = []
    if "anomaly_consensus" in an.columns:
        cols = [c for c in ["region", "date", "combined_score", "top_anomaly_driver", "if_severity"]
                if c in an.columns]
        consensus_events = safe_dict(an[an["anomaly_consensus"] == 1][cols]
                                     .sort_values("combined_score", ascending=False).head(14))
    return {"summary": {"total": total, "if_count": if_count, "stat_count": stat_cnt,
                         "consensus": consensus, "rate_pct": rate,
                         "top_regions": top_regions, "consensus_events": consensus_events}}


@router.get("/anomalies/timeline")
async def anomalies_timeline(region: str | None = None, refresh: bool = Query(False)):
    an = load("anomaly_results", refresh)
    if an.empty:
        return {"timeline": []}
    if region and "region" in an.columns:
        an = an[an["region"] == region]
    date_col = _find_date_col(an)
    cols = [c for c in ["combined_score", "anomaly_flag", "if_severity", "top_anomaly_driver"]
            if c in an.columns]
    if date_col:
        cols = [date_col] + cols
    df = an[cols].sort_values(date_col) if date_col else an[cols]
    if "combined_score" in df.columns:
        df = df.copy()
        df["combined_score"] = df["combined_score"].round(4)
    return {"timeline": safe_dict(df)}


@router.get("/anomalies/regions")
async def anomaly_regions(refresh: bool = Query(False)):
    an = load("anomaly_results", refresh)
    if an.empty or "region" not in an.columns:
        return {"regions": []}
    return {"regions": sorted(an["region"].unique().tolist())}


@router.get("/forecasts")
async def get_forecasts(refresh: bool = Query(False)):
    fc = load("forecasts", refresh)
    if fc.empty:
        return {"forecasts": [], "regions": []}
    regions = sorted(fc["region"].unique().tolist()) if "region" in fc.columns else []
    return {"forecasts": safe_dict(fc), "regions": regions}


@router.get("/forecasts/scores")
async def forecast_scores(refresh: bool = Query(False)):
    score_path = Path("models/prediction/prediction_scores.parquet")
    if score_path.exists() and not refresh:
        return {"scores": safe_dict(pd.read_parquet(score_path))}
    scores_pkl = Path("models/prediction/scores.pkl")
    models_pkl = Path("models/prediction/all_models.pkl")
    if not scores_pkl.exists():
        return {"scores": [], "message": "Run Notebook 05 to generate scores.pkl"}
    try:
        import joblib
        scores_dict = joblib.load(str(scores_pkl))
        best_models: dict[str, str] = {}
        if models_pkl.exists():
            md = joblib.load(str(models_pkl))
            best_models = {r: v.get("best", "") for r, v in md.items()}
        rows = []
        for region, region_scores in scores_dict.items():
            best = best_models.get(region, "")
            for model, metrics in region_scores.items():
                rows.append({"region": region, "model": model, "is_best": model == best,
                             "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
                             "mape": metrics.get("mape")})
        if not rows:
            return {"scores": [], "message": "scores.pkl is empty"}
        df = pd.DataFrame(rows)
        try:
            score_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(score_path, index=False)
        except Exception as exc:
            logger.warning("could not write prediction_scores.parquet: %s", exc)
        return {"scores": safe_dict(df)}
    except Exception as exc:
        logger.exception("forecast_scores error")
        return {"scores": [], "message": f"Error: {exc}"}


@router.get("/forecasts/history")
async def forecast_history(region: str | None = None, refresh: bool = Query(False)):
    agg = load("complaint_daily_agg", refresh)
    if agg.empty:
        return {"history": []}
    if region and "region" in agg.columns:
        agg = agg[agg["region"] == region]
    date_col  = _find_date_col(agg)
    base_cols = [c for c in ["region", "total_complaints"] if c in agg.columns]
    cols      = ([date_col] + base_cols) if date_col else base_cols
    n_regions = max(agg["region"].nunique(), 1) if "region" in agg.columns else 1
    df = agg[cols].sort_values(date_col) if date_col else agg[cols]
    return {"history": safe_dict(df.tail(45 * n_regions))}


@router.get("/segments/profiles")
async def segment_profiles(refresh: bool = Query(False)):
    km = load("kmeans_users", refresh)
    if km.empty or "kmeans_cluster" not in km.columns:
        return {"profiles": [], "scatter": [], "n_clusters": 0}
    num_cols = [c for c in km.select_dtypes(include=[np.number]).columns
                if c not in ("kmeans_cluster", "pca_x", "pca_y", "id")]
    profiles = []
    for cid in sorted(km["kmeans_cluster"].unique()):
        cdf = km[km["kmeans_cluster"] == cid]
        p = {"cluster_id": int(cid), "n_users": int(len(cdf)),
             "pct": round(len(cdf) / len(km) * 100, 1)}
        for col in num_cols[:10]:
            p[col] = round(float(cdf[col].mean()), 3)
        profiles.append(p)
    pca_cols = [c for c in ["pca_x", "pca_y", "kmeans_cluster"] if c in km.columns]
    scatter  = km[pca_cols].sample(min(2000, len(km)), random_state=42) if pca_cols else pd.DataFrame()
    pca_var = silhouette = dbi = db_clusters = db_noise = None
    pca_pkl = Path("models/clustering/pca.pkl")
    if pca_pkl.exists():
        try:
            import joblib
            pca_var = round(float(joblib.load(str(pca_pkl)).explained_variance_ratio_.sum() * 100), 1)
        except Exception as exc:
            logger.warning("pca.pkl: %s", exc)
    feat_cols = [c for c in km.select_dtypes(include=[np.number]).columns
                 if c not in ("kmeans_cluster", "pca_x", "pca_y")]
    if feat_cols and km["kmeans_cluster"].nunique() > 1:
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            X, y = km[feat_cols].fillna(0).values, km["kmeans_cluster"].values
            if len(X) > 5000:
                idx = np.random.default_rng(42).choice(len(X), 5000, replace=False)
                X, y = X[idx], y[idx]
            silhouette = round(float(silhouette_score(X, y)), 3)
            dbi        = round(float(davies_bouldin_score(X, y)), 3)
        except Exception as exc:
            logger.warning("silhouette/DBI: %s", exc)
    db_path = Path("models/clustering/dbscan_users.parquet")
    if db_path.exists():
        try:
            db_labels   = pd.read_parquet(str(db_path))["dbscan_cluster"].values
            db_clusters = int(len(set(db_labels)) - (1 if -1 in db_labels else 0))
            db_noise    = int((db_labels == -1).sum())
        except Exception as exc:
            logger.warning("dbscan: %s", exc)
    return {"profiles": profiles, "scatter": safe_dict(scatter), "kpi_columns": num_cols[:6],
            "n_clusters": len(profiles), "silhouette_score": silhouette, "davies_bouldin": dbi,
            "pca_variance_pct": pca_var, "dbscan_clusters": db_clusters, "dbscan_noise": db_noise}


@router.get("/segments/region-distribution")
async def segment_region_distribution(refresh: bool = Query(False)):
    km = load("kmeans_users", refresh)
    if km.empty or "kmeans_cluster" not in km.columns or "region" not in km.columns:
        return {"distribution": []}
    cross = (pd.crosstab(km["region"], km["kmeans_cluster"], normalize="index")
             .mul(100).round(1).reset_index())
    cross.columns = ["region"] + [f"cluster_{c}" for c in cross.columns[1:]]
    return {"distribution": safe_dict(cross)}


@router.get("/root-cause/results")
async def root_cause_results():
    json_path = Path("models/classification/root_cause_results.json")
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    rf_path   = Path("models/classification/random_forest.pkl")
    xgb_path  = Path("models/classification/xgboost_classifier.pkl")
    le_path   = Path("models/classification/label_encoder.pkl")
    feat_path = Path("models/classification/feature_cols.pkl")
    if not all(p.exists() for p in [rf_path, xgb_path, le_path]):
        return {"best_model": None, "rf_report": {}, "xgb_report": {}, "classes": [],
                "feature_importance": [], "confusion_matrices": {},
                "message": "Run Notebook 05 to generate model files"}
    try:
        import joblib
        rf           = joblib.load(str(rf_path))
        xgb_model    = joblib.load(str(xgb_path))
        le           = joblib.load(str(le_path))
        feature_cols = joblib.load(str(feat_path)) if feat_path.exists() else []
        rf_imps  = list(getattr(rf,        "feature_importances_", []))
        xgb_imps = list(getattr(xgb_model, "feature_importances_", []))
        fi = []
        for i, feat in enumerate(feature_cols):
            a = float(rf_imps[i])  if i < len(rf_imps)  else 0.0
            b = float(xgb_imps[i]) if i < len(xgb_imps) else 0.0
            fi.append({"feature": feat, "importance_rf": round(a, 5),
                       "importance_xgb": round(b, 5), "importance_mean": round((a + b) / 2, 5)})
        fi.sort(key=lambda x: x["importance_mean"], reverse=True)
        return {"best_model": "xgboost", "classes": list(le.classes_),
                "rf_report": {}, "xgb_report": {}, "feature_importance": fi,
                "confusion_matrices": {}, "message": "Partial — run full notebook for metrics"}
    except Exception as exc:
        logger.exception("root_cause_results error")
        return {"best_model": None, "rf_report": {}, "xgb_report": {}, "classes": [],
                "feature_importance": [], "confusion_matrices": {}, "message": f"Error: {exc}"}


@router.get("/status")
async def status():
    out = {}
    for k, p in PATHS.items():
        cached_df, ts = _cache.get(k, (None, None))
        age  = round(time.monotonic() - ts, 1) if ts else None
        info = {"exists": p.exists(), "path": str(p),
                "cached": cached_df is not None and not cached_df.empty,
                "cache_age": f"{age}s" if age is not None else "not cached"}
        if cached_df is not None and not cached_df.empty:
            info["columns"]   = list(cached_df.columns)
            info["row_count"] = len(cached_df)
        elif p.exists():
            try:
                import pyarrow.parquet as pq
                info["columns"] = pq.read_schema(str(p)).names
            except Exception:
                info["columns"] = "unreadable"
        out[k] = info
    return out


# ── App assembly ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from src.nlp.auth_api import init_db
        init_db()
        logger.info("Auth DB initialised")
    except Exception as exc:
        logger.warning("Auth DB init failed (non-fatal): %s", exc)
    yield
    _cache.clear()
    logger.info("Cache cleared on shutdown")


app = FastAPI(title="SpiriComp Analytics API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": str(request.url.path)},
        headers={"Access-Control-Allow-Origin": "*"},
    )

# ── FIX: notifications import wrapped in try/except ──────────────────
# Previously this was a bare import — if notifications.py was missing,
# the entire module crashed and app was never defined → all routes 404.
try:
    from src.api.notifications import router as notif_router
    app.include_router(notif_router)
    logger.info("Notification routes registered")
except Exception as exc:
    logger.warning("Notifications module not available (non-fatal): %s", exc)

# ── Analytics routes (MUST come after app is defined) ─────────────────
app.include_router(router)

# ── Auth routes ───────────────────────────────────────────────────────
try:
    from src.nlp.auth_api import router as auth_router
    app.include_router(auth_router)
    logger.info("Auth routes registered")
except Exception as exc:
    logger.warning("Could not register auth routes: %s", exc)

# ── NLP / complaint routes ────────────────────────────────────────────
try:
    from src.nlp.nlp_api import router as nlp_router
    app.include_router(nlp_router)
    logger.info("NLP routes registered — complaints form active")
except Exception as exc:
    logger.warning("NLP module not available (complaints form will 404): %s", exc)