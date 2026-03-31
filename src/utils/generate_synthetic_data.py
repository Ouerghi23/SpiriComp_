import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import random
from datetime import datetime, timedelta
from loguru import logger

#Load Config 
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

RANDOM_STATE = cfg["models"]["random_state"]
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

REGIONS      = cfg["data"]["regions"]
SERVICE_TYPES = cfg["data"]["service_types"]
CATEGORIES   = cfg["data"]["complaint_categories"]
OUT_DIR      = Path(__file__).resolve().parents[2] / cfg["paths"]["synthetic_data"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Category → Service mapping 
CATEGORY_SERVICE_MAP = {
    "No Service":             ["Data", "Voice", "SMS"],
    "Slow Data":              ["Data"],
    "Call Drop":              ["Voice"],
    "Call Setup Failure":     ["Voice"],
    "SMS Failure":            ["SMS"],
    "Poor Voice Quality":     ["Voice"],
    "Intermittent Connection":["Data", "Voice"],
    "Roaming Issue":          ["Data", "Voice", "SMS"],
}

SUBCATEGORY_MAP = {
    "No Service":             ["Coverage Gap", "SIM Issue", "Network Outage"],
    "Slow Data":              ["Congestion", "Coverage Edge", "Device Issue"],
    "Call Drop":              ["Handover Failure", "Signal Loss", "Interference"],
    "Call Setup Failure":     ["Network Busy", "Routing Error", "Congestion"],
    "SMS Failure":            ["Delivery Failure", "Encoding Error", "Routing"],
    "Poor Voice Quality":     ["Codec Issue", "Packet Loss", "Jitter"],
    "Intermittent Connection":["Handover", "Coverage Fluctuation", "Device"],
    "Roaming Issue":          ["Partner Network", "Authentication", "Billing"],
}

CUSTOMER_SEGMENTS = ["Standard", "Premium", "Enterprise", "VIP"]
PRIORITIES        = ["Low", "Medium", "High", "Critical"]

#  Cell Tower IDs (simulated) 
CELLS = [f"CELL_{str(i).zfill(4)}" for i in range(1, 201)]

# Region → approximate lat/lon bounding boxes
REGION_COORDS = {
    "Tunis":     (36.7, 36.9,  10.1, 10.3),
    "Sfax":      (34.7, 34.9,  10.7, 10.9),
    "Sousse":    (35.8, 36.0,  10.5, 10.7),
    "Kairouan":  (35.6, 35.8,   9.9, 10.1),
    "Bizerte":   (37.2, 37.4,   9.8, 10.0),
    "Gabes":     (33.8, 34.0,   9.9, 10.1),
    "Ariana":    (36.8, 37.0,  10.1, 10.3),
    "Gafsa":     (34.3, 34.5,   8.7,  8.9),
    "Monastir":  (35.7, 35.9,  10.7, 10.9),
    "Ben Arous": (36.7, 36.8,  10.2, 10.4),
}


def _random_lat_lon(region: str):
    lat_min, lat_max, lon_min, lon_max = REGION_COORDS[region]
    return (
        round(np.random.uniform(lat_min, lat_max), 5),
        round(np.random.uniform(lon_min, lon_max), 5),
    )


def _inject_hotspot_bias(region: str, hour: int) -> float:
    """Return a complaint rate multiplier for region/time hotspots."""
    hotspot_regions = {"Tunis": 2.5, "Sfax": 1.8, "Sousse": 1.5}
    peak_hours      = {8, 9, 12, 13, 17, 18, 19, 20}
    region_mult = hotspot_regions.get(region, 1.0)
    time_mult   = 1.8 if hour in peak_hours else 1.0
    return region_mult * time_mult


# ************************
def generate_complaints(n_records: int = 50_000,
                        start: str = "2023-01-01",
                        end:   str = "2024-06-30") -> pd.DataFrame:

    logger.info(f"Generating {n_records:,} complaint records ...")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days

    records = []
    for i in range(n_records):
        region   = random.choice(REGIONS)
        hour     = np.random.choice(range(24),
                       p=_hour_distribution())
        day_offset = np.random.randint(0, delta)
        ts = start_dt + timedelta(days=day_offset,
                                  hours=int(hour),
                                  minutes=np.random.randint(0, 60),
                                  seconds=np.random.randint(0, 60))

        # Bias: hotspot regions produce more complaints at peak hours
        mult = _inject_hotspot_bias(region, hour)
        category = _weighted_category(region)
        service  = random.choice(CATEGORY_SERVICE_MAP[category])
        lat, lon = _random_lat_lon(region)

        records.append({
            "case_id":              f"CASE_{str(i+1).zfill(7)}",
            "timestamp":            ts.strftime("%Y-%m-%d %H:%M:%S"),
            "service_type":         service,
            "complaint_category":   category,
            "complaint_subcategory":random.choice(SUBCATEGORY_MAP[category]),
            "region":               region,
            "city":                 region,        # simplified; extend with city list if needed
            "latitude":             lat,
            "longitude":            lon,
            "cell_id":              random.choice(CELLS),
            "customer_segment":     random.choices(
                                        CUSTOMER_SEGMENTS,
                                        weights=[50, 30, 15, 5])[0],
            "priority":             random.choices(
                                        PRIORITIES,
                                        weights=[40, 35, 20, 5])[0],
            "msisdn":               f"216{np.random.randint(20_000_000, 99_999_999)}",
            "_hotspot_multiplier":  round(mult, 2),   # debug column; drop before modelling
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Inject ~3 % missing values in non-key columns (realistic data quality issue)
    for col in ["complaint_subcategory", "cell_id", "latitude", "longitude"]:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    out_path = OUT_DIR / "complaints.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Complaints saved → {out_path}  ({len(df):,} rows)")
    return df


def _hour_distribution():
    """Realistic hourly distribution (higher probability during day/evening)."""
    weights = np.array([
        1, 1, 1, 1, 1, 2,        # 00-05
        3, 5, 8, 8, 7, 7,        # 06-11
        8, 7, 6, 6, 7, 9,        # 12-17
        10, 9, 7, 5, 3, 2,       # 18-23
    ], dtype=float)
    return weights / weights.sum()


def _weighted_category(region: str) -> str:
    """Some complaint categories dominate in certain regions."""
    if region in ("Tunis", "Ariana", "Ben Arous"):
        weights = [5, 20, 10, 10, 8, 10, 25, 12]   # urban: slow data dominates
    elif region in ("Gabes", "Gafsa"):
        weights = [20, 10, 15, 15, 8, 12, 12, 8]   # rural: no service dominates
    else:
        weights = [12, 15, 13, 12, 8, 12, 18, 10]  # balanced
    total = sum(weights)
    probs = [w / total for w in weights]
    return np.random.choice(CATEGORIES, p=probs)



# KPI DATASET
def generate_kpi_data(n_msisdns: int = 5_000,
                      start: str = "2023-01-01",
                      end:   str = "2024-06-30") -> pd.DataFrame:
   
    logger.info(f"Generating KPI records for {n_msisdns:,} users ...")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days

    msisdns = [f"216{np.random.randint(20_000_000, 99_999_999)}"
               for _ in range(n_msisdns)]

    records = []
    for msisdn in msisdns:
        # Each MSISDN has between 20–200 session snapshots
        n_sessions = np.random.randint(20, 200)
        region = random.choice(REGIONS)
        cell   = random.choice(CELLS)

        for _ in range(n_sessions):
            day_offset = np.random.randint(0, delta)
            ts = start_dt + timedelta(days=day_offset,
                                      hours=np.random.randint(0, 24),
                                      minutes=np.random.randint(0, 60))

            # Introduce degradation events for ~15% of sessions
            degraded = np.random.random() < 0.15
            deg_factor = 0.3 if degraded else 1.0

            #  Data KPIs 
            dl_tp = max(0.1, np.random.lognormal(2.5, 0.8) * deg_factor)
            ul_tp = max(0.1, np.random.lognormal(1.5, 0.7) * deg_factor)
            lat   = max(5,   np.random.lognormal(3.5, 0.5) / deg_factor)
            pkt   = min(100, max(0, np.random.exponential(1.5) / deg_factor))
            dssr  = min(100, max(0, np.random.normal(95, 3) * deg_factor))
            d_qoe = _compute_data_qoe(dl_tp, lat, pkt, dssr)

            #  Voice KPIs 
            cssr  = min(100, max(0, np.random.normal(97, 2) * deg_factor))
            cdr   = min(100, max(0, np.random.exponential(0.8) / deg_factor))
            mos   = min(5,   max(1, np.random.normal(3.8, 0.4) * deg_factor))
            hsr   = min(100, max(0, np.random.normal(96, 2) * deg_factor))
            v_qoe = _compute_voice_qoe(cssr, cdr, mos, hsr)

            # ── Composite QoE 
            qoe_score = round(0.55 * d_qoe + 0.45 * v_qoe, 2)
            qoe_cat   = ("Good" if qoe_score >= 80
                         else "Fair" if qoe_score >= 60
                         else "Poor")

            records.append({
                "msisdn":                    msisdn,
                "timestamp":                 ts.strftime("%Y-%m-%d %H:%M:%S"),
                "cell_id":                   cell,
                "region":                    region,
                # Data
                "dl_throughput_mbps":        round(dl_tp, 3),
                "ul_throughput_mbps":        round(ul_tp, 3),
                "latency_ms":                round(lat, 1),
                "packet_loss_pct":           round(pkt, 3),
                "data_session_success_rate": round(dssr, 2),
                "data_qoe_score":            round(d_qoe, 2),
                # Voice
                "call_setup_success_rate":   round(cssr, 2),
                "call_drop_rate":            round(cdr, 3),
                "voice_quality_score_mos":   round(mos, 2),
                "handover_success_rate":     round(hsr, 2),
                "voice_qoe_score":           round(v_qoe, 2),
                # Composite
                "qoe_score":                 qoe_score,
                "qoe_category":              qoe_cat,
                "is_degraded_session":       int(degraded),   # ground truth label
            })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Inject ~2% missing values in KPI columns
    kpi_cols = ["dl_throughput_mbps", "latency_ms", "call_drop_rate", "voice_quality_score_mos"]
    for col in kpi_cols:
        mask = np.random.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    out_path = OUT_DIR / "kpi_data.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"KPI data saved → {out_path}  ({len(df):,} rows)")
    return df


def _compute_data_qoe(dl_tp, latency, packet_loss, success_rate) -> float:
    """Simple composite data QoE score (0–100)."""
    tp_score  = min(100, (dl_tp / 50) * 100)          # 50 Mbps = perfect
    lat_score = max(0, 100 - (latency / 5))            # 500ms = 0
    pkt_score = max(0, 100 - (packet_loss * 10))
    sr_score  = success_rate
    return round(0.35*tp_score + 0.25*lat_score + 0.20*pkt_score + 0.20*sr_score, 2)


def _compute_voice_qoe(cssr, cdr, mos, hsr) -> float:
    """Simple composite voice QoE score (0–100)."""
    mos_score = (mos / 5) * 100
    cdr_score = max(0, 100 - cdr * 20)
    return round(0.30*cssr + 0.25*cdr_score + 0.30*mos_score + 0.15*hsr, 2)



if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Huawei PFE — Synthetic Dataset Generator")
    logger.info("=" * 60)

    complaints = generate_complaints(n_records=50_000)
    kpi_data   = generate_kpi_data(n_msisdns=5_000)

    logger.info("\nDataset Summary")
    logger.info(f"  Complaints : {len(complaints):>10,} rows  ×  {len(complaints.columns)} columns")
    logger.info(f"  KPI Data   : {len(kpi_data):>10,} rows  ×  {len(kpi_data.columns)} columns")
    logger.info(f"\nFiles written to: {OUT_DIR.resolve()}")
    logger.info("Replace these files with real Huawei data when available.")