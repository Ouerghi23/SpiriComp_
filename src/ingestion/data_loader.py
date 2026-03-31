from pathlib import Path
import pandas as pd
import yaml
from loguru import logger


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


def _resolve_data_path() -> Path:
    real_path = Path(cfg["paths"]["raw_data"])
    synth_path = Path(cfg["paths"]["synthetic_data"])

    real_complaints = real_path / cfg["data"]["complaint_file"]
    if real_complaints.exists():
        logger.info("Real dataset detected — loading from data/raw/")
        return real_path
    else:
        logger.warning("Real data not found — loading synthetic data from data/synthetic/")
        logger.warning("Replace data/synthetic/ files with real Huawei data when available.")
        return synth_path


def load_complaints(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = _resolve_data_path() / cfg["data"]["complaint_file"]

    logger.info(f"Loading complaints from {path}")
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df = _standardise_complaints(df)
    logger.success(f"Complaints loaded: {len(df):,} rows × {df.shape[1]} columns")
    return df


def load_kpi_data(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = _resolve_data_path() / cfg["data"]["kpi_file"]

    logger.info(f"Loading KPI data from {path}")
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df = _standardise_kpi(df)
    logger.success(f"KPI data loaded: {len(df):,} rows × {df.shape[1]} columns")
    return df


def _standardise_complaints(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = ["service_type", "complaint_category", "complaint_subcategory",
                "region", "city", "customer_segment", "priority"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Derived time columns (useful for EDA and feature engineering)
    df["date"]        = df["timestamp"].dt.date
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["week"]        = df["timestamp"].dt.isocalendar().week.astype(int)
    df["month"]       = df["timestamp"].dt.month
    df["year"]        = df["timestamp"].dt.year

    return df


def _standardise_kpi(df: pd.DataFrame) -> pd.DataFrame:
    kpi_num_cols = [
        "dl_throughput_mbps", "ul_throughput_mbps", "latency_ms",
        "packet_loss_pct", "data_session_success_rate", "data_qoe_score",
        "call_setup_success_rate", "call_drop_rate", "voice_quality_score_mos",
        "handover_success_rate", "voice_qoe_score", "qoe_score",
    ]
    for col in kpi_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"]  = df["timestamp"].dt.date
    df["hour"]  = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["year"]  = df["timestamp"].dt.year

    return df


def get_data_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\n  Missing values:")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"    {col:<35} {cnt:>6,}  ({pct:.1f}%)")
    print(f"{'='*55}\n")