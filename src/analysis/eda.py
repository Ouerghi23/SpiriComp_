import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import yaml
from loguru import logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

FIGURES_DIR = Path(cfg["paths"]["figures"])
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

#  Plot style 
sns.set_theme(style="darkgrid", palette="husl")
COLORS = sns.color_palette("husl", 10)


# COMPLAINT EDA


def complaint_category_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    # pandas-version-safe: build counts explicitly to avoid reset_index column name changes
    vc = df["complaint_category"].value_counts()
    counts = pd.DataFrame({"complaint_category": vc.index, "count": vc.values})

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=counts, x="complaint_category", y="count", palette="husl", ax=ax)
    ax.set_title("Complaint Volume by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Complaints")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "complaint_category_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved → {path}")
    return fig


def complaint_by_service_type(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    counts = df["service_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=COLORS[:len(counts)], startangle=140)
    ax.set_title("Complaints by Service Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "complaints_service_type_pie.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def complaint_volume_over_time(df: pd.DataFrame,
                               freq: str = "W",
                               save: bool = True) -> plt.Figure:
    ts = (df.set_index("timestamp")
            .resample(freq)
            .size()
            .reset_index(name="count"))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts["timestamp"], ts["count"], linewidth=1.5, color=COLORS[0])
    ax.fill_between(ts["timestamp"], ts["count"], alpha=0.15, color=COLORS[0])
    ax.set_title(f"Complaint Volume Over Time (freq={freq})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Complaints")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"complaint_volume_timeseries_{freq}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def complaint_heatmap_hour_day(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (df.groupby(["day_of_week", "hour"])
               .size()
               .unstack(fill_value=0)
               .reindex(order))

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, ax=ax,
                cbar_kws={"label": "Complaint Count"})
    ax.set_title("Complaint Heatmap: Hour of Day × Day of Week", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "complaint_heatmap_hour_day.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def complaint_by_region(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    counts = df["region"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="barh", ax=ax, color=COLORS)
    ax.set_title("Complaint Volume by Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Complaints")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "complaints_by_region.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def missing_value_report(df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
    total = len(df)
    missing = df.isnull().sum()
    pct     = (missing / total * 100).round(2)
    report  = pd.DataFrame({
        "column":   missing.index,
        "missing":  missing.values,
        "pct":      pct.values,
        "dtype":    [str(df[c].dtype) for c in missing.index],
    }).query("missing > 0").sort_values("pct", ascending=False).reset_index(drop=True)
    logger.info(f"{name}: {len(report)} columns with missing values")
    return report



# KPI EDA

def kpi_distribution_plots(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    kpi_cols = [c for c in [
        "dl_throughput_mbps", "ul_throughput_mbps", "latency_ms",
        "packet_loss_pct", "data_session_success_rate", "data_qoe_score",
        "call_setup_success_rate", "call_drop_rate",
        "voice_quality_score_mos", "voice_qoe_score", "qoe_score",
    ] if c in df.columns]

    ncols = 3
    nrows = int(np.ceil(len(kpi_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(kpi_cols):
        axes[i].hist(df[col].dropna(), bins=50, color=COLORS[i % len(COLORS)], edgecolor="none", alpha=0.8)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("KPI Value Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "kpi_distributions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved → {path}")
    return fig


def qoe_score_by_region(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    order = df.groupby("region")["qoe_score"].median().sort_values().index
    sns.boxplot(data=df, x="region", y="qoe_score",
                order=order, palette="RdYlGn", ax=ax)
    ax.axhline(80, color="green", linestyle="--", linewidth=1, label="Good threshold (80)")
    ax.axhline(60, color="orange", linestyle="--", linewidth=1, label="Fair threshold (60)")
    ax.set_title("QoE Score Distribution by Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("QoE Score")
    ax.legend()
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "qoe_by_region_boxplot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def kpi_correlation_matrix(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    kpi_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ("hour", "month", "year", "is_degraded_session")]
    corr = df[kpi_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("KPI Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "kpi_correlation_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig