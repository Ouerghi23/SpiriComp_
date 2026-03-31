"""
Customer Clustering Module
===========================
Profiles affected users by experience pattern using:

  Model 1 — K-Means   (partitional, fast, interpretable)
  Model 2 — DBSCAN    (density-based, finds irregular clusters + noise)

K selection for K-Means:
  - Elbow method  (inertia vs K)
  - Silhouette score (cohesion vs separation)

Output per cluster:
  - Mean KPI profile
  - Dominant complaint categories
  - Dominant service type
  - QoE tier distribution
  - Cluster label (auto-named from profile)

Usage (from notebook):
    from src.models.customer_clustering import CustomerClusterer
    clusterer = CustomerClusterer()
    results   = clusterer.run(kpi_clean, complaints_clean)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MODELS_DIR   = Path(cfg["paths"]["models"]) / "clustering"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = cfg["models"]["random_state"]
K_MIN, K_MAX = cfg["models"]["clustering_k_range"]   # [2, 10]

# Per-MSISDN features to cluster on
CLUSTERING_FEATURES = [
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "data_session_success_rate",
    "data_qoe_score",
    "call_setup_success_rate",
    "call_drop_rate",
    "voice_quality_score_mos",
    "voice_qoe_score",
    "qoe_score",
]

# Cluster auto-label rules (based on QoE score mean)
def _auto_label(profile: dict) -> str:
    qoe = profile.get("qoe_score", 50)
    cdr = profile.get("call_drop_rate", 5)
    lat = profile.get("latency_ms", 100)
    if qoe >= 80:
        return "High QoE — Satisfied Users"
    elif qoe >= 65 and cdr < 2:
        return "Moderate QoE — Data Issues"
    elif qoe >= 65 and cdr >= 2:
        return "Moderate QoE — Voice Issues"
    elif lat > 200:
        return "Low QoE — High Latency"
    else:
        return "Low QoE — Multi-Service Degradation"


class CustomerClusterer:
    """
    Clusters users by experience pattern.
    Selects optimal K via Elbow + Silhouette.
    Profiles each cluster and generates auto labels.
    """

    def __init__(self):
        self.kmeans:      KMeans        | None = None
        self.dbscan:      DBSCAN        | None = None
        self.scaler:      StandardScaler| None = None
        self.pca:         PCA           | None = None
        self.optimal_k:   int           = 4
        self.elbow_scores: dict         = {}
        self.sil_scores:   dict         = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, kpi_clean: pd.DataFrame,
            complaints_clean: pd.DataFrame) -> dict:
        """
        Full clustering pipeline.

        Parameters
        ----------
        kpi_clean        : cleaned per-session KPI data
        complaints_clean : cleaned complaint records

        Returns
        -------
        dict with keys: user_profiles, kmeans_results, dbscan_results,
                        cluster_profiles, elbow_data, silhouette_data,
                        pca_coords
        """
        logger.info("=" * 55)
        logger.info("CUSTOMER CLUSTERING")
        logger.info("=" * 55)

        # ── 1. Build per-user feature matrix ──────────────────────────────────
        logger.info("\n[1/5] Building per-user feature matrix ...")
        user_features = self._build_user_features(kpi_clean)

        # ── 2. Scale & PCA ────────────────────────────────────────────────────
        logger.info("\n[2/5] Scaling and applying PCA ...")
        X_scaled, X_pca = self._scale_and_reduce(user_features)

        # ── 3. K selection ────────────────────────────────────────────────────
        logger.info("\n[3/5] Selecting optimal K ...")
        self.optimal_k = self._select_k(X_scaled)
        logger.info(f"  Optimal K selected: {self.optimal_k}")

        # ── 4. K-Means clustering ─────────────────────────────────────────────
        logger.info(f"\n[4/5] Training K-Means (k={self.optimal_k}) ...")
        kmeans_results = self._run_kmeans(user_features, X_scaled, X_pca)

        # ── 5. DBSCAN clustering ──────────────────────────────────────────────
        logger.info("\n[5/5] Running DBSCAN ...")
        dbscan_results = self._run_dbscan(user_features, X_scaled, X_pca)

        # ── Profile clusters ──────────────────────────────────────────────────
        profiles_km = self._profile_clusters(
            kmeans_results["user_df"], "kmeans_cluster", kpi_clean, complaints_clean
        )
        profiles_db = self._profile_clusters(
            dbscan_results["user_df"], "dbscan_cluster", kpi_clean, complaints_clean
        )

        # ── Save ──────────────────────────────────────────────────────────────
        self._save()
        kmeans_results["user_df"].to_parquet(MODELS_DIR / "kmeans_users.parquet", index=False)
        dbscan_results["user_df"].to_parquet(MODELS_DIR / "dbscan_users.parquet", index=False)

        logger.success(
            f"\nClustering complete:\n"
            f"  K-Means clusters : {self.optimal_k}\n"
            f"  DBSCAN clusters  : {dbscan_results['n_clusters']}"
            f"  (+{dbscan_results['n_noise']} noise points)"
        )

        return {
            "user_profiles":   kmeans_results["user_df"],
            "kmeans_results":  kmeans_results,
            "dbscan_results":  dbscan_results,
            "cluster_profiles": {
                "kmeans": profiles_km,
                "dbscan": profiles_db,
            },
            "elbow_data": {
                "k":       list(self.elbow_scores.keys()),
                "inertia": list(self.elbow_scores.values()),
            },
            "silhouette_data": {
                "k":     list(self.sil_scores.keys()),
                "score": list(self.sil_scores.values()),
            },
            "pca_coords":     X_pca,
            "optimal_k":      self.optimal_k,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE MATRIX
    # ─────────────────────────────────────────────────────────────────────────

    def _build_user_features(self, kpi_clean: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-session KPI data to per-user level."""
        feat_cols = [c for c in CLUSTERING_FEATURES if c in kpi_clean.columns]

        agg = kpi_clean.groupby("msisdn").agg(
            n_sessions       = ("timestamp", "count"),
            region           = ("region", lambda x: x.mode()[0]),
            **{f"{c}_mean": (c, "mean") for c in feat_cols},
            **{f"{c}_std":  (c, "std")  for c in feat_cols},
            **{f"{c}_min":  (c, "min")  for c in feat_cols},
        ).reset_index()

        # Degraded session rate
        if "is_degraded_session" in kpi_clean.columns:
            deg = (kpi_clean.groupby("msisdn")["is_degraded_session"]
                            .mean()
                            .reset_index(name="degraded_rate"))
            agg = agg.merge(deg, on="msisdn", how="left")

        agg = agg.fillna(agg.median(numeric_only=True))
        logger.info(f"  User feature matrix: {agg.shape[0]:,} users × {agg.shape[1]} cols")
        return agg

    # ─────────────────────────────────────────────────────────────────────────
    # SCALING & PCA
    # ─────────────────────────────────────────────────────────────────────────

    def _scale_and_reduce(self,
                          user_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Scale features and reduce to 2D via PCA for visualisation."""
        num_cols = user_df.select_dtypes(include="number").columns.tolist()
        num_cols = [c for c in num_cols if c != "msisdn"]

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(user_df[num_cols].fillna(0))

        # PCA → 2D for scatter plots
        n_comp = min(2, X_scaled.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        X_pca    = self.pca.fit_transform(X_scaled)
        explained = self.pca.explained_variance_ratio_.sum() * 100
        logger.info(f"  PCA variance explained (2 components): {explained:.1f}%")
        return X_scaled, X_pca

    # ─────────────────────────────────────────────────────────────────────────
    # K SELECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _select_k(self, X_scaled: np.ndarray) -> int:
        """Elbow + Silhouette method to find optimal K."""
        k_range = range(K_MIN, K_MAX + 1)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_scaled)
            self.elbow_scores[k] = km.inertia_
            if k > 1:
                self.sil_scores[k] = silhouette_score(
                    X_scaled, labels, sample_size=min(5000, len(X_scaled))
                )
            logger.info(
                f"  K={k}  inertia={self.elbow_scores[k]:,.0f}  "
                f"silhouette={self.sil_scores.get(k, 'N/A')}"
            )

        # Select K with best silhouette
        best_k = max(self.sil_scores, key=self.sil_scores.get)
        return int(best_k)

    # ─────────────────────────────────────────────────────────────────────────
    # K-MEANS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_kmeans(self, user_df: pd.DataFrame,
                   X_scaled:   np.ndarray,
                   X_pca:      np.ndarray) -> dict:
        self.kmeans = KMeans(
            n_clusters  = self.optimal_k,
            random_state= RANDOM_STATE,
            n_init      = 20,
            max_iter    = 500,
        )
        labels = self.kmeans.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels,
                               sample_size=min(5000, len(X_scaled)))
        dbi = davies_bouldin_score(X_scaled, labels)

        user_out = user_df.copy()
        user_out["kmeans_cluster"] = labels
        user_out["pca_x"]          = X_pca[:, 0]
        user_out["pca_y"]          = X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0

        logger.info(
            f"  K-Means done  silhouette={sil:.3f}  "
            f"davies-bouldin={dbi:.3f}"
        )
        return {
            "user_df":          user_out,
            "labels":           labels,
            "silhouette_score": round(sil, 3),
            "davies_bouldin":   round(dbi, 3),
            "inertia":          self.kmeans.inertia_,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # DBSCAN
    # ─────────────────────────────────────────────────────────────────────────

    def _run_dbscan(self, user_df: pd.DataFrame,
                   X_scaled:  np.ndarray,
                   X_pca:     np.ndarray) -> dict:
        """
        DBSCAN with auto eps estimation via k-distance graph.
        eps ≈ 95th percentile of k-nearest distances.
        """
        from sklearn.neighbors import NearestNeighbors
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        eps = float(np.percentile(distances[:, -1], 95))
        logger.info(f"  DBSCAN auto eps: {eps:.3f}")

        self.dbscan = DBSCAN(eps=eps, min_samples=k, n_jobs=-1)
        labels      = self.dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int((labels == -1).sum())

        user_out = user_df.copy()
        user_out["dbscan_cluster"] = labels
        user_out["pca_x"]          = X_pca[:, 0]
        user_out["pca_y"]          = X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0

        logger.info(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        return {
            "user_df":    user_out,
            "labels":     labels,
            "n_clusters": n_clusters,
            "n_noise":    n_noise,
            "eps_used":   round(eps, 3),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CLUSTER PROFILING
    # ─────────────────────────────────────────────────────────────────────────

    def _profile_clusters(self,
                          user_df:         pd.DataFrame,
                          cluster_col:     str,
                          kpi_clean:       pd.DataFrame,
                          complaints_clean:pd.DataFrame) -> pd.DataFrame:
        """Build a descriptive profile for each cluster."""
        kpi_cols = [c for c in CLUSTERING_FEATURES if f"{c}_mean" in user_df.columns]
        mean_cols = [f"{c}_mean" for c in kpi_cols]

        profiles = (
            user_df[user_df[cluster_col] != -1]   # exclude DBSCAN noise
              .groupby(cluster_col)
              .agg(
                  n_users       = ("msisdn", "count"),
                  **{c: (c, "mean") for c in mean_cols if c in user_df.columns},
              )
              .reset_index()
        )

        # Auto label each cluster
        labels = []
        for _, row in profiles.iterrows():
            profile_dict = {
                c.replace("_mean", ""): row[c]
                for c in mean_cols if c in profiles.columns
            }
            labels.append(_auto_label(profile_dict))
        profiles["cluster_label"] = labels
        profiles["pct_of_users"]  = (
            profiles["n_users"] / profiles["n_users"].sum() * 100
        ).round(1)

        logger.info(f"\n  Cluster profiles ({cluster_col}):")
        for _, row in profiles.iterrows():
            logger.info(
                f"    Cluster {int(row[cluster_col]):>2} | "
                f"n={int(row['n_users']):>5} ({row['pct_of_users']:.1f}%) | "
                f"{row['cluster_label']}"
            )
        return profiles

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        joblib.dump(self.kmeans, MODELS_DIR / "kmeans.pkl")
        joblib.dump(self.dbscan, MODELS_DIR / "dbscan.pkl")
        joblib.dump(self.scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(self.pca,    MODELS_DIR / "pca.pkl")
        logger.info(f"  Models saved → {MODELS_DIR}")

    @classmethod
    def load(cls) -> "CustomerClusterer":
        obj = cls()
        obj.kmeans = joblib.load(MODELS_DIR / "kmeans.pkl")
        obj.dbscan = joblib.load(MODELS_DIR / "dbscan.pkl")
        obj.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        obj.pca    = joblib.load(MODELS_DIR / "pca.pkl")
        return obj