"""
Root Cause Classification Module
==================================
Maps KPI feature vectors to known complaint root causes.

  Model 1 — Random Forest   (interpretable baseline)
  Model 2 — XGBoost         (best performance, SHAP explainable)

The target variable is complaint_category derived from the cleaned
complaint dataset, joined with KPI features at (region, date) level.

Output:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix
  - SHAP feature importance (global + per-prediction)
  - Saved model artifacts

Usage (from notebook):
    from src.models.root_cause_classifier import RootCauseClassifier
    clf = RootCauseClassifier()
    results = clf.run(complaints_clean, feature_matrix)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — explainability plots unavailable")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MODELS_DIR   = Path(cfg["paths"]["models"]) / "classification"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = cfg["models"]["random_state"]
TEST_SIZE    = cfg["models"]["test_size"]

# KPI features most relevant to root cause identification
CLASSIFICATION_FEATURES = [
    "dl_throughput_mbps_mean", "dl_throughput_mbps_p10",
    "ul_throughput_mbps_mean",
    "latency_ms_mean", "latency_ms_max",
    "packet_loss_pct_mean",
    "data_session_success_rate_mean",
    "data_qoe_score_mean",
    "call_setup_success_rate_mean",
    "call_drop_rate_mean",
    "voice_quality_score_mos_mean",
    "handover_success_rate_mean",
    "voice_qoe_score_mean",
    "qoe_score_mean", "qoe_score_p10",
    "degraded_session_rate_pct",
    "total_complaints_roll_mean_7d",
    "is_weekend", "is_peak_hour",
    "month_sin", "month_cos",
]

# Map complaint categories to root cause groups
# (simplifies multi-class problem for better model performance)
ROOT_CAUSE_MAP = {
    "Slow Data":              "Data_Performance",
    "Intermittent Connection":"Data_Performance",
    "No Service":             "Coverage",
    "Roaming Issue":          "Coverage",
    "Call Drop":              "Voice_Quality",
    "Poor Voice Quality":     "Voice_Quality",
    "Call Setup Failure":     "Network_Congestion",
    "Sms Failure":            "Network_Congestion",
    "Other":                  "Other",
}


class RootCauseClassifier:
    """
    Trains Random Forest and XGBoost classifiers to map
    KPI degradation patterns to complaint root causes.
    """

    def __init__(self):
        self.rf_model:   RandomForestClassifier | None = None
        self.xgb_model:  XGBClassifier          | None = None
        self.label_enc:  LabelEncoder            | None = None
        self.feature_cols: list[str]             = []
        self.classes_:    np.ndarray             | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, complaints_clean: pd.DataFrame,
            feature_matrix: pd.DataFrame) -> dict:
        """
        Full root cause classification pipeline.

        Parameters
        ----------
        complaints_clean : cleaned complaint records (from Phase 2)
        feature_matrix   : joined feature matrix (region × date level)

        Returns
        -------
        dict with keys: rf_report, xgb_report, shap_values,
                        feature_importance, confusion_matrices, best_model
        """
        logger.info("=" * 55)
        logger.info("ROOT CAUSE CLASSIFICATION")
        logger.info("=" * 55)

        # ── 1. Build labelled dataset ─────────────────────────────────────────
        logger.info("\n[1/4] Building labelled dataset ...")
        X, y, self.feature_cols = self._build_dataset(complaints_clean, feature_matrix)

        # ── 2. Encode labels ──────────────────────────────────────────────────
        self.label_enc = LabelEncoder()
        y_enc          = self.label_enc.fit_transform(y)
        self.classes_  = self.label_enc.classes_
        logger.info(f"  Classes ({len(self.classes_)}): {list(self.classes_)}")

        # ── 3. Train/test split (stratified) ─────────────────────────────────
        split = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y_enc[:split],  y_enc[split:]
        logger.info(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

        # ── 4. Train models ───────────────────────────────────────────────────
        logger.info("\n[2/4] Training Random Forest ...")
        rf_results  = self._train_random_forest(X_train, X_test, y_train, y_test)

        logger.info("\n[3/4] Training XGBoost ...")
        xgb_results = self._train_xgboost(X_train, X_test, y_train, y_test)

        # ── 5. SHAP explainability (XGBoost) ─────────────────────────────────
        shap_values = None
        if SHAP_AVAILABLE:
            logger.info("\n[4/4] Computing SHAP values ...")
            shap_values = self._compute_shap(X_test)
        else:
            logger.info("\n[4/4] SHAP skipped (not installed)")

        # ── 6. Select best model ──────────────────────────────────────────────
        best = ("xgboost" if xgb_results["f1_macro"] >= rf_results["f1_macro"]
                else "random_forest")
        logger.info(f"\n  Best model: {best.upper()}")
        logger.info(
            f"  RF  F1-macro: {rf_results['f1_macro']:.3f}  "
            f"Accuracy: {rf_results['accuracy']:.3f}"
        )
        logger.info(
            f"  XGB F1-macro: {xgb_results['f1_macro']:.3f}  "
            f"Accuracy: {xgb_results['accuracy']:.3f}"
        )

        # ── Save ──────────────────────────────────────────────────────────────
        self._save()

        return {
            "rf_report":          rf_results,
            "xgb_report":         xgb_results,
            "shap_values":        shap_values,
            "feature_importance": self._feature_importance(),
            "confusion_matrices": {
                "random_forest": rf_results["confusion_matrix"],
                "xgboost":       xgb_results["confusion_matrix"],
            },
            "classes":     list(self.classes_),
            "best_model":  best,
            "X_test":      X_test,
            "y_test":      y_test,
            "feature_cols": self.feature_cols,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_dataset(self, complaints: pd.DataFrame,
                       feature_matrix: pd.DataFrame
                       ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """
        Join complaint categories with KPI features at (region, date) level.
        Target = root cause group derived from complaint_category.
        """
        # Dominant complaint category per region-day
        complaints["date"] = pd.to_datetime(complaints["timestamp"]).dt.date
        complaints["date"] = pd.to_datetime(complaints["date"])

        # Map to root cause group
        complaints["root_cause"] = (
            complaints["complaint_category"]
            .map(ROOT_CAUSE_MAP)
            .fillna("Other")
        )

        # Dominant root cause per region-day
        dominant = (
            complaints.groupby(["region", "date"])["root_cause"]
                      .agg(lambda x: x.value_counts().index[0])
                      .reset_index()
        )

        # Merge with feature matrix
        feature_matrix["date"] = pd.to_datetime(feature_matrix["date"])
        merged = feature_matrix.merge(dominant, on=["region", "date"], how="inner")

        # Select feature columns
        feat_cols = [c for c in CLASSIFICATION_FEATURES if c in merged.columns]

        # Add region dummies
        region_dummies = [c for c in merged.columns if c.startswith("region_")]
        feat_cols = feat_cols + region_dummies

        X = merged[feat_cols].fillna(merged[feat_cols].median())
        y = merged["root_cause"].values

        logger.info(
            f"  Dataset: {len(X):,} rows × {len(feat_cols)} features  "
            f"| {len(set(y))} classes"
        )
        logger.info(f"  Class distribution:\n{pd.Series(y).value_counts().to_string()}")

        return X, y, feat_cols

    # ─────────────────────────────────────────────────────────────────────────
    # RANDOM FOREST
    # ─────────────────────────────────────────────────────────────────────────

    def _train_random_forest(self,
                             X_train, X_test, y_train, y_test) -> dict:
        self.rf_model = RandomForestClassifier(
            n_estimators     = 300,
            max_depth        = 12,
            min_samples_leaf = 5,
            class_weight     = "balanced",
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
        )
        self.rf_model.fit(X_train, y_train)
        y_pred = self.rf_model.predict(X_test)

        # Cross-validation F1
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            self.rf_model, X_train, y_train,
            scoring="f1_macro", cv=cv, n_jobs=-1
        )

        report = classification_report(
            y_test, y_pred,
            target_names=self.classes_,
            output_dict=True,
            zero_division=0
        )
        return {
            "f1_macro":          f1_score(y_test, y_pred, average="macro", zero_division=0),
            "accuracy":          accuracy_score(y_test, y_pred),
            "cv_f1_mean":        cv_scores.mean(),
            "cv_f1_std":         cv_scores.std(),
            "classification_report": report,
            "confusion_matrix":  confusion_matrix(y_test, y_pred),
            "y_pred":            y_pred,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # XGBOOST
    # ─────────────────────────────────────────────────────────────────────────

    def _train_xgboost(self,
                       X_train, X_test, y_train, y_test) -> dict:
        self.xgb_model = XGBClassifier(
            n_estimators     = 400,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            use_label_encoder= False,
            eval_metric      = "mlogloss",
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
            verbosity        = 0,
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_pred = self.xgb_model.predict(X_test)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            self.xgb_model, X_train, y_train,
            scoring="f1_macro", cv=cv, n_jobs=-1
        )

        report = classification_report(
            y_test, y_pred,
            target_names=self.classes_,
            output_dict=True,
            zero_division=0
        )
        return {
            "f1_macro":          f1_score(y_test, y_pred, average="macro", zero_division=0),
            "accuracy":          accuracy_score(y_test, y_pred),
            "cv_f1_mean":        cv_scores.mean(),
            "cv_f1_std":         cv_scores.std(),
            "classification_report": report,
            "confusion_matrix":  confusion_matrix(y_test, y_pred),
            "y_pred":            y_pred,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SHAP
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_shap(self, X_test: pd.DataFrame) -> object:
        """Compute SHAP values for XGBoost model."""
        try:
            explainer   = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer.shap_values(X_test)
            logger.info(f"  SHAP values computed: shape {np.array(shap_values).shape}")
            joblib.dump(shap_values, MODELS_DIR / "shap_values.pkl")
            return shap_values
        except Exception as e:
            logger.warning(f"  SHAP computation failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────────

    def _feature_importance(self) -> pd.DataFrame:
        """Return merged RF + XGB feature importances."""
        rows = []
        if self.rf_model and self.feature_cols:
            for feat, imp in zip(self.feature_cols, self.rf_model.feature_importances_):
                rows.append({"feature": feat, "importance_rf": imp})
        df_rf = pd.DataFrame(rows)

        rows = []
        if self.xgb_model and self.feature_cols:
            for feat, imp in zip(self.feature_cols, self.xgb_model.feature_importances_):
                rows.append({"feature": feat, "importance_xgb": imp})
        df_xgb = pd.DataFrame(rows)

        if df_rf.empty and df_xgb.empty:
            return pd.DataFrame()

        merged = df_rf.merge(df_xgb, on="feature", how="outer").fillna(0)
        merged["importance_mean"] = (
            merged[["importance_rf", "importance_xgb"]].mean(axis=1)
        )
        return merged.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        joblib.dump(self.rf_model,   MODELS_DIR / "random_forest.pkl")
        joblib.dump(self.xgb_model,  MODELS_DIR / "xgboost_classifier.pkl")
        joblib.dump(self.label_enc,  MODELS_DIR / "label_encoder.pkl")
        joblib.dump(self.feature_cols, MODELS_DIR / "feature_cols.pkl")
        logger.info(f"  Models saved → {MODELS_DIR}")

    @classmethod
    def load(cls) -> "RootCauseClassifier":
        obj = cls()
        obj.rf_model     = joblib.load(MODELS_DIR / "random_forest.pkl")
        obj.xgb_model    = joblib.load(MODELS_DIR / "xgboost_classifier.pkl")
        obj.label_enc    = joblib.load(MODELS_DIR / "label_encoder.pkl")
        obj.feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
        obj.classes_     = obj.label_enc.classes_
        return obj

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict root cause + confidence for new data."""
        X = X[self.feature_cols].fillna(0)
        probs     = self.xgb_model.predict_proba(X)
        pred_idx  = probs.argmax(axis=1)
        pred_label= self.label_enc.inverse_transform(pred_idx)
        confidence= probs.max(axis=1)
        return pd.DataFrame({
            "predicted_root_cause": pred_label,
            "confidence":           confidence.round(3),
        })