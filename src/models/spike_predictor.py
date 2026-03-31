"""
Spike Prediction Module
========================
Forecasts complaint volume by region and service type using three models:

  Model 1 — ARIMA/SARIMA   (classical time-series baseline)
  Model 2 — Prophet        (trend + seasonality, handles missing data well)
  Model 3 — XGBoost        (ML regressor on engineered lag/rolling features)

All models are trained per-region independently.
Results are compared on a held-out test period using MAE, RMSE, MAPE.
The best model per region is selected and saved.

Usage (from notebook):
    from src.models.spike_predictor import SpikePredictor
    predictor = SpikePredictor()
    results   = predictor.run(complaint_agg, feature_matrix)
"""

from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed — skipping Prophet model")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MODELS_DIR       = Path(cfg["paths"]["models"]) / "prediction"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE     = cfg["models"]["random_state"]
FORECAST_HORIZON = cfg["models"]["forecast_horizon_days"]   # 7

# Lag/rolling features to use for XGBoost
XGB_FEATURES = [
    "total_complaints_lag_1d",
    "total_complaints_lag_3d",
    "total_complaints_lag_7d",
    "total_complaints_lag_14d",
    "total_complaints_roll_mean_7d",
    "total_complaints_roll_std_7d",
    "total_complaints_roll_mean_14d",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "is_weekend", "is_peak_hour",
    "qoe_score_mean", "qoe_score_p10",
    "dl_throughput_mbps_mean",
    "latency_ms_mean",
    "call_drop_rate_mean",
    "degraded_session_rate_pct",
]


class SpikePredictor:
    """
    Trains and evaluates three forecasting models per region.
    Selects the best model based on MAE on the test period.
    Generates a 7-day forecast using the best model.
    """

    def __init__(self):
        self.models:    dict = {}   # region → best model artifact
        self.scores:    dict = {}   # region → {model: {mae, rmse, mape}}
        self.forecasts: dict = {}   # region → forecast DataFrame
        self.best_model_per_region: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, complaint_agg: pd.DataFrame,
            feature_matrix:  pd.DataFrame) -> dict:
        """
        Full spike prediction pipeline.

        Parameters
        ----------
        complaint_agg  : daily complaint aggregates (region × date)
        feature_matrix : joined feature matrix with lag/rolling features

        Returns
        -------
        dict with keys: scores, forecasts, best_models, summary
        """
        logger.info("=" * 55)
        logger.info("SPIKE PREDICTION")
        logger.info("=" * 55)

        regions = complaint_agg["region"].unique()
        logger.info(f"Training models for {len(regions)} regions × 3 model types\n")

        for region in sorted(regions):
            logger.info(f"  ── Region: {region} ──────────────────────────")
            region_scores = {}

            # Filter region data
            ts_df = complaint_agg[complaint_agg["region"] == region].sort_values("date").copy()
            fm_df = feature_matrix[feature_matrix["region"] == region].sort_values("date").copy()

            if len(ts_df) < 30:
                logger.warning(f"  {region}: insufficient data ({len(ts_df)} rows), skipping")
                continue

            # Train/test split (last FORECAST_HORIZON days = test)
            split = len(ts_df) - FORECAST_HORIZON
            train_ts, test_ts = ts_df.iloc[:split], ts_df.iloc[split:]

            # ── Model 1: ARIMA ────────────────────────────────────────────────
            try:
                arima_preds, arima_model = self._train_arima(train_ts, test_ts)
                region_scores["arima"] = _eval_metrics(
                    test_ts["total_complaints"].values, arima_preds
                )
                logger.info(
                    f"    ARIMA  — MAE: {region_scores['arima']['mae']:.2f}  "
                    f"MAPE: {region_scores['arima']['mape']:.1f}%"
                )
            except Exception as e:
                logger.warning(f"    ARIMA failed for {region}: {e}")
                arima_preds, arima_model = None, None

            # ── Model 2: Prophet ──────────────────────────────────────────────
            if PROPHET_AVAILABLE:
                try:
                    prophet_preds, prophet_model = self._train_prophet(train_ts, test_ts)
                    region_scores["prophet"] = _eval_metrics(
                        test_ts["total_complaints"].values, prophet_preds
                    )
                    logger.info(
                        f"    Prophet— MAE: {region_scores['prophet']['mae']:.2f}  "
                        f"MAPE: {region_scores['prophet']['mape']:.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"    Prophet failed for {region}: {e}")
                    prophet_preds, prophet_model = None, None
            else:
                prophet_preds, prophet_model = None, None

            # ── Model 3: XGBoost ──────────────────────────────────────────────
            try:
                feat_cols = [c for c in XGB_FEATURES if c in fm_df.columns]
                xgb_preds, xgb_model = self._train_xgboost(fm_df, feat_cols, split)
                region_scores["xgboost"] = _eval_metrics(
                    test_ts["total_complaints"].values, xgb_preds
                )
                logger.info(
                    f"    XGBoost— MAE: {region_scores['xgboost']['mae']:.2f}  "
                    f"MAPE: {region_scores['xgboost']['mape']:.1f}%"
                )
            except Exception as e:
                logger.warning(f"    XGBoost failed for {region}: {e}")
                xgb_preds, xgb_model = None, None

            # ── Select best model ─────────────────────────────────────────────
            available = {
                "arima":   (arima_preds,   arima_model),
                "prophet": (prophet_preds, prophet_model),
                "xgboost": (xgb_preds,     xgb_model),
            }
            best_name = min(
                {k: v for k, v in region_scores.items()},
                key=lambda k: region_scores[k]["mae"]
            )
            self.best_model_per_region[region] = best_name
            self.scores[region]                = region_scores
            self.models[region]                = {
                "best":    best_name,
                "model":   available[best_name][1],
                "all":     {k: v[1] for k, v in available.items() if v[1] is not None},
            }
            logger.info(f"    ✓ Best model: {best_name.upper()}")

            # ── 7-day forecast from best model ────────────────────────────────
            self.forecasts[region] = self._generate_forecast(
                best_name, ts_df, fm_df,
                available[best_name][1],
                feat_cols if "xgb" in best_name else []
            )

        # ── Save ──────────────────────────────────────────────────────────────
        self._save()
        forecast_df = self._build_forecast_dataframe()
        forecast_df.to_parquet(MODELS_DIR / "forecasts.parquet", index=False)

        summary = self._build_summary()
        logger.success(f"\nSpike prediction complete for {len(self.scores)} regions")
        _print_score_table(self.scores)

        return {
            "scores":      self.scores,
            "forecasts":   forecast_df,
            "best_models": self.best_model_per_region,
            "summary":     summary,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _train_arima(self, train: pd.DataFrame,
                     test:  pd.DataFrame) -> tuple[np.ndarray, object]:
        y_train = train["total_complaints"].values
        model   = SARIMAX(
            y_train,
            order=(2, 1, 2),
            seasonal_order=(1, 0, 1, 7),   # weekly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit     = model.fit(disp=False)
        preds   = fit.forecast(steps=len(test))
        return np.maximum(preds, 0), fit

    def _train_prophet(self, train: pd.DataFrame,
                       test:  pd.DataFrame) -> tuple[np.ndarray, object]:
        prophet_train = pd.DataFrame({
            "ds": pd.to_datetime(train["date"]),
            "y":  train["total_complaints"].values,
        })
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(prophet_train)

        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        preds = forecast.tail(len(test))["yhat"].values
        return np.maximum(preds, 0), model

    def _train_xgboost(self, fm_df: pd.DataFrame,
                       feat_cols: list[str],
                       split_idx: int) -> tuple[np.ndarray, XGBRegressor]:
        X = fm_df[feat_cols].fillna(0)
        y = fm_df["total_complaints"].fillna(0)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train         = y.iloc[:split_idx]

        model = XGBRegressor(
            n_estimators      = 300,
            max_depth         = 5,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbosity         = 0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y.iloc[split_idx:])],
                  verbose=False)
        preds = np.maximum(model.predict(X_test), 0)
        return preds, model

    def _generate_forecast(self, model_name: str,
                           ts_df:      pd.DataFrame,
                           fm_df:      pd.DataFrame,
                           model:      object,
                           feat_cols:  list[str]) -> pd.DataFrame:
        """Generate FORECAST_HORIZON-day ahead predictions."""
        last_date = pd.to_datetime(ts_df["date"]).max()
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=FORECAST_HORIZON, freq="D"
        )
        if model_name == "arima" and model is not None:
            preds = np.maximum(model.forecast(steps=FORECAST_HORIZON), 0)
        elif model_name == "prophet" and model is not None:
            fut = model.make_future_dataframe(periods=FORECAST_HORIZON)
            fc  = model.predict(fut)
            preds = np.maximum(fc.tail(FORECAST_HORIZON)["yhat"].values, 0)
        elif model_name == "xgboost" and model is not None and feat_cols:
            # Use last known feature row as proxy (simplification)
            last_features = fm_df[feat_cols].fillna(0).iloc[[-1]]
            preds = np.tile(
                np.maximum(model.predict(last_features)[0], 0),
                FORECAST_HORIZON
            )
        else:
            preds = np.full(FORECAST_HORIZON, ts_df["total_complaints"].tail(7).mean())

        return pd.DataFrame({
            "date":       future_dates,
            "forecast":   preds,
            "model_used": model_name,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_forecast_dataframe(self) -> pd.DataFrame:
        rows = []
        for region, fc_df in self.forecasts.items():
            fc_df = fc_df.copy()
            fc_df["region"] = region
            rows.append(fc_df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _build_summary(self) -> pd.DataFrame:
        rows = []
        for region, models in self.scores.items():
            for model_name, metrics in models.items():
                rows.append({
                    "region":     region,
                    "model":      model_name,
                    "mae":        metrics["mae"],
                    "rmse":       metrics["rmse"],
                    "mape":       metrics["mape"],
                    "is_best":    model_name == self.best_model_per_region.get(region),
                })
        return pd.DataFrame(rows).sort_values(["region", "mae"])

    def _save(self):
        joblib.dump(self.models,    MODELS_DIR / "all_models.pkl")
        joblib.dump(self.scores,    MODELS_DIR / "scores.pkl")
        joblib.dump(self.forecasts, MODELS_DIR / "forecasts_dict.pkl")
        logger.info(f"  Models saved → {MODELS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 2)}


def _print_score_table(scores: dict):
    logger.info("\n  Model Performance Summary (MAE):")
    logger.info(f"  {'Region':<15} {'ARIMA':>8} {'Prophet':>10} {'XGBoost':>10} {'Winner':>10}")
    logger.info("  " + "-" * 55)
    for region, s in sorted(scores.items()):
        arima_mae   = f"{s['arima']['mae']:.2f}"   if "arima"   in s else "  N/A"
        prophet_mae = f"{s['prophet']['mae']:.2f}" if "prophet" in s else "  N/A"
        xgb_mae     = f"{s['xgboost']['mae']:.2f}" if "xgboost" in s else "  N/A"
        winner      = min(s, key=lambda k: s[k]["mae"]) if s else "N/A"
        logger.info(f"  {region:<15} {arima_mae:>8} {prophet_mae:>10} {xgb_mae:>10} {winner:>10}")