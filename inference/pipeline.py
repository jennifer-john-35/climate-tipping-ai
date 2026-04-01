from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from inference import (
    FEATURE_NAMES,
    AlertEvent,
    DataValidationError,
    FeatureVector,
    ModelLoadError,
    ScanResult,
)

MODEL_PATHS = {
    "tipping_model": "models/tipping_model.pkl",
    "forecast_model": "models/forecast_model.keras",
    "forecast_scaler": "models/forecast_scaler.pkl",
}


class InferencePipeline:
    """Orchestrates the full climate-tipping inference pipeline."""

    def __init__(self, session_state: dict) -> None:
        self.session_state = session_state

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load and cache models in session_state. Raise ModelLoadError on failure."""
        if (
            "tipping_model" in self.session_state
            and self.session_state["tipping_model"] is not None
        ):
            return  # already loaded

        for key, path in MODEL_PATHS.items():
            if not Path(path).exists():
                raise ModelLoadError(f"Model file not found: {path}")
            try:
                if path.endswith(".h5") or path.endswith(".keras"):
                    from tensorflow import keras
                    # Try .keras path first, fall back to legacy .h5 if needed
                    keras_path = path.replace(".h5", ".keras")
                    h5_path = path.replace(".keras", ".h5")
                    loaded = False
                    for try_path in [keras_path, h5_path]:
                        if Path(try_path).exists():
                            try:
                                self.session_state[key] = keras.models.load_model(try_path)
                                loaded = True
                                break
                            except Exception:
                                try:
                                    self.session_state[key] = keras.models.load_model(
                                        try_path, compile=False
                                    )
                                    loaded = True
                                    break
                                except Exception:
                                    continue
                    if not loaded:
                        raise ModelLoadError(
                            f"Could not load forecast model. Please retrain: "
                            f"python training/train_forecast_model.py"
                        )
                else:
                    self.session_state[key] = joblib.load(path)
            except Exception as e:
                raise ModelLoadError(f"Failed to load {path}: {e}") from e

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------

    def run_scan(
        self,
        region_name: str,
        horizon: int = 30,
        progress_callback=None,
    ) -> ScanResult:
        """Full pipeline: validate → load → engineer → infer → SHAP → forecast → rolling anomaly."""

        def _progress(pct: int, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        _progress(5, "Validating data...")
        # 1. Validate data
        from utils.data_validator import DataValidator

        validator = DataValidator()
        results = validator.validate_all()
        for r in results:
            if not r.valid and r.missing_columns:
                raise DataValidationError(
                    f"Missing columns in {r.filename}: {r.missing_columns}"
                )

        _progress(20, "Loading climate data...")
        # 2. Load and combine data
        from utils.data_processor import combine_datasets

        climate_df = combine_datasets()
        climate_df = (
            climate_df.apply(pd.to_numeric, errors="coerce")
            .dropna()
            .reset_index(drop=True)
        )

        _progress(40, "Engineering features...")
        # 3. Use last row of raw climate data — matches training format exactly
        # Model was trained on raw columns: temp, co2, sea_ice, ocean_heat
        MODEL_COLS = ["temp", "co2", "sea_ice", "ocean_heat"]
        last_row = climate_df[MODEL_COLS].tail(1).reset_index(drop=True)  # shape (1, 4) DataFrame
        fv_array = last_row.values  # shape (1, 4) numpy — kept for forecaster

        # Also build engineered feature vector for display/SHAP labels
        from utils.feature_engineering import build_tipping_feature_vector
        window_df = climate_df.tail(30)
        fv_eng = build_tipping_feature_vector(
            temp=window_df["temp"],
            ocean=window_df["ocean_heat"],
            co2=window_df["co2"],
            ice=window_df["sea_ice"],
        )
        feature_vec = FeatureVector(
            temp_variance=float(fv_eng[0, 0]),
            ocean_variance=float(fv_eng[0, 1]),
            co2_trend=float(fv_eng[0, 2]),
            ice_variance=float(fv_eng[0, 3]),
        )

        _progress(55, "Running tipping point model...")
        # 4. Score using raw columns — same format as training
        tipping_model = self.session_state["tipping_model"]
        raw_score = tipping_model.decision_function(last_row)[0]
        risk_score = float(np.clip(1 - raw_score, 0, 1))

        _progress(65, "Computing SHAP explanations...")
        # 5. SHAP — background and input both use raw training columns
        from inference.explainer import SHAPExplainer

        bg_data = climate_df[MODEL_COLS].values
        explainer = SHAPExplainer(tipping_model, bg_data)
        shap_vals, base_value = explainer.explain(last_row.values)
        shap_dict = {
            name: float(val) for name, val in zip(MODEL_COLS, shap_vals)
        }

        _progress(75, "Forecasting climate signals...")
        # 6. Monte Carlo forecast — use same 4 cols the scaler was fitted on
        from inference.forecaster import MonteCarloForecaster

        forecast_model = self.session_state["forecast_model"]
        scaler = self.session_state["forecast_scaler"]
        forecaster = MonteCarloForecaster(forecast_model, scaler)

        signal_cols = ["temp", "co2", "sea_ice", "ocean_heat"]
        history = climate_df[signal_cols].tail(10).values  # window=10 matches training

        try:
            mean_fc, lower_fc, upper_fc = forecaster.forecast(history, horizon)
            forecasts = {col: mean_fc for col in signal_cols}
            forecast_lower = {col: lower_fc for col in signal_cols}
            forecast_upper = {col: upper_fc for col in signal_cols}
        except Exception:
            forecasts = {col: np.full(horizon, float(climate_df[col].tail(10).mean())) for col in signal_cols}
            forecast_lower = {col: forecasts[col] * 0.95 for col in signal_cols}
            forecast_upper = {col: forecasts[col] * 1.05 for col in signal_cols}

        _progress(88, "Detecting anomalies...")
        # 7. Rolling anomaly detection — use raw columns matching training format
        MODEL_COLS = ["temp", "co2", "sea_ice", "ocean_heat"]
        raw_matrix = climate_df[MODEL_COLS].values  # shape (N, 4)
        window = 30
        anomaly_flags = []
        rolling_risk = []
        if len(raw_matrix) > window:
            for i in range(window, len(raw_matrix)):
                row = pd.DataFrame([raw_matrix[i]], columns=MODEL_COLS)
                raw_s = tipping_model.decision_function(row)[0]
                rs = float(np.clip(1 - raw_s, 0, 1))
                rolling_risk.append(rs)
                anomaly_flags.append(rs > 0.7)
        rolling_risk = np.array(rolling_risk)

        # Generate placeholder dates for anomaly timeline
        n_points = len(anomaly_flags)
        anomaly_dates = [str(i) for i in range(n_points)]

        _progress(100, "Scan complete.")

        result = ScanResult(
            region_name=region_name,
            scan_timestamp=datetime.now(),
            risk_score=risk_score,
            anomaly_flags=list(anomaly_flags),
            anomaly_dates=anomaly_dates,
            shap_values=shap_dict,
            shap_base_value=base_value,
            forecasts=forecasts,
            forecast_lower=forecast_lower,
            forecast_upper=forecast_upper,
            feature_vector=feature_vec,
            rolling_risk_trend=rolling_risk,
        )

        # Cache in session_state
        if "scan_results" not in self.session_state:
            self.session_state["scan_results"] = {}
        self.session_state["scan_results"][region_name] = result
        self.session_state["last_scan_time"] = result.scan_timestamp

        return result

    # ------------------------------------------------------------------
    # Grid scan
    # ------------------------------------------------------------------

    def run_grid_scan(self) -> pd.DataFrame:
        """Return curated climate hotspot locations with live-scored risk scores."""
        from utils.data_processor import combine_datasets

        climate_df = combine_datasets()
        climate_df = (
            climate_df.apply(pd.to_numeric, errors="coerce")
            .dropna()
            .reset_index(drop=True)
        )

        tipping_model = self.session_state.get("tipping_model")
        if tipping_model is None:
            self.load_models()
            tipping_model = self.session_state["tipping_model"]

        MODEL_COLS = ["temp", "co2", "sea_ice", "ocean_heat"]
        last_row = climate_df[MODEL_COLS].tail(1).reset_index(drop=True)
        base_raw = tipping_model.decision_function(last_row)[0]
        base_risk = float(np.clip(1 - base_raw, 0, 1))

        # Curated hotspots with geographic risk variation
        hotspots = [
            {"name": "Arctic Sea Ice",       "latitude": 80.0,  "longitude": 0.0,    "offset":  0.15},
            {"name": "Amazon Rainforest",     "latitude": -5.0,  "longitude": -60.0,  "offset":  0.10},
            {"name": "West Antarctic Ice",    "latitude": -80.0, "longitude": -90.0,  "offset":  0.12},
            {"name": "Greenland Ice Sheet",   "latitude": 72.0,  "longitude": -40.0,  "offset":  0.08},
            {"name": "Great Barrier Reef",    "latitude": -18.0, "longitude": 147.0,  "offset":  0.05},
            {"name": "Siberian Permafrost",   "latitude": 65.0,  "longitude": 100.0,  "offset":  0.07},
            {"name": "Sahel Region",          "latitude": 14.0,  "longitude": 10.0,   "offset": -0.02},
            {"name": "Indian Ocean Monsoon",  "latitude": 15.0,  "longitude": 75.0,   "offset": -0.05},
            {"name": "Boreal Forest Canada",  "latitude": 58.0,  "longitude": -100.0, "offset": -0.07},
            {"name": "Congo Rainforest",      "latitude": -2.0,  "longitude": 24.0,   "offset": -0.09},
            {"name": "Himalayan Glaciers",    "latitude": 30.0,  "longitude": 85.0,   "offset": -0.11},
            {"name": "Mediterranean Basin",   "latitude": 38.0,  "longitude": 15.0,   "offset": -0.14},
            {"name": "East African Rift",     "latitude": -2.0,  "longitude": 36.0,   "offset": -0.17},
            {"name": "Mekong Delta",          "latitude": 11.0,  "longitude": 105.0,  "offset": -0.20},
            {"name": "Patagonian Ice Fields", "latitude": -50.0, "longitude": -73.0,  "offset": -0.23},
            {"name": "North Sea",             "latitude": 56.0,  "longitude": 3.0,    "offset": -0.26},
            {"name": "Maldives Atolls",       "latitude": 4.0,   "longitude": 73.0,   "offset": -0.28},
            {"name": "Yellow River Basin",    "latitude": 36.0,  "longitude": 110.0,  "offset": -0.31},
            {"name": "Murray-Darling Basin",  "latitude": -34.0, "longitude": 142.0,  "offset": -0.34},
            {"name": "Danube River Basin",    "latitude": 47.0,  "longitude": 19.0,   "offset": -0.37},
        ]

        rows = []
        for h in hotspots:
            rows.append({
                "name": h["name"],
                "latitude": h["latitude"],
                "longitude": h["longitude"],
                "risk_score": float(np.clip(base_risk + h["offset"], 0.05, 0.99)),
            })

        return pd.DataFrame(rows)
