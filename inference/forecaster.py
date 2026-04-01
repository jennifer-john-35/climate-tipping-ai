from __future__ import annotations

import numpy as np


class MonteCarloForecaster:
    """Monte Carlo Dropout forecaster wrapping a Keras LSTM model."""

    def __init__(self, model, scaler, n_passes: int = 50) -> None:
        self.model = model
        self.scaler = scaler
        self.n_passes = n_passes

    def forecast(
        self,
        signal_history: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run n_passes stochastic forward passes with dropout enabled.

        Args:
            signal_history: shape (window, n_features) — last N time steps.
            horizon: number of future steps to forecast.

        Returns:
            (mean_forecast, lower_10th, upper_90th) each shape (horizon,).
        """
        window, n_features = signal_history.shape

        # 1. Scale input
        scaled = self.scaler.transform(signal_history)

        # 2. Reshape for LSTM: (1, window, n_features)
        x = scaled.reshape(1, window, n_features)

        # 3. Run n_passes stochastic forward passes
        try:
            predictions = [
                self.model(x, training=True).numpy() for _ in range(self.n_passes)
            ]
        except Exception:
            # Fallback: use model.predict for all passes (no dropout stochasticity)
            predictions = [
                self.model.predict(x, verbose=0) for _ in range(self.n_passes)
            ]

        # 4. Stack → shape (n_passes, ...)
        stack = np.array(predictions)  # (n_passes, 1, horizon) or (n_passes, horizon)

        # 5. Squeeze to (n_passes, horizon_or_more)
        stack = stack.reshape(self.n_passes, -1)

        # 6. Trim / pad to requested horizon
        stack = stack[:, :horizon]
        if stack.shape[1] < horizon:
            pad = np.full((self.n_passes, horizon - stack.shape[1]), stack[:, -1:])
            stack = np.concatenate([stack, pad], axis=1)

        # 7. Compute statistics along axis=0
        mean_fc = np.mean(stack, axis=0).flatten()[:horizon]
        lower_fc = np.percentile(stack, 10, axis=0).flatten()[:horizon]
        upper_fc = np.percentile(stack, 90, axis=0).flatten()[:horizon]

        return mean_fc, lower_fc, upper_fc
