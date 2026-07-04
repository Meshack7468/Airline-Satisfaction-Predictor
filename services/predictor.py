"""
This file loads the tuned Xgboost model , prepares incoming
data the same way it was prepared during training, and returns a
prediction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger("airline_satisfaction.predictor")

# These are the category values the model was trained on.
CATEGORY_MAPPINGS: dict[str, dict[str, int]] = {
    "Gender": {
        "Female": 0,
        "Male": 1,
    },
    "Customer Type": {
        "Loyal Customer": 0,
        "disloyal Customer": 1,
    },
    "Type of Travel": {
        "Business travel": 0,
        "Personal Travel": 1,
    },
    "Class": {
        "Business": 0,
        "Eco": 1,
        "Eco Plus": 2,
    },
}


NORMALIZED_VALUES: dict[str, dict[str, str]] = {
    "Customer Type": {
        "Disloyal Customer": "disloyal Customer",
    },
}

# Scaled columns during training.
SCALE_COLUMNS: tuple[str, ...] = (
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
)

# The exact column order the model saw during training 
# (the "satisfaction" target column is left out, since that's not an input
# feature). feature_order.json must match this list exactly
EXPECTED_FEATURE_ORDER: tuple[str, ...] = (
    "Gender",
    "Customer Type",
    "Age",
    "Type of Travel",
    "Class",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
)


# Turns the model's 0/1 output into a readable label.
CLASS_LABELS: dict[int, str] = {
    0: "Neutral or Dissatisfied",
    1: "Satisfied",
}


class ArtifactLoadError(RuntimeError):
    """Error Raised when the Tuned-Xgboost cannot be loaded."""


class UnknownCategoryError(ValueError):
    """Error Raised when a categorical field contains a value unseen during training."""

    def __init__(self, field: str, value: Any, known_values: list[str]) -> None:
        self.field = field
        self.value = value
        self.known_values = known_values
        super().__init__(
            f"Unknown value {value!r} for field '{field}'. "
            f"Expected one of: {known_values}."
        )


class PredictionFailedError(RuntimeError):
    """Error Raised when the model fails to produce a prediction."""


class PredictorService:
    """
    Loads the model once and serves predictions.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self._artifacts_dir = artifacts_dir
        self._model: Any = None
        self._scaler: Any = None
        self._feature_order: list[str] = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check Whether all artifacts have been successfully loaded."""
        return self._loaded

    def load(self) -> None:
        """
        Load the model, scaler, and feature order from as in the notebook.

        """
        model_path = self._artifacts_dir / "model.joblib"
        scaler_path = self._artifacts_dir / "scaler.joblib"
        feature_order_path = self._artifacts_dir / "feature_order.json"

        for path in (model_path, scaler_path, feature_order_path):
            if not path.exists():
                logger.error("Missing required artifact: %s", path)
                raise ArtifactLoadError(f"Required artifact not found: {path}")

        try:
            # Load the tuned xgboost model.
            logger.info("Loading model from %s", model_path)
            self._model = joblib.load(model_path)

            # Load the scaler used for the numeric columns.
            logger.info("Loading scaler from %s", scaler_path)
            self._scaler = joblib.load(scaler_path)

            # Load the column order the model expects.
            logger.info("Loading feature order from %s", feature_order_path)
            with feature_order_path.open("r", encoding="utf-8") as f:
                self._feature_order = json.load(f)

        except Exception as exc:  
            logger.exception("Failed to load model artifacts")
            raise ArtifactLoadError(
                f"Failed to load one or more model artifacts: {exc}"
            ) from exc

        self._validate_feature_order()
        self._loaded = True
        logger.info("All model artifacts loaded successfully")

    def _validate_feature_order(self) -> None:
        """Check that feature_order.json matches the training column order exactly."""
        if not isinstance(self._feature_order, list) or not self._feature_order:
            raise ArtifactLoadError(
                "feature_order.json must contain a non-empty list of feature names"
            )
        if not all(isinstance(name, str) for name in self._feature_order):
            raise ArtifactLoadError("feature_order.json must contain only strings")

        if tuple(self._feature_order) != EXPECTED_FEATURE_ORDER:
            logger.error(
                "feature_order.json does not match the expected training order. "
                "Expected: %s | Got: %s",
                list(EXPECTED_FEATURE_ORDER),
                self._feature_order,
            )
            raise ArtifactLoadError(
                "feature_order.json does not match the exact column order the "
                "model was trained on. Expected: "
                f"{list(EXPECTED_FEATURE_ORDER)}"
            )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise ArtifactLoadError(
                "Predictor artifacts are not loaded. Call load() at startup "
                "before serving predictions."
            )

    def _encode_categoricals(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Turn category text into the numbers the model expects.

        """
        encoded = dict(raw)

        for field, mapping in CATEGORY_MAPPINGS.items():
            raw_value = raw.get(field)

            # Convert frontend values to match the training data.
            field_fixes = NORMALIZED_VALUES.get(field, {})
            normalized_value = field_fixes.get(raw_value, raw_value)

            # Check that the category exists.
            if normalized_value not in mapping:
                raise UnknownCategoryError(field, raw_value, list(mapping.keys()))

            
            encoded[field] = mapping[normalized_value]

        return encoded

    def _build_feature_frame(self, encoded: dict[str, Any]) -> pd.DataFrame:
        """Build the DataFrame in the same order used during training."""
        missing = [name for name in self._feature_order if name not in encoded]
        if missing:
            raise PredictionFailedError(
                f"Encoded payload is missing expected features: {missing}"
            )

        row = {name: encoded[name] for name in self._feature_order}
        return pd.DataFrame([row], columns=self._feature_order)

    def _scale_numeric_columns(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Scale only the three numeric columns. Everything else stays the same."""
        scaled_df = input_df.copy()
        scale_subset = list(SCALE_COLUMNS)
        scaled_df[scale_subset] = self._scaler.transform(scaled_df[scale_subset])
        return scaled_df

    def predict(self, raw: dict[str, Any]) -> str:
        
        self._ensure_loaded()

        try:
            encoded = self._encode_categoricals(raw)
            feature_frame = self._build_feature_frame(encoded)
            scaled_frame = self._scale_numeric_columns(feature_frame)
        except UnknownCategoryError:
            raise
        except Exception as exc:  
            logger.exception("Preprocessing failed for payload: %s", raw)
            raise PredictionFailedError(f"Preprocessing failed: {exc}") from exc

        try:

            # Run the prediction.
            raw_prediction = self._model.predict(scaled_frame)
            predicted_class = int(pd.Series(raw_prediction).iloc[0])
        except Exception as exc:  
            logger.exception("Model inference failed for payload: %s", raw)
            raise PredictionFailedError(f"Model inference failed: {exc}") from exc

        label = CLASS_LABELS.get(predicted_class)
        if label is None:
            logger.error("Model returned unexpected class value: %s", predicted_class)
            raise PredictionFailedError(
                f"Model returned an unrecognized class value: {predicted_class}"
            )

        logger.info("Prediction successful: class=%s label=%s", predicted_class, label)
        return label
