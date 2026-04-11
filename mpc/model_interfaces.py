from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Literal

import numpy as np
import pandas as pd


Direction = Literal["heating", "cooling"]

MIN_ACTIVE_MINUTES = 5.0
MAX_ACTIVE_MINUTES = 240.0
MIN_DRIFT_MINUTES = 5.0
MAX_DRIFT_MINUTES = 480.0
DEFAULT_ACTIVE_MODEL_PATH = Path("models/active_time_xgb.pkl")
DEFAULT_DRIFT_MODEL_PATH = Path("models/drift_time_xgb.pkl")


def _hour_features(timestamp: pd.Timestamp) -> tuple[float, float]:
    angle = 2 * np.pi * timestamp.hour / 24.0
    return float(np.sin(angle)), float(np.cos(angle))


def _month_features(timestamp: pd.Timestamp) -> tuple[float, float]:
    angle = 2 * np.pi * timestamp.month / 12.0
    return float(np.sin(angle)), float(np.cos(angle))


def _as_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp if isinstance(timestamp, pd.Timestamp) else pd.Timestamp(timestamp)


def _clip_minutes(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(value):
        return upper
    return float(np.clip(value, lower, upper))


@dataclass(frozen=True)
class PredictorMetadata:
    active_rows: int
    drift_rows: int
    source_active: str
    source_drift: str


class FirstPassagePredictor:
    metadata: PredictorMetadata

    def predict_active_time(
        self,
        current_temp: float,
        target_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        system_running: bool,
        home_id: str | None = None,
        direction: Direction = "heating",
    ) -> float:
        raise NotImplementedError

    def predict_drift_time(
        self,
        current_temp: float,
        boundary_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        home_id: str | None = None,
        direction: Direction = "heating",
    ) -> float:
        raise NotImplementedError


@dataclass
class _GBMArtifact:
    artifact: dict
    source: Path

    @classmethod
    def load(cls, path: str | Path, expected_model_type: str) -> _GBMArtifact:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model artifact not found: {path}")
        with path.open("rb") as f:
            artifact = pickle.load(f)
        model_type = artifact.get("model_type")
        if model_type != expected_model_type:
            raise ValueError(f"expected {expected_model_type!r} in {path}, got {model_type!r}")
        return cls(artifact=artifact, source=path)

    @property
    def n_train_rows(self) -> int:
        return int(self.artifact.get("n_train_rows", 0))


class XGBFirstPassagePredictor(FirstPassagePredictor):
    def __init__(
        self,
        active_model: _GBMArtifact,
        drift_model: _GBMArtifact,
    ) -> None:
        self.active_model = active_model
        self.drift_model = drift_model
        self.metadata = PredictorMetadata(
            active_rows=active_model.n_train_rows,
            drift_rows=drift_model.n_train_rows,
            source_active=str(active_model.source),
            source_drift=str(drift_model.source),
        )

    @classmethod
    def from_model_files(
        cls,
        active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
        drift_model_path: str | Path = DEFAULT_DRIFT_MODEL_PATH,
    ) -> XGBFirstPassagePredictor:
        return cls(
            active_model=_GBMArtifact.load(
                active_model_path,
                expected_model_type="hybrid_gbm_active_time_to_target",
            ),
            drift_model=_GBMArtifact.load(
                drift_model_path,
                expected_model_type="gbm_drift_time_to_boundary",
            ),
        )

    def predict_active_time(
        self,
        current_temp: float,
        target_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        system_running: bool,
        home_id: str | None = None,
        direction: Direction = "heating",
    ) -> float:
        gap = _active_gap(current_temp, target_temp, direction)
        if gap <= 1e-6:
            return 0.0

        row = _active_feature_row(
            current_temp=current_temp,
            target_temp=target_temp,
            outdoor_temp=outdoor_temp,
            timestamp=_as_timestamp(timestamp),
            system_running=system_running,
            direction=direction,
        )
        log_pred = _predict_log_minutes(self.active_model.artifact, row)

        mode = 1 if direction == "heating" else 0
        if home_id is not None:
            home_mode_key = (home_id, mode)
            home_mode_residuals = self.active_model.artifact.get("home_mode_residuals", {})
            home_residuals = self.active_model.artifact.get("home_residuals", {})
            if home_mode_key in home_mode_residuals:
                log_pred += float(home_mode_residuals[home_mode_key])
            elif home_id in home_residuals:
                log_pred += float(home_residuals[home_id])

        return _artifact_minutes(self.active_model.artifact, log_pred, MIN_ACTIVE_MINUTES, MAX_ACTIVE_MINUTES)

    def predict_drift_time(
        self,
        current_temp: float,
        boundary_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        home_id: str | None = None,
        direction: Direction = "heating",
    ) -> float:
        margin = _drift_margin(current_temp, boundary_temp, direction)
        if margin <= 1e-6:
            return 0.0

        row = _drift_feature_row(
            current_temp=current_temp,
            boundary_temp=boundary_temp,
            outdoor_temp=outdoor_temp,
            timestamp=_as_timestamp(timestamp),
            direction=direction,
        )
        log_pred = _predict_log_minutes(self.drift_model.artifact, row)
        return _artifact_minutes(self.drift_model.artifact, log_pred, MIN_DRIFT_MINUTES, MAX_DRIFT_MINUTES)


def _predict_log_minutes(artifact: dict, row: dict[str, float]) -> float:
    feature_cols = artifact["feature_cols"]
    x = pd.DataFrame([row], columns=feature_cols).fillna(0).values
    return float(artifact["model"].predict(x)[0])


def _artifact_minutes(artifact: dict, log_minutes: float, default_min: float, default_max: float) -> float:
    min_minutes = float(artifact.get("min_duration", default_min))
    max_minutes = float(artifact.get("max_duration", default_max))
    return _clip_minutes(float(np.exp(log_minutes)), min_minutes, max_minutes)


def _active_gap(current_temp: float, target_temp: float, direction: Direction) -> float:
    if direction == "heating":
        return max(target_temp - current_temp, 0.0)
    return max(current_temp - target_temp, 0.0)


def _drift_margin(current_temp: float, boundary_temp: float, direction: Direction) -> float:
    if direction == "heating":
        return max(current_temp - boundary_temp, 0.0)
    return max(boundary_temp - current_temp, 0.0)


def _active_feature_row(
    current_temp: float,
    target_temp: float,
    outdoor_temp: float,
    timestamp: pd.Timestamp,
    system_running: bool,
    direction: Direction,
) -> dict[str, float]:
    hour_sin, hour_cos = _hour_features(timestamp)
    month_sin, month_cos = _month_features(timestamp)
    gap = _active_gap(current_temp, target_temp, direction)
    thermal_drive = outdoor_temp - current_temp
    signed_thermal_drive = thermal_drive if direction == "heating" else -thermal_drive
    return {
        "log_gap": float(np.log(gap + 0.1)),
        "abs_gap": float(gap),
        "is_heating": 1.0 if direction == "heating" else 0.0,
        "system_running": 1.0 if system_running else 0.0,
        "signed_thermal_drive": float(signed_thermal_drive),
        "outdoor_temp": float(outdoor_temp),
        "start_temp": float(current_temp),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "hour": float(timestamp.hour),
        "month": float(timestamp.month),
        "day_of_week": float(timestamp.weekday()),
        "indoor_humidity": 50.0,
        "outdoor_humidity": 50.0,
    }


def _drift_feature_row(
    current_temp: float,
    boundary_temp: float,
    outdoor_temp: float,
    timestamp: pd.Timestamp,
    direction: Direction,
) -> dict[str, float]:
    hour_sin, hour_cos = _hour_features(timestamp)
    month_sin, month_cos = _month_features(timestamp)
    margin = _drift_margin(current_temp, boundary_temp, direction)
    thermal_drive = outdoor_temp - current_temp
    signed_thermal_drive = -thermal_drive if direction == "heating" else thermal_drive
    return {
        "margin": float(margin),
        "log_margin": float(np.log(margin + 0.1)),
        "is_heating": 1.0 if direction == "heating" else 0.0,
        "signed_thermal_drive": float(signed_thermal_drive),
        "outdoor_temp": float(outdoor_temp),
        "start_temp": float(current_temp),
        "boundary_temp": float(boundary_temp),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "hour": float(timestamp.hour),
        "month": float(timestamp.month),
        "day_of_week": float(timestamp.weekday()),
    }
