from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def _hour_features(timestamp: pd.Timestamp) -> tuple[float, float]:
    angle = 2 * np.pi * timestamp.hour / 24.0
    return float(np.sin(angle)), float(np.cos(angle))


def _month_features(timestamp: pd.Timestamp) -> tuple[float, float]:
    angle = 2 * np.pi * timestamp.month / 12.0
    return float(np.sin(angle)), float(np.cos(angle))


@dataclass(frozen=True)
class PredictorMetadata:
    active_rows: int
    drift_rows: int
    source_active: str
    source_drift: str


class FirstPassagePredictor:
    def predict_heat_time(
        self,
        current_temp: float,
        target_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        system_running: bool,
        home_id: str | None = None,
    ) -> float:
        raise NotImplementedError

    def predict_drift_time(
        self,
        current_temp: float,
        boundary_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        home_id: str | None = None,
    ) -> float:
        raise NotImplementedError


@dataclass
class _RidgeModel:
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    bias: float
    min_target: float
    max_target: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        alpha: float = 1.0,
    ) -> _RidgeModel:
        means = features.mean(axis=0)
        scales = features.std(axis=0)
        scales[scales < 1e-6] = 1.0
        x = (features - means) / scales
        reg = alpha * np.eye(x.shape[1], dtype=float)
        weights = np.linalg.solve(x.T @ x + reg, x.T @ targets)
        bias = float(targets.mean())
        return cls(
            means=means,
            scales=scales,
            weights=weights,
            bias=bias,
            min_target=float(targets.min()),
            max_target=float(targets.max()),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (features - self.means) / self.scales
        pred = x @ self.weights + self.bias
        return np.clip(pred, self.min_target, self.max_target)


class PhysicsFallbackPredictor(FirstPassagePredictor):
    def __init__(
        self,
        heating_rate_f_per_hour: float = 2.0,
        passive_loss_rate_per_hour: float = 0.08,
        min_minutes: float = 5.0,
    ) -> None:
        self.heating_rate_f_per_hour = heating_rate_f_per_hour
        self.passive_loss_rate_per_hour = passive_loss_rate_per_hour
        self.min_minutes = min_minutes

    def predict_heat_time(
        self,
        current_temp: float,
        target_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        system_running: bool,
        home_id: str | None = None,
    ) -> float:
        gap = max(target_temp - current_temp, 0.0)
        if gap <= 1e-6:
            return self.min_minutes
        thermal_penalty = max(current_temp - outdoor_temp, 0.0) / 40.0
        running_bonus = 0.85 if system_running else 1.0
        effective_rate = max(self.heating_rate_f_per_hour * (1.0 - 0.35 * thermal_penalty), 0.4)
        return max(gap / effective_rate * 60.0 * running_bonus, self.min_minutes)

    def predict_drift_time(
        self,
        current_temp: float,
        boundary_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        home_id: str | None = None,
    ) -> float:
        margin = max(current_temp - boundary_temp, 0.0)
        if margin <= 1e-6:
            return self.min_minutes
        drive = max(current_temp - outdoor_temp, 1.0)
        effective_rate = max(self.passive_loss_rate_per_hour * drive, 0.05)
        return max(margin / effective_rate * 60.0, self.min_minutes)


class FittedFirstPassagePredictor(FirstPassagePredictor):
    def __init__(
        self,
        active_model: _RidgeModel | None,
        drift_model: _RidgeModel | None,
        fallback: FirstPassagePredictor | None = None,
        metadata: PredictorMetadata | None = None,
    ) -> None:
        self.active_model = active_model
        self.drift_model = drift_model
        self.fallback = fallback or PhysicsFallbackPredictor()
        self.metadata = metadata or PredictorMetadata(0, 0, "fallback", "fallback")

    @classmethod
    def from_local_data(
        cls,
        active_path: str | Path = "data/setpoint_responses.parquet",
        drift_path: str | Path = "data/drift_episodes.parquet",
        max_active_rows: int = 80_000,
        max_drift_rows: int = 80_000,
        random_seed: int = 42,
    ) -> FittedFirstPassagePredictor:
        active_model = None
        drift_model = None
        active_rows = 0
        drift_rows = 0
        active_source = "fallback"
        drift_source = "fallback"

        active_path = Path(active_path)
        drift_path = Path(drift_path)

        if active_path.exists():
            active_df = pd.read_parquet(active_path)
            active_samples = _prepare_active_training_rows(active_df, max_active_rows, random_seed)
            if len(active_samples) > 0:
                x_active = np.vstack(active_samples["features"].to_numpy())
                y_active = np.log(active_samples["duration_min"].to_numpy())
                active_model = _RidgeModel.fit(x_active, y_active, alpha=2.0)
                active_rows = len(active_samples)
                active_source = str(active_path)

        if drift_path.exists():
            drift_df = pd.read_parquet(drift_path)
            drift_samples = _prepare_drift_training_rows(drift_df, max_drift_rows, random_seed)
            if len(drift_samples) > 0:
                x_drift = np.vstack(drift_samples["features"].to_numpy())
                y_drift = np.log(drift_samples["duration_min"].to_numpy())
                drift_model = _RidgeModel.fit(x_drift, y_drift, alpha=2.0)
                drift_rows = len(drift_samples)
                drift_source = str(drift_path)

        metadata = PredictorMetadata(
            active_rows=active_rows,
            drift_rows=drift_rows,
            source_active=active_source,
            source_drift=drift_source,
        )
        return cls(active_model=active_model, drift_model=drift_model, metadata=metadata)

    def predict_heat_time(
        self,
        current_temp: float,
        target_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        system_running: bool,
        home_id: str | None = None,
    ) -> float:
        if target_temp <= current_temp + 1e-6 or self.active_model is None:
            return self.fallback.predict_heat_time(
                current_temp=current_temp,
                target_temp=target_temp,
                outdoor_temp=outdoor_temp,
                timestamp=timestamp,
                system_running=system_running,
                home_id=home_id,
            )
        features = _active_feature_vector(
            current_temp=current_temp,
            target_temp=target_temp,
            outdoor_temp=outdoor_temp,
            timestamp=timestamp,
            system_running=system_running,
        )
        pred_log_minutes = float(self.active_model.predict(features[None, :])[0])
        pred_minutes = float(np.exp(pred_log_minutes))
        fallback_minutes = self.fallback.predict_heat_time(
            current_temp=current_temp,
            target_temp=target_temp,
            outdoor_temp=outdoor_temp,
            timestamp=timestamp,
            system_running=system_running,
            home_id=home_id,
        )
        return float(np.clip(pred_minutes, 5.0, max(fallback_minutes * 3.0, 15.0)))

    def predict_drift_time(
        self,
        current_temp: float,
        boundary_temp: float,
        outdoor_temp: float,
        timestamp: pd.Timestamp,
        home_id: str | None = None,
    ) -> float:
        if current_temp <= boundary_temp + 1e-6 or self.drift_model is None:
            return self.fallback.predict_drift_time(
                current_temp=current_temp,
                boundary_temp=boundary_temp,
                outdoor_temp=outdoor_temp,
                timestamp=timestamp,
                home_id=home_id,
            )
        features = _drift_feature_vector(
            current_temp=current_temp,
            boundary_temp=boundary_temp,
            outdoor_temp=outdoor_temp,
            timestamp=timestamp,
        )
        pred_log_minutes = float(self.drift_model.predict(features[None, :])[0])
        pred_minutes = float(np.exp(pred_log_minutes))
        fallback_minutes = self.fallback.predict_drift_time(
            current_temp=current_temp,
            boundary_temp=boundary_temp,
            outdoor_temp=outdoor_temp,
            timestamp=timestamp,
            home_id=home_id,
        )
        return float(np.clip(pred_minutes, 5.0, max(fallback_minutes * 3.0, 15.0)))


def _prepare_active_training_rows(
    df: pd.DataFrame,
    max_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    durations = (df.groupby("episode_id")["timestep_idx"].max() + 1) * 5
    grouped = (
        df.sort_values(["episode_id", "timestep_idx"])
        .groupby("episode_id", sort=False)
        .first()
        .reset_index()
    )
    grouped = grouped[grouped["change_type"] == "heat_increase"].copy()
    grouped["duration_min"] = grouped["episode_id"].map(durations)
    grouped = grouped[grouped["duration_min"].between(5, 240)]
    grouped["system_running"] = (
        grouped["HeatingEquipmentStage1_RunTime"].fillna(0).gt(0)
        | grouped["CoolingEquipmentStage1_RunTime"].fillna(0).gt(0)
    ).astype(float)
    grouped["timestamp"] = pd.to_datetime(grouped["timestamp"])
    grouped = grouped.dropna(
        subset=["Indoor_AverageTemperature", "target_setpoint", "Outdoor_Temperature", "timestamp"]
    )
    if len(grouped) > max_rows:
        grouped = grouped.sample(n=max_rows, random_state=random_seed)
    grouped["features"] = grouped.apply(
        lambda row: _active_feature_vector(
            current_temp=float(row["Indoor_AverageTemperature"]),
            target_temp=float(row["target_setpoint"]),
            outdoor_temp=float(row["Outdoor_Temperature"]),
            timestamp=pd.Timestamp(row["timestamp"]),
            system_running=bool(row["system_running"]),
        ),
        axis=1,
    )
    return grouped[["features", "duration_min"]]


def _prepare_drift_training_rows(
    df: pd.DataFrame,
    max_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    grouped = (
        df[df["timestep_idx"] == 0]
        .copy()
    )
    grouped = grouped[
        (grouped["drift_direction"] == "warming_drift")
        & (grouped["crossed_boundary"] == True)
        & grouped["time_to_boundary_min"].between(5, 480)
    ].copy()
    grouped["timestamp"] = pd.to_datetime(grouped["timestamp"])
    grouped = grouped.dropna(
        subset=["start_temp", "boundary_temp", "Outdoor_Temperature", "timestamp", "time_to_boundary_min"]
    )
    if len(grouped) > max_rows:
        grouped = grouped.sample(n=max_rows, random_state=random_seed)
    grouped["features"] = grouped.apply(
        lambda row: _drift_feature_vector(
            current_temp=float(row["start_temp"]),
            boundary_temp=float(row["boundary_temp"]),
            outdoor_temp=float(row["Outdoor_Temperature"]),
            timestamp=pd.Timestamp(row["timestamp"]),
        ),
        axis=1,
    )
    grouped = grouped.rename(columns={"time_to_boundary_min": "duration_min"})
    return grouped[["features", "duration_min"]]


def _active_feature_vector(
    current_temp: float,
    target_temp: float,
    outdoor_temp: float,
    timestamp: pd.Timestamp,
    system_running: bool,
) -> np.ndarray:
    hour_sin, hour_cos = _hour_features(timestamp)
    month_sin, month_cos = _month_features(timestamp)
    gap = max(target_temp - current_temp, 0.0)
    thermal_drive = outdoor_temp - current_temp
    return np.array(
        [
            current_temp,
            target_temp,
            outdoor_temp,
            gap,
            np.log(gap + 0.1),
            thermal_drive,
            hour_sin,
            hour_cos,
            month_sin,
            month_cos,
            1.0 if system_running else 0.0,
        ],
        dtype=float,
    )


def _drift_feature_vector(
    current_temp: float,
    boundary_temp: float,
    outdoor_temp: float,
    timestamp: pd.Timestamp,
) -> np.ndarray:
    hour_sin, hour_cos = _hour_features(timestamp)
    month_sin, month_cos = _month_features(timestamp)
    margin = max(current_temp - boundary_temp, 0.0)
    thermal_delta = current_temp - outdoor_temp
    return np.array(
        [
            current_temp,
            boundary_temp,
            outdoor_temp,
            margin,
            np.log(margin + 0.1),
            thermal_delta,
            np.log(abs(thermal_delta) + 0.1),
            hour_sin,
            hour_cos,
            month_sin,
            month_cos,
        ],
        dtype=float,
    )
