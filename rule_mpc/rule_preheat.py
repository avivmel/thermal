from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from mpc.model_interfaces import FirstPassagePredictor


RuleForecastProvider = Callable[[dict, int, int], tuple[list[pd.Timestamp], np.ndarray]]


@dataclass(frozen=True)
class RulePreheatMPCConfig:
    timestep_minutes: int = 5
    normal_heat_setpoint: float = 68.0
    peak_heat_setpoint: float = 66.0
    peak_lower_comfort: float = 66.0
    cool_setpoint: float = 76.0
    peak_start_hour: int = 17
    peak_end_hour: int = 20
    candidate_preheat_targets: tuple[float, ...] = (
        68.5,
        69.0,
        69.5,
        70.0,
        70.5,
        71.0,
        72.0,
    )
    earliest_preheat_hour: int = 14
    latest_preheat_start_margin_min: float = 10.0
    drift_safety_margin_min: float = 15.0
    heat_time_safety_margin_min: float = 30.0
    max_allowed_indoor_temp: float = 72.0
    max_preheat_runtime_min: float | None = None
    preheat_runtime_weight: float = 0.25
    peak_runtime_weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self) -> None:
        if not self.candidate_preheat_targets:
            raise ValueError("candidate_preheat_targets must be non-empty")


@dataclass(frozen=True)
class RulePreheatCandidate:
    preheat_target: float
    preheat_start: pd.Timestamp
    predicted_heat_time_min: float
    predicted_coast_time_min: float
    predicted_residual_peak_runtime_min: float
    score: float
    feasible: bool
    reason: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["preheat_start"] = self.preheat_start.isoformat()
        return data


@dataclass(frozen=True)
class RulePreheatPlan:
    preheat_start: pd.Timestamp | None
    preheat_target: float
    peak_setback: float
    predicted_heat_time_min: float
    predicted_coast_time_min: float
    predicted_residual_peak_runtime_min: float
    score: float
    peak_duration_min: float
    current_heat_setpoint: float
    feasible: bool
    reason: str
    candidates: tuple[RulePreheatCandidate, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["preheat_start"] = None if self.preheat_start is None else self.preheat_start.isoformat()
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data


class RulePreheatMPCController:
    def __init__(
        self,
        predictor: FirstPassagePredictor,
        config: RulePreheatMPCConfig | None = None,
        forecast_provider: RuleForecastProvider | None = None,
        horizon_steps: int | None = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or RulePreheatMPCConfig()
        self.forecast_provider = forecast_provider or persistence_forecast
        self.horizon_steps = horizon_steps or (24 * 60 // self.config.timestep_minutes)
        self.last_plan: RulePreheatPlan | None = None

    def __call__(self, obs: dict) -> dict:
        plan = self.plan(obs)
        return {
            "heat_setpoint": plan.current_heat_setpoint,
            "cool_setpoint": self.config.cool_setpoint,
        }

    def plan(self, obs: dict) -> RulePreheatPlan:
        now = pd.Timestamp(obs["timestamp"])
        current_temp = float(obs["indoor_temp"])
        outdoor_now = float(obs["outdoor_temp"])
        hvac_running = obs.get("hvac_mode") == "heating"
        home_id = obs.get("home_id")

        timestamps, outdoor_forecast = self.forecast_provider(
            obs,
            self.horizon_steps,
            self.config.timestep_minutes,
        )
        timestamps = [pd.Timestamp(ts) for ts in timestamps]
        outdoor_forecast = np.asarray(outdoor_forecast, dtype=float)
        if len(timestamps) == 0:
            timestamps = [now]
            outdoor_forecast = np.array([outdoor_now], dtype=float)

        peak_start, peak_end = self._peak_window(now)
        peak_duration_min = max((peak_end - peak_start).total_seconds() / 60.0, 0.0)

        if now >= peak_end:
            return self._store_plan(
                RulePreheatPlan(
                    preheat_start=None,
                    preheat_target=self.config.normal_heat_setpoint,
                    peak_setback=self.config.peak_heat_setpoint,
                    predicted_heat_time_min=0.0,
                    predicted_coast_time_min=float("inf"),
                    predicted_residual_peak_runtime_min=0.0,
                    score=0.0,
                    peak_duration_min=peak_duration_min,
                    current_heat_setpoint=self.config.normal_heat_setpoint,
                    feasible=True,
                    reason="peak_window_has_ended",
                )
            )

        peak_outdoor = self._mean_outdoor_between(timestamps, outdoor_forecast, peak_start, peak_end)

        if now < peak_start:
            current_coast_time = self.predictor.predict_drift_time(
                current_temp=current_temp,
                boundary_temp=self.config.peak_lower_comfort,
                outdoor_temp=peak_outdoor,
                timestamp=peak_start,
                home_id=home_id,
            )
            current_residual_peak_runtime = self._residual_peak_runtime(
                current_coast_time,
                peak_duration_min,
                include_safety_margin=False,
            )
            if current_residual_peak_runtime <= 0:
                return self._store_plan(
                    RulePreheatPlan(
                        preheat_start=None,
                        preheat_target=current_temp,
                        peak_setback=self.config.peak_heat_setpoint,
                        predicted_heat_time_min=0.0,
                        predicted_coast_time_min=current_coast_time,
                        predicted_residual_peak_runtime_min=current_residual_peak_runtime,
                        score=0.0,
                        peak_duration_min=peak_duration_min,
                        current_heat_setpoint=self.config.normal_heat_setpoint,
                        feasible=True,
                        reason="current_temperature_can_coast_through_peak",
                    )
                )

        if peak_start <= now < peak_end:
            minutes_to_peak_end = max((peak_end - now).total_seconds() / 60.0, 0.0)
            coast_time = self.predictor.predict_drift_time(
                current_temp=current_temp,
                boundary_temp=self.config.peak_lower_comfort,
                outdoor_temp=peak_outdoor,
                timestamp=now,
                home_id=home_id,
            )
            if self._survives_peak(coast_time, minutes_to_peak_end):
                heat_setpoint = self.config.peak_heat_setpoint
                reason = "coast_through_remaining_peak"
            else:
                heat_setpoint = self.config.peak_lower_comfort
                reason = "maintain_peak_lower_comfort"

            return self._store_plan(
                RulePreheatPlan(
                    preheat_start=None,
                    preheat_target=current_temp,
                    peak_setback=self.config.peak_heat_setpoint,
                    predicted_heat_time_min=0.0,
                    predicted_coast_time_min=coast_time,
                    predicted_residual_peak_runtime_min=self._residual_peak_runtime(
                        coast_time,
                        minutes_to_peak_end,
                        include_safety_margin=True,
                    ),
                    score=0.0,
                    peak_duration_min=peak_duration_min,
                    current_heat_setpoint=heat_setpoint,
                    feasible=True,
                    reason=reason,
                )
            )

        candidates = self._build_candidates(
            now=now,
            current_temp=current_temp,
            outdoor_temp=outdoor_now,
            peak_outdoor=peak_outdoor,
            peak_start=peak_start,
            peak_end=peak_end,
            peak_duration_min=peak_duration_min,
            timestamps=timestamps,
            outdoor_forecast=outdoor_forecast,
            hvac_running=hvac_running,
            home_id=home_id,
        )
        feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
        selected = min(feasible_candidates, key=lambda candidate: (candidate.score, candidate.preheat_target)) if feasible_candidates else None

        if now < peak_start:
            if selected is None:
                return self._store_plan(
                    RulePreheatPlan(
                        preheat_start=None,
                        preheat_target=self.config.normal_heat_setpoint,
                        peak_setback=self.config.peak_heat_setpoint,
                        predicted_heat_time_min=0.0,
                        predicted_coast_time_min=0.0,
                        predicted_residual_peak_runtime_min=peak_duration_min,
                        score=float("inf"),
                        peak_duration_min=peak_duration_min,
                        current_heat_setpoint=self.config.normal_heat_setpoint,
                        feasible=False,
                        reason="no_feasible_preheat_target",
                        candidates=tuple(candidates),
                    )
                )

            heat_setpoint = (
                selected.preheat_target
                if now >= selected.preheat_start
                else self.config.normal_heat_setpoint
            )
            return self._store_plan(
                RulePreheatPlan(
                    preheat_start=selected.preheat_start,
                    preheat_target=selected.preheat_target,
                    peak_setback=self.config.peak_heat_setpoint,
                    predicted_heat_time_min=selected.predicted_heat_time_min,
                    predicted_coast_time_min=selected.predicted_coast_time_min,
                    predicted_residual_peak_runtime_min=selected.predicted_residual_peak_runtime_min,
                    score=selected.score,
                    peak_duration_min=peak_duration_min,
                    current_heat_setpoint=heat_setpoint,
                    feasible=True,
                    reason="lowest_peak_runtime_score",
                    candidates=tuple(candidates),
                )
            )

        return self._store_plan(
            RulePreheatPlan(
                preheat_start=None,
                preheat_target=self.config.normal_heat_setpoint,
                peak_setback=self.config.peak_heat_setpoint,
                predicted_heat_time_min=0.0,
                predicted_coast_time_min=float("inf"),
                predicted_residual_peak_runtime_min=0.0,
                score=0.0,
                peak_duration_min=peak_duration_min,
                current_heat_setpoint=self.config.normal_heat_setpoint,
                feasible=True,
                reason="before_peak_without_selected_candidate",
                candidates=tuple(candidates),
            )
        )

    def _build_candidates(
        self,
        now: pd.Timestamp,
        current_temp: float,
        outdoor_temp: float,
        peak_outdoor: float,
        peak_start: pd.Timestamp,
        peak_end: pd.Timestamp,
        peak_duration_min: float,
        timestamps: list[pd.Timestamp],
        outdoor_forecast: np.ndarray,
        hvac_running: bool,
        home_id: str | None,
    ) -> list[RulePreheatCandidate]:
        candidates: list[RulePreheatCandidate] = []
        earliest_start = peak_start.normalize() + pd.Timedelta(hours=self.config.earliest_preheat_hour)
        latest_start = peak_start - pd.Timedelta(minutes=self.config.latest_preheat_start_margin_min)

        for target in self.config.candidate_preheat_targets:
            first_pass_heat_time = self.predictor.predict_heat_time(
                current_temp=current_temp,
                target_temp=target,
                outdoor_temp=outdoor_temp,
                timestamp=now,
                system_running=hvac_running,
                home_id=home_id,
            )
            first_pass_start = peak_start - pd.Timedelta(
                minutes=first_pass_heat_time + self.config.heat_time_safety_margin_min
            )
            outdoor_at_start = self._nearest_outdoor(timestamps, outdoor_forecast, first_pass_start)
            starts_now = now >= first_pass_start
            heat_time = self.predictor.predict_heat_time(
                current_temp=current_temp,
                target_temp=target,
                outdoor_temp=outdoor_at_start,
                timestamp=now if starts_now else first_pass_start,
                system_running=hvac_running if starts_now else False,
                home_id=home_id,
            )
            preheat_start = peak_start - pd.Timedelta(
                minutes=heat_time + self.config.heat_time_safety_margin_min
            )
            coast_time = self.predictor.predict_drift_time(
                current_temp=target,
                boundary_temp=self.config.peak_lower_comfort,
                outdoor_temp=peak_outdoor,
                timestamp=peak_start,
                home_id=home_id,
            )
            feasible, reason = self._candidate_feasibility(
                target=target,
                preheat_start=preheat_start,
                heat_time=heat_time,
                now=now,
                earliest_start=earliest_start,
                latest_start=latest_start,
            )
            residual_peak_runtime = self._residual_peak_runtime(
                coast_time,
                peak_duration_min,
                include_safety_margin=True,
            )
            score = self._candidate_score(
                predicted_heat_time_min=heat_time,
                predicted_residual_peak_runtime_min=residual_peak_runtime,
            )
            candidates.append(
                RulePreheatCandidate(
                    preheat_target=target,
                    preheat_start=preheat_start,
                    predicted_heat_time_min=heat_time,
                    predicted_coast_time_min=coast_time,
                    predicted_residual_peak_runtime_min=residual_peak_runtime,
                    score=score,
                    feasible=feasible,
                    reason=reason,
                )
            )
        return candidates

    def _candidate_feasibility(
        self,
        target: float,
        preheat_start: pd.Timestamp,
        heat_time: float,
        now: pd.Timestamp,
        earliest_start: pd.Timestamp,
        latest_start: pd.Timestamp,
    ) -> tuple[bool, str]:
        if target > self.config.max_allowed_indoor_temp:
            return False, "preheat_target_exceeds_max_allowed_indoor_temp"
        if preheat_start < earliest_start:
            return False, "preheat_start_before_earliest_allowed_time"
        if preheat_start < now:
            return False, "preheat_start_before_current_time"
        if preheat_start > latest_start:
            return False, "preheat_start_after_latest_allowed_time"
        if (
            self.config.max_preheat_runtime_min is not None
            and heat_time > self.config.max_preheat_runtime_min
        ):
            return False, "predicted_preheat_runtime_too_high"
        return True, "feasible"

    def _candidate_score(
        self,
        predicted_heat_time_min: float,
        predicted_residual_peak_runtime_min: float,
    ) -> float:
        return (
            self.config.peak_runtime_weight * predicted_residual_peak_runtime_min
            + self.config.preheat_runtime_weight * predicted_heat_time_min
        )

    def _residual_peak_runtime(
        self,
        coast_time_min: float,
        required_duration_min: float,
        include_safety_margin: bool,
    ) -> float:
        margin = self.config.drift_safety_margin_min if include_safety_margin else 0.0
        return max(required_duration_min + margin - coast_time_min, 0.0)

    def _survives_peak(self, coast_time_min: float, required_duration_min: float) -> bool:
        return coast_time_min >= required_duration_min + self.config.drift_safety_margin_min

    def _peak_window(self, now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        day = now.normalize()
        peak_start = day + pd.Timedelta(hours=self.config.peak_start_hour)
        peak_end = day + pd.Timedelta(hours=self.config.peak_end_hour)
        if peak_end <= peak_start:
            peak_end += pd.Timedelta(days=1)
        return peak_start, peak_end

    def _mean_outdoor_between(
        self,
        timestamps: list[pd.Timestamp],
        outdoor_forecast: np.ndarray,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        mask = np.array([(start <= ts < end) for ts in timestamps], dtype=bool)
        if mask.any():
            return float(outdoor_forecast[mask].mean())
        return float(outdoor_forecast[0])

    def _nearest_outdoor(
        self,
        timestamps: list[pd.Timestamp],
        outdoor_forecast: np.ndarray,
        target: pd.Timestamp,
    ) -> float:
        if not timestamps:
            return float(outdoor_forecast[0])
        deltas = np.array([abs((ts - target).total_seconds()) for ts in timestamps], dtype=float)
        return float(outdoor_forecast[int(deltas.argmin())])

    def _store_plan(self, plan: RulePreheatPlan) -> RulePreheatPlan:
        self.last_plan = plan
        return plan


def persistence_forecast(
    obs: dict,
    horizon_steps: int,
    timestep_minutes: int,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    start = pd.Timestamp(obs["timestamp"])
    timestamps = [
        start + pd.Timedelta(minutes=timestep_minutes * offset)
        for offset in range(horizon_steps)
    ]
    outdoor = np.full(horizon_steps, float(obs["outdoor_temp"]), dtype=float)
    return timestamps, outdoor
