from __future__ import annotations


class Baseline:
    """Hold setpoints constant regardless of time or price."""

    def __init__(
        self,
        heat_setpoint: float = 68.0,
        cool_setpoint: float = 76.0,
    ) -> None:
        self.heat_setpoint = heat_setpoint
        self.cool_setpoint = cool_setpoint

    def __call__(self, obs: dict) -> dict:
        return {"heat_setpoint": self.heat_setpoint, "cool_setpoint": self.cool_setpoint}


class PreCool:
    """
    Pre-cool before peak, then setback during peak.

    During [peak_start - precool_hours, peak_start): lower cooling setpoint by precool_offset.
    During [peak_start, peak_end): raise cooling setpoint by setback.
    Otherwise: baseline setpoints.
    Heating setpoint is always base_heat.
    """

    def __init__(
        self,
        precool_offset: float = 2.0,
        precool_hours: float = 2,
        peak_start: int = 17,
        peak_end: int = 20,
        setback: float = 2.0,
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None:
        self.precool_offset = precool_offset
        self.precool_hours = precool_hours
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.setback = setback
        self.base_heat = base_heat
        self.base_cool = base_cool

    def __call__(self, obs: dict) -> dict:
        hour = obs["hour"]
        precool_threshold = self.peak_start - self.precool_hours
        if precool_threshold <= hour < self.peak_start:
            cool_sp = self.base_cool - self.precool_offset
        elif self.peak_start <= hour < self.peak_end:
            cool_sp = self.base_cool + self.setback
        else:
            cool_sp = self.base_cool
        return {"heat_setpoint": self.base_heat, "cool_setpoint": cool_sp}


class PreHeat:
    """
    Pre-heat before peak, then setback during peak.

    During [peak_start - preheat_hours, peak_start): raise heating setpoint by preheat_offset.
    During [peak_start, peak_end): lower heating setpoint by setback.
    Otherwise: baseline setpoints.
    Cooling setpoint is always base_cool.
    """

    def __init__(
        self,
        preheat_offset: float = 2.0,
        preheat_hours: float = 2,
        peak_start: int = 17,
        peak_end: int = 20,
        setback: float = 2.0,
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None:
        self.preheat_offset = preheat_offset
        self.preheat_hours = preheat_hours
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.setback = setback
        self.base_heat = base_heat
        self.base_cool = base_cool

    def __call__(self, obs: dict) -> dict:
        hour = obs["hour"]
        preheat_threshold = self.peak_start - self.preheat_hours
        if preheat_threshold <= hour < self.peak_start:
            heat_sp = self.base_heat + self.preheat_offset
        elif self.peak_start <= hour < self.peak_end:
            heat_sp = self.base_heat - self.setback
        else:
            heat_sp = self.base_heat
        return {"heat_setpoint": heat_sp, "cool_setpoint": self.base_cool}


class Setback:
    """
    Raise cooling (or lower heating) setpoint during peak. No pre-conditioning.

    During [peak_start, peak_end):
        if mode in ("cooling", "both"): cool_setpoint = base_cool + magnitude
        if mode in ("heating", "both"): heat_setpoint = base_heat - magnitude
    Outside peak: baseline setpoints.
    """

    def __init__(
        self,
        magnitude: float = 4.0,
        peak_start: int = 17,
        peak_end: int = 20,
        mode: str = "cooling",
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None:
        if mode not in ("cooling", "heating", "both"):
            raise ValueError(f"mode must be 'cooling', 'heating', or 'both'; got {mode!r}")
        self.magnitude = magnitude
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.mode = mode
        self.base_heat = base_heat
        self.base_cool = base_cool

    def __call__(self, obs: dict) -> dict:
        hour = obs["hour"]
        heat_sp = self.base_heat
        cool_sp = self.base_cool
        if self.peak_start <= hour < self.peak_end:
            if self.mode in ("cooling", "both"):
                cool_sp = self.base_cool + self.magnitude
            if self.mode in ("heating", "both"):
                heat_sp = self.base_heat - self.magnitude
        return {"heat_setpoint": heat_sp, "cool_setpoint": cool_sp}


class PriceResponse:
    """
    Linear mapping from electricity price to cooling setpoint adjustment.

    Below threshold_low: apply adjust_low to cool_setpoint.
    Above threshold_high: apply adjust_high to cool_setpoint.
    Between thresholds: linearly interpolate.
    Heating setpoint is never adjusted.
    """

    def __init__(
        self,
        threshold_low: float = 0.10,
        threshold_high: float = 0.25,
        adjust_low: float = -1.0,
        adjust_high: float = 2.0,
        base_heat: float = 68.0,
        base_cool: float = 76.0,
    ) -> None:
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.adjust_low = adjust_low
        self.adjust_high = adjust_high
        self.base_heat = base_heat
        self.base_cool = base_cool

    def __call__(self, obs: dict) -> dict:
        price = obs["electricity_price"]
        if price >= self.threshold_high:
            adjust = self.adjust_high
        elif price <= self.threshold_low:
            adjust = self.adjust_low
        else:
            t = (price - self.threshold_low) / (self.threshold_high - self.threshold_low)
            adjust = self.adjust_low + t * (self.adjust_high - self.adjust_low)
        return {
            "heat_setpoint": self.base_heat,
            "cool_setpoint": self.base_cool + adjust,
        }
