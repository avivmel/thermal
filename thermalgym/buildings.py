from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"

_EPW_FILES = {
    "cold": _DATA_DIR / "weather" / "USA_MN_Duluth.Intl.AP.727450_TMY3.epw",
    "hot": _DATA_DIR / "weather" / "USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw",
    "mixed": _DATA_DIR / "weather" / "USA_OH_Columbus-Port.Columbus.Intl.AP.724280_TMY3.epw",
}


@dataclass(frozen=True)
class Building:
    id: str
    idf_path: Path
    epw_path: Path
    climate_zone: str       # "cold" | "hot" | "mixed"
    floor_area_sqft: int
    vintage: str            # "pre1980" | "1980_2000" | "post2000"
    hvac_type: str          # "heatpump" | "ac_resistance" | "resistance_only"
    hvac_capacity_kw: float
    zone_name: str = "conditioned space"  # EnergyPlus thermal zone name in the IDF


def _make_building(
    id: str,
    climate_zone: str,
    floor_area_sqft: int,
    vintage: str,
    hvac_type: str,
    hvac_capacity_kw: float,
    zone_name: str = "conditioned space",
) -> Building:
    return Building(
        id=id,
        idf_path=_DATA_DIR / "buildings" / f"{id}.idf",
        epw_path=_EPW_FILES[climate_zone],
        climate_zone=climate_zone,
        floor_area_sqft=floor_area_sqft,
        vintage=vintage,
        hvac_type=hvac_type,
        hvac_capacity_kw=hvac_capacity_kw,
        zone_name=zone_name,
    )


BUILDINGS: dict[str, Building] = {b.id: b for b in [
    _make_building("small_cold_heatpump",    "cold",  1000, "pre1980",    "heatpump",      5.0),
    _make_building("medium_cold_heatpump",   "cold",  2000, "1980_2000",  "heatpump",      8.0),
    _make_building("large_cold_resistance",  "cold",  3000, "post2000",   "ac_resistance", 12.0),
    _make_building("small_hot_heatpump",     "hot",   1000, "pre1980",    "heatpump",      5.0),
    _make_building("medium_hot_heatpump",    "hot",   2000, "1980_2000",  "heatpump",      8.0),
    _make_building("large_hot_ac",           "hot",   3000, "post2000",   "ac_resistance", 14.0),
    _make_building("small_mixed_heatpump",   "mixed", 1000, "pre1980",    "heatpump",      5.0),
    _make_building("medium_mixed_heatpump",  "mixed", 2000, "1980_2000",  "heatpump",      8.0),
    _make_building("large_mixed_ac",         "mixed", 3000, "post2000",   "ac_resistance", 12.0),
]}


def get_building(id: str) -> Building:
    """Return building by ID. Raises KeyError with helpful message if not found."""
    if id not in BUILDINGS:
        raise KeyError(f"Unknown building {id!r}. Available: {sorted(BUILDINGS)}")
    return BUILDINGS[id]


def get_buildings(**filters) -> list[Building]:
    """
    Filter BUILDINGS by attribute equality.

    Examples::

        get_buildings(climate_zone="cold")
        get_buildings(vintage="post2000", hvac_type="heatpump")

    Returns list (possibly empty). Never raises.
    Raises TypeError if a filter key is not a Building field.
    """
    valid_fields = {f for f in Building.__dataclass_fields__}  # type: ignore[attr-defined]
    for key in filters:
        if key not in valid_fields:
            raise TypeError(
                f"Invalid filter key {key!r}. Valid Building fields: {sorted(valid_fields)}"
            )
    return [b for b in BUILDINGS.values() if all(getattr(b, k) == v for k, v in filters.items())]
