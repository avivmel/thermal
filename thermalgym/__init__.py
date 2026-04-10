"""
ThermalGym — EnergyPlus-backed demand response simulation environment.

Quick start::

    import thermalgym

    env = thermalgym.ThermalEnv(building="medium_cold_heatpump")
    obs = env.reset(date="2017-07-15")
    while not env.done:
        action = {"heat_setpoint": 70, "cool_setpoint": 76}
        obs = env.step(action)
    print(env.history.head())

"""

from thermalgym.env import ThermalEnv, HEAT_MIN, HEAT_MAX, COOL_MIN, COOL_MAX
from thermalgym.buildings import BUILDINGS, get_building, get_buildings, Building
from thermalgym.generate import generate_episodes, evaluate
from thermalgym.policies import Baseline, PreCool, PreHeat, Setback, PriceResponse
from mpc import MPCController

__version__ = "0.1.0"
__all__ = [
    # Environment
    "ThermalEnv",
    # Setpoint bounds
    "HEAT_MIN", "HEAT_MAX", "COOL_MIN", "COOL_MAX",
    # Buildings
    "Building", "BUILDINGS", "get_building", "get_buildings",
    # Data generation
    "generate_episodes", "evaluate",
    # Policies
    "Baseline", "PreCool", "PreHeat", "Setback", "PriceResponse",
    # MPC
    "MPCController",
]
