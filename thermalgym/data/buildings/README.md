# Building IDF Files

These EnergyPlus IDF files are required for ThermalGym simulation but are not included in the repository due to size.

## Required Files

- small_cold_heatpump.idf
- medium_cold_heatpump.idf
- large_cold_resistance.idf
- small_hot_heatpump.idf
- medium_hot_heatpump.idf
- large_hot_ac.idf
- small_mixed_heatpump.idf
- medium_mixed_heatpump.idf
- large_mixed_ac.idf

## Requirements

Each IDF must:
- Define exactly one thermal zone named "LIVING ZONE"
- Include `ZoneControl:Thermostat` referencing `ThermostatSetpoint:DualSetpoint` objects
- Include electric HVAC system matching the building's hvac_type
- Use the `RunPeriod` object (dates will be overridden at runtime)

## Sources

Source from ResStock archetypes (NREL) or EnergyPlus example files.
