# Ecobee Donate Your Data (DYD) Dataset Documentation

## Overview

This dataset is a curated subset of the **Ecobee Donate Your Data (DYD)** program, containing smart thermostat data from 1,000 single-family homes across four U.S. states. The data spans the entire year of 2017 and provides insights into HVAC system operations, occupant behavior, and building thermal dynamics.

| Attribute | Value |
|-----------|-------|
| **Data Source** | Ecobee Donate Your Data Program |
| **Time Period** | January 1, 2017 – December 31, 2017 |
| **Temporal Resolution** | 5-minute intervals |
| **Number of Buildings** | 1,000 single-family homes |
| **Total Data Size** | ~31.66 GB (processed NetCDF files) |
| **Data Curator** | Na Luo, Lawrence Berkeley National Laboratory |
| **Curation Date** | March 31, 2022 |

---

## Background

### The Ecobee DYD Program

Ecobee's "Donate Your Data" program allows users of Ecobee smart thermostats to anonymously contribute their thermostat data for research purposes. The data includes:

- User-reported metadata (home and occupant characteristics)
- Thermostat sensor measurements (temperature, humidity, motion)
- HVAC system operational data (runtime, setpoints, modes)
- Remote sensor data (up to 5 additional room sensors per home)

This subset was curated by Lawrence Berkeley National Laboratory (LBNL) to provide a geographically diverse sample spanning multiple climate zones.

### Data Citation

**DOI:** 10.25584/ecobee/1854924

**Reference Publication:** [https://doi.org/10.1088/1748-9326/ac092f](https://doi.org/10.1088/1748-9326/ac092f)

### Acknowledgement

> This effort was supported by the Assistant Secretary for Energy Efficiency and Renewable Energy, Office of Building Technologies of the United States Department of Energy, under Contract No. DE-AC02-05CH11231. The authors appreciate the cooperation of Ecobee Inc. and its anonymous customers who participated in the Donate Your Data program.

---

## Geographic Coverage

The dataset covers four U.S. states representing different IECC (International Energy Conservation Code) climate zones:

| State | Climate Zone | Climate Description | Homes in Dataset |
|-------|--------------|---------------------|------------------|
| California (CA) | 3C | Warm-Marine | 346 |
| Texas (TX) | 2A | Hot-Humid | 244 |
| Illinois (IL) | 5A | Cold-Humid | 248 |
| New York (NY) | 4A | Mixed-Humid | 133 |
| Unspecified | — | — | 19 |
| **Total** | — | — | **990*** |

*Note: The processed files contain 990 homes with valid state assignments out of the original 1,000.

---

## Building Characteristics

| Characteristic | Range/Description |
|----------------|-------------------|
| **Building Type** | Single-family residential |
| **Year Built** | 1900–2016 |
| **Number of Floors** | 1–5 |
| **Floor Area** | Up to 6,500 sq ft |
| **Number of Occupants** | Up to 10 |

### HVAC System Types

The dataset includes various HVAC configurations:
- Split direct expansion cooling + furnace
- Heat pumps (single and multi-stage)
- Multi-stage heating and cooling equipment

**Fuel Types:** Natural gas, electricity

---

## Data Files Description

### File Structure

```
ecobee_processed_dataset/
├── Metadata file_Ecobee.json    # Building and dataset metadata
├── Brick model_Ecobee.pdf       # Semantic data model diagram
├── Ecobee_dataset_cleaning_report.docx
└── clean_data/
    ├── Jan_clean.nc             # January 2017 (2.70 GB)
    ├── Feb_clean.nc             # February 2017 (2.45 GB)
    ├── Mar_clean.nc             # March 2017 (2.70 GB)
    ├── Apr_clean.nc             # April 2017 (2.60 GB)
    ├── May_clean.nc             # May 2017 (2.69 GB)
    ├── Jun_clean.nc             # June 2017 (2.60 GB)
    ├── Jul_clean.nc             # July 2017 (2.69 GB)
    ├── Aug_clean.nc             # August 2017 (2.68 GB)
    ├── Sep_clean.nc             # September 2017 (2.60 GB)
    ├── Oct_clean.nc             # October 2017 (2.67 GB)
    ├── Nov_clean.nc             # November 2017 (2.60 GB)
    └── Dec_clean.nc             # December 2017 (2.68 GB)
```

### Data Format

The processed data is stored in **NetCDF4** format, a self-describing binary format widely used for multidimensional scientific data. Each monthly file follows the **MHKiT-Cloud Data Standards v. 1.0** convention.

---

## Data Schema

### Dimensions

| Dimension | Description | Typical Size |
|-----------|-------------|--------------|
| `id` | Home identifier dimension | 990 |
| `time` | Time steps (5-minute intervals) | 8,640–8,928 per month |

### Variables

#### Identifiers and Metadata

| Variable | Type | Dimensions | Description |
|----------|------|------------|-------------|
| `id` | string | (id) | Anonymized home identifier (SHA-256 hash) |
| `time` | int64 | (time) | Unix timestamp (seconds since 1970-01-01 UTC) |
| `State` | string | (id, time) | U.S. state code (CA, TX, IL, NY) |

#### Thermostat Schedule & Events

| Variable | Type | Dimensions | Description |
|----------|------|------------|-------------|
| `Schedule` | string | (id, time) | Current schedule mode: `Home`, `Away`, `Sleep`, or empty |
| `Event` | string | (id, time) | Active event: `Hold`, custom events (e.g., `custom_6`), or empty |
| `HVAC_Mode` | int32 | (id, time) | HVAC operating mode (-9999 = missing/unavailable) |

#### Indoor Environmental Measurements

| Variable | Type | Dimensions | Unit | Description |
|----------|------|------------|------|-------------|
| `Indoor_AverageTemperature` | float64 | (id, time) | °F | Weighted average of all sensor temperatures |
| `Indoor_Humidity` | float64 | (id, time) | % | Indoor relative humidity |
| `Indoor_CoolSetpoint` | float64 | (id, time) | °F | Cooling temperature setpoint |
| `Indoor_HeatSetpoint` | float64 | (id, time) | °F | Heating temperature setpoint |

#### Thermostat Sensor

| Variable | Type | Dimensions | Unit | Description |
|----------|------|------------|------|-------------|
| `Thermostat_Temperature` | float64 | (id, time) | °F | Temperature at thermostat location |
| `Thermostat_DetectedMotion` | float64 | (id, time) | binary | Motion detected (1.0 = yes) |

#### Remote Sensors (1–5)

Each home may have up to 5 remote sensors. For each sensor N (1–5):

| Variable | Type | Dimensions | Unit | Description |
|----------|------|------------|------|-------------|
| `RemoteSensorN_Temperature` | float64 | (id, time) | °F | Temperature at remote sensor N |
| `RemoteSensorN_DetectedMotion` | float64 | (id, time) | binary | Motion detected at sensor N |

#### Outdoor Environmental Measurements

| Variable | Type | Dimensions | Unit | Description |
|----------|------|------------|------|-------------|
| `Outdoor_Temperature` | float64 | (id, time) | °F | Outdoor air temperature |
| `Outdoor_Humidity` | float64 | (id, time) | % | Outdoor relative humidity |

#### HVAC Equipment Runtime

Runtime values represent seconds of operation within each 5-minute interval (maximum = 300 seconds).

| Variable | Type | Dimensions | Unit | Description |
|----------|------|------------|------|-------------|
| `HeatingEquipmentStage1_RunTime` | float64 | (id, time) | seconds | Stage 1 heating runtime |
| `HeatingEquipmentStage2_RunTime` | float64 | (id, time) | seconds | Stage 2 heating runtime |
| `HeatingEquipmentStage3_RunTime` | float64 | (id, time) | seconds | Stage 3 heating runtime |
| `CoolingEquipmentStage1_RunTime` | float64 | (id, time) | seconds | Stage 1 cooling runtime |
| `CoolingEquipmentStage2_RunTime` | float64 | (id, time) | seconds | Stage 2 cooling runtime |
| `HeatPumpsStage1_RunTime` | float64 | (id, time) | seconds | Stage 1 heat pump runtime |
| `HeatPumpsStage2_RunTime` | float64 | (id, time) | seconds | Stage 2 heat pump runtime |
| `Fan_RunTime` | float64 | (id, time) | seconds | Fan runtime |

---

## Brick Schema Model

The dataset follows the [Brick Schema](https://brickschema.org/) ontology for describing building systems. The semantic model includes:

```
                    ┌─────────────────────────────┐
                    │          Outside            │
                    └─────────────────────────────┘
                              ▲
           ┌──────────────────┴──────────────────┐
           │                                      │
┌──────────┴──────────┐            ┌──────────────┴──────────────┐
│Outside_Air_Humidity │            │  Outside_Air_Temperature    │
│      Sensor         │            │         Sensor              │
└─────────────────────┘            └─────────────────────────────┘

                    ┌─────────────────────────────┐
                    │         Thermostat          │◄─── isLocationOf ───┐
                    └─────────────────────────────┘                     │
                              ▲                                         │
         ┌────────────────────┼────────────────────┐                   │
         │                    │                    │                    │
┌────────┴────────┐  ┌────────┴────────┐  ┌───────┴───────┐     ┌─────┴─────┐
│Cooling_Temp_    │  │Heating_Temp_    │  │ Motion_Sensor │     │   Floor   │
│   Setpoint      │  │   Setpoint      │  └───────────────┘     └───────────┘
└─────────────────┘  └─────────────────┘                              │
                                                                       │
                    ┌─────────────────────────────┐                   │
                    │           Room              │◄── isLocationOf ──┘
                    └─────────────────────────────┘
                              ▲
                    ┌─────────┴─────────┐
                    │                   │
         ┌──────────┴──────────┐       │
         │Zone_Air_Temperature │       │
         │      Sensor         │       │
         └─────────────────────┘       │
                    ▲                  │
                    │ isMeasuredBy     │ isRegulatedBy
                    │                  │
┌───────────────────┴──────────┐       │
│Averaged_Zone_Air_Temperature │       │
│          Sensor              │       │
├──────────────────────────────┤       ▼
│  Relative_Humidity_Sensor    │  ┌─────────────────┐
└──────────────────────────────┘  │   HVAC_System   │
                                  └────────┬────────┘
                                           │ hasPoint
                                           ▼
                                  ┌─────────────────┐
                                  │ Run_Time_Sensor │
                                  └─────────────────┘
```

### Key Relationships

| Relationship | Description |
|--------------|-------------|
| `isPointOf` | Sensor/setpoint belongs to a system or location |
| `isMeasuredBy` | Physical quantity measured by a sensor |
| `isRegulatedBy` | Room conditions controlled by HVAC system |
| `feeds` | HVAC system provides conditioned air to room |
| `hasPoint` | System has an associated measurement point |
| `isLocationOf` | Physical location of equipment |

---

## Data Quality

### Data Cleaning Process

The raw data was processed with the following curation strategies:
- Selection of 1,000 single-family homes
- Equal distribution across four U.S. states
- Missing data gaps filled using **linear interpolation**

### Quality Metrics

| Data Category | Missing Rate | Outlier Rate |
|---------------|--------------|--------------|
| Indoor Environmental | 2% | 1% |
| Outdoor Environmental | 3% | 0.5% |
| System Operational | 2% | 1% |
| Control Logic | 2% | 1% |

### Missing Value Indicators

| Indicator | Meaning |
|-----------|---------|
| `0.0` | Missing/no data (for float variables) |
| `-9999` | Missing/unavailable (for HVAC_Mode) |
| Empty string `""` | No active event/schedule |
| `--` (masked) | NumPy masked value |

---

## Data Statistics Summary

### Temperature Ranges (January 2017)

| Measurement | Min | Max | Mean |
|-------------|-----|-----|------|
| Indoor Average Temperature | 42°F | 85°F | 68°F |
| Outdoor Temperature | -9°F | 94°F | 43°F |
| Cooling Setpoint | 58°F | 100°F | — |
| Heating Setpoint | 45°F | 82°F | — |

### HVAC Runtime Statistics (January 2017)

| Equipment | Min Runtime | Max Runtime | Mean Runtime |
|-----------|-------------|-------------|--------------|
| Heating Stage 1 | 15 sec | 300 sec | 219 sec |
| Cooling Stage 1 | 15 sec | 300 sec | 219 sec |
| Heat Pump Stage 1 | 15 sec | 300 sec | 235 sec |
| Fan | 15 sec | 300 sec | 220 sec |

---

## Usage Examples

### Loading Data with Python (xarray)

```python
import xarray as xr

# Load a single month
ds = xr.open_dataset('clean_data/Jan_clean.nc')

# Access indoor temperature for a specific home
home_id = ds.id.values[0]
temp = ds['Indoor_AverageTemperature'].sel(id=home_id)

# Plot daily average temperature
temp.resample(time='1D').mean().plot()
```

### Loading Data with Python (netCDF4)

```python
import netCDF4 as nc
import numpy as np

dataset = nc.Dataset('clean_data/Jan_clean.nc', 'r')

# Get all home IDs
home_ids = dataset.variables['id'][:]

# Get time as datetime objects
time_var = dataset.variables['time']
times = nc.num2date(time_var[:], units=time_var.units)

# Get indoor temperature for first home
indoor_temp = dataset.variables['Indoor_AverageTemperature'][0, :]

dataset.close()
```

### Filtering by State

```python
import xarray as xr

ds = xr.open_dataset('clean_data/Jan_clean.nc')

# Get California homes only
ca_mask = ds['State'].isel(time=0) == 'CA'
ca_homes = ds.where(ca_mask, drop=True)
```

---

## Potential Applications

1. **Human-Building Interaction (HBI)** — Analyze occupant thermostat interactions and setpoint preferences

2. **Indoor Environmental Quality (IEQ)** — Study temperature and humidity patterns across different climate zones

3. **Building Thermal Dynamics** — Model heat transfer characteristics using indoor/outdoor temperature differentials

4. **HVAC System Performance** — Evaluate equipment runtime patterns and efficiency

5. **Occupancy Detection** — Use motion sensor data to infer occupancy patterns

6. **Demand Response Studies** — Analyze how residential HVAC loads respond to setpoint changes

7. **Machine Learning Applications** — Train models for load forecasting, anomaly detection, or predictive maintenance

---

## Access and Licensing

| Resource | Link |
|----------|------|
| Raw Data Access | [https://bbd.labworks.org/ds/bbd/ecobee](https://bbd.labworks.org/ds/bbd/ecobee) |
| More Information | [https://www.ecobee.com/donate-your-data/](https://www.ecobee.com/donate-your-data/) |
| DOI | 10.25584/ecobee/1854924 |

### Contact

**Na Luo**
Lawrence Berkeley National Laboratory
Email: nluo@lbl.gov

---

## Appendix: Complete Variable List

| # | Variable Name | Type | Description |
|---|---------------|------|-------------|
| 1 | `id` | string | Anonymized home identifier |
| 2 | `time` | int64 | Unix timestamp (UTC) |
| 3 | `State` | string | U.S. state code |
| 4 | `Event` | string | Active thermostat event |
| 5 | `Schedule` | string | Current schedule mode |
| 6 | `Indoor_AverageTemperature` | float64 | Average indoor temperature |
| 7 | `Indoor_CoolSetpoint` | float64 | Cooling setpoint |
| 8 | `Indoor_HeatSetpoint` | float64 | Heating setpoint |
| 9 | `Indoor_Humidity` | float64 | Indoor relative humidity |
| 10 | `HeatingEquipmentStage1_RunTime` | float64 | Heating stage 1 runtime |
| 11 | `HeatingEquipmentStage2_RunTime` | float64 | Heating stage 2 runtime |
| 12 | `HeatingEquipmentStage3_RunTime` | float64 | Heating stage 3 runtime |
| 13 | `CoolingEquipmentStage1_RunTime` | float64 | Cooling stage 1 runtime |
| 14 | `CoolingEquipmentStage2_RunTime` | float64 | Cooling stage 2 runtime |
| 15 | `HeatPumpsStage1_RunTime` | float64 | Heat pump stage 1 runtime |
| 16 | `HeatPumpsStage2_RunTime` | float64 | Heat pump stage 2 runtime |
| 17 | `Fan_RunTime` | float64 | Fan runtime |
| 18 | `Thermostat_Temperature` | float64 | Thermostat sensor temperature |
| 19 | `Thermostat_DetectedMotion` | float64 | Thermostat motion detection |
| 20 | `RemoteSensor1_Temperature` | float64 | Remote sensor 1 temperature |
| 21 | `RemoteSensor1_DetectedMotion` | float64 | Remote sensor 1 motion |
| 22 | `RemoteSensor2_Temperature` | float64 | Remote sensor 2 temperature |
| 23 | `RemoteSensor2_DetectedMotion` | float64 | Remote sensor 2 motion |
| 24 | `RemoteSensor3_Temperature` | float64 | Remote sensor 3 temperature |
| 25 | `RemoteSensor3_DetectedMotion` | float64 | Remote sensor 3 motion |
| 26 | `RemoteSensor4_Temperature` | float64 | Remote sensor 4 temperature |
| 27 | `RemoteSensor4_DetectedMotion` | float64 | Remote sensor 4 motion |
| 28 | `RemoteSensor5_Temperature` | float64 | Remote sensor 5 temperature |
| 29 | `RemoteSensor5_DetectedMotion` | float64 | Remote sensor 5 motion |
| 30 | `Outdoor_Temperature` | float64 | Outdoor air temperature |
| 31 | `Outdoor_Humidity` | float64 | Outdoor relative humidity |
| 32 | `HVAC_Mode` | int32 | HVAC operating mode |

---

*Document generated: 2024*
*Data version: March 2022 curation*
