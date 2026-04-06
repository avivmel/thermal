# Software Requirements Specification: ThermalGym

**Version**: 1.1
**Date**: 2026-04-05

---

## 1. Introduction

### 1.1 Purpose

ThermalGym is a simulation environment for developing and evaluating residential demand response (DR) control policies. It enables researchers to test thermostat-based DR strategies against physically realistic building models before deploying to real homes.

### 1.2 Scope

ThermalGym provides:
- Simulation of residential thermal dynamics using EnergyPlus
- A library of representative US homes across climate zones
- A thermostat setpoint control interface (no direct HVAC power control)
- Training data generation compatible with existing Ecobee datasets
- Evaluation framework for DR policy performance

ThermalGym does NOT:
- Control real thermostats or HVAC equipment
- Provide pre-trained ML models
- Simulate non-residential buildings
- Model non-electric HVAC systems (gas, oil)

### 1.3 Definitions

| Term | Definition |
|------|------------|
| **DR (Demand Response)** | Reducing or shifting electricity consumption during peak demand periods |
| **Setpoint** | Target temperature for thermostat control (heating or cooling) |
| **Episode** | A contiguous simulation period from setpoint change to target reached |
| **Pre-conditioning** | Heating/cooling a building before a DR event to store thermal energy |
| **Setback** | Temporarily relaxing comfort bounds to reduce HVAC load |
| **TOU (Time-of-Use)** | Electricity pricing that varies by time of day |

### 1.4 References

- Ecobee Donate Your Data dataset (LBNL)
- ResStock residential building stock model (NREL)
- EnergyPlus building energy simulation (DOE)
- CityLearn demand response environment

---

## 2. Overall Description

### 2.1 Product Perspective

ThermalGym fills a gap between:
- **CityLearn**: Uses learned (LSTM) thermal models with direct power control
- **Real thermostats**: Provide setpoint control but no simulation capability
- **Raw EnergyPlus**: Requires expertise; not designed for policy iteration

ThermalGym combines EnergyPlus physics with a thermostat-realistic control interface.

### 2.2 User Characteristics

**Primary users**: Researchers developing DR algorithms who:
- Have Python programming experience
- Understand basic HVAC and building thermal concepts
- May not have EnergyPlus expertise

### 2.3 Constraints

- C1: Simulations must run faster than real-time
- C2: Requires EnergyPlus installation (version 23.1+)
- C3: Building library limited to US residential archetypes
- C4: Electric HVAC systems only

### 2.4 Assumptions

- A1: Users have access to sufficient compute for batch simulations
- A2: Training data format compatibility with existing Ecobee pipeline is required
- A3: Single-zone thermal models are sufficient for initial release

---

## 3. Functional Requirements

### 3.1 Building Library

**FR-BL-1**: The system shall provide a library of pre-configured residential building models.

**FR-BL-2**: The building library shall include homes from at least three climate zones:
- Cold (heating-dominated, e.g., IECC Zone 6)
- Hot (cooling-dominated, e.g., IECC Zone 2)
- Mixed (heating and cooling, e.g., IECC Zone 4)

**FR-BL-3**: Each climate zone shall include buildings with varying:
- Floor area (small: ~1000 sqft, medium: ~2000 sqft, large: ~3000 sqft)
- Construction vintage (pre-1980, 1980-2000, post-2000)

**FR-BL-4**: The system shall support only electric HVAC types:
- Air-source heat pump (heating and cooling)
- Central air conditioning with electric resistance heating
- Electric resistance heating only

**FR-BL-5**: Each building model shall include metadata:
- Unique identifier
- Climate zone
- HVAC type and capacity (kW)
- Floor area
- Construction vintage

**FR-BL-6**: Building models shall be derived from ResStock archetypes to ensure physical realism.

### 3.2 Thermal Simulation

**FR-TS-1**: The system shall simulate building thermal dynamics using EnergyPlus.

**FR-TS-2**: The simulation timestep shall be configurable: 5 minutes, 15 minutes, or 60 minutes.

**FR-TS-3**: The simulation shall compute at each timestep:
- Indoor air temperature
- HVAC electric power consumption
- HVAC operating mode (heating, cooling, or off)

**FR-TS-4**: The system shall provide weather files for each supported climate zone.

**FR-TS-5**: The simulation shall model realistic thermostat behavior:
- HVAC turns on when indoor temperature crosses setpoint
- HVAC turns off when setpoint is satisfied
- Deadband behavior to prevent short-cycling

### 3.3 Control Interface

**FR-CI-1**: The control interface shall accept only thermostat setpoints as inputs:
- Heating setpoint (°F)
- Cooling setpoint (°F)

**FR-CI-2**: The system shall NOT allow direct control of HVAC power level.

**FR-CI-3**: Setpoints shall be constrained to realistic ranges:
- Heating setpoint: 55°F to 75°F
- Cooling setpoint: 70°F to 85°F

**FR-CI-4**: The control interface shall provide observations at each timestep:
- Current indoor temperature (°F)
- Current outdoor temperature (°F)
- Current HVAC power consumption (kW)
- Current HVAC mode (heating/cooling/off)
- Current setpoints (°F)
- Time information (hour, day of week, month)

**FR-CI-5**: The control interface shall accept an optional electricity price signal for price-responsive policies.

### 3.4 Training Data Generation

**FR-TD-1**: The system shall generate training episodes for time-to-target prediction models.

**FR-TD-2**: Generated episodes shall match the schema of the existing `setpoint_responses.parquet` dataset:
- Episode identifiers (home_id, episode_id)
- Time series of temperatures and HVAC state
- Target variable: time_to_target_minutes

**FR-TD-3**: The system shall generate heating increase episodes:
- Initial indoor temperature below heating setpoint
- Episode ends when indoor temperature reaches setpoint
- Configurable temperature gap range (e.g., 1-5°F)

**FR-TD-4**: The system shall generate cooling decrease episodes:
- Initial indoor temperature above cooling setpoint
- Episode ends when indoor temperature reaches setpoint
- Configurable temperature gap range

**FR-TD-5**: The system shall allow specification of:
- Number of episodes to generate
- Range of outdoor temperatures
- Range of initial temperature gaps
- Subset of buildings to use

**FR-TD-6**: The system shall vary initial conditions across episodes:
- Time of day
- Day of year (seasonal variation)
- Building thermal state (pre-conditioned vs. cold start)

### 3.5 Demand Response Scenarios

**FR-DR-1**: The system shall support simulation of pre-conditioning scenarios:
- Pre-cool: Lower cooling setpoint before peak period, then relax during peak
- Pre-heat: Raise heating setpoint before peak period, then relax during peak

**FR-DR-2**: The system shall support simulation of setback scenarios:
- Cooling setback: Raise cooling setpoint during peak period
- Heating setback: Lower heating setpoint during peak period

**FR-DR-3**: The system shall support simulation of price-responsive scenarios:
- Provide time-varying electricity price signal
- Allow policies to adjust setpoints based on price

**FR-DR-4**: The system shall allow configuration of:
- Peak period start and end times
- Pre-conditioning lead time
- Setback magnitude (°F)
- Price signal time series

### 3.6 Policy Evaluation

**FR-PE-1**: The system shall compute energy metrics:
- Total energy consumption (kWh)
- Peak power demand (kW)
- Peak-to-average ratio
- Energy during peak period (kWh)
- Energy shifted from peak to off-peak (kWh)

**FR-PE-2**: The system shall compute comfort metrics:
- Hours with temperature outside comfort bounds
- Degree-hours of discomfort
- Maximum temperature deviation from setpoint

**FR-PE-3**: The system shall compute cost metrics:
- Total electricity cost (given price signal)
- Cost during peak period
- Cost savings vs. baseline

**FR-PE-4**: The system shall support comparison of multiple policies over the same scenarios.

**FR-PE-5**: The system shall allow batch evaluation across:
- Multiple buildings
- Multiple weather conditions
- Multiple DR event configurations

### 3.7 Baseline Policies

**FR-BP-1**: The system shall include a baseline (no-DR) policy that maintains constant setpoints.

**FR-BP-2**: The system shall include a pre-conditioning policy with configurable:
- Pre-conditioning offset (°F)
- Pre-conditioning duration (hours)
- Peak period definition

**FR-BP-3**: The system shall include a setback policy with configurable:
- Setback magnitude (°F)
- Setback period definition

**FR-BP-4**: The system shall include a price-response policy with configurable:
- Price thresholds
- Setpoint adjustments per threshold

---

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-P-1**: Simulation speed: The system shall simulate at least 1 day of building operation within 60 seconds on a standard workstation.

### 4.2 Usability

**NFR-U-1**: Users shall be able to run a basic simulation with fewer than 10 lines of code.

**NFR-U-2**: The system shall include usage examples for common workflows.

### 4.3 Compatibility

**NFR-C-1**: The system shall run on macOS, Linux, and Windows.

**NFR-C-2**: The system shall be compatible with Python 3.9+.

**NFR-C-3**: Generated training data shall be directly usable with the existing thermal prediction pipeline.

### 4.4 Extensibility

**NFR-E-1**: Users shall be able to add custom building models (IDF files).

**NFR-E-2**: Users shall be able to implement custom policies using a documented interface.

**NFR-E-3**: Users shall be able to add custom weather files.

---

## 5. External Interface Requirements

### 5.1 User Interface

The system shall provide a Python API. No graphical user interface is required.

### 5.2 Hardware Interfaces

None. The system runs on standard compute hardware.

### 5.3 Software Interfaces

| External System | Interface | Purpose |
|-----------------|-----------|---------|
| EnergyPlus | Python API (pyenergyplus) | Thermal simulation engine |
| Weather files | EPW format | Weather input data |
| Building models | IDF format | Building definitions |
| Training data | Parquet format | Output compatibility |

### 5.4 Communication Interfaces

None. The system operates locally.

---

## 6. Data Requirements

### 6.1 Input Data

| Data | Format | Source |
|------|--------|--------|
| Building models | EnergyPlus IDF | ResStock / bundled |
| Weather data | EPW | EnergyPlus weather database / bundled |
| Electricity prices | CSV or in-code | User-provided or bundled defaults |

### 6.2 Output Data

| Data | Format | Description |
|------|--------|-------------|
| Training episodes | Parquet | Matches setpoint_responses.parquet schema |
| Policy evaluation results | CSV/DataFrame | Metrics per policy per scenario |
| Simulation time series | CSV/DataFrame | Full timestep-level data |

---

## 7. Design Decisions

### 7.1 Occupancy Modeling

**Decision**: Use fixed schedules from ResStock.

**Rationale**:
- ResStock IDFs already include occupancy schedules with internal heat gains
- Dynamic occupancy adds complexity without clear benefit for DR policy testing
- Primary thermal drivers are weather and HVAC, not occupancy variation

### 7.2 Multi-Zone Buildings

**Decision**: Single-zone only for initial release.

**Rationale**:
- Most ResStock models use single-zone or simplified multi-zone representations
- Single-zone is sufficient for thermostat setpoint control (real thermostats control one zone)
- Dramatically simpler implementation and faster simulation
- Multi-zone support may be added in a future release if needed

### 7.3 HVAC Staging

**Decision**: Single-stage (on/off) equipment for initial release.

**Rationale**:
- Single-stage is still common in residential stock, especially older homes
- Produces more pronounced cycling behavior, which is realistic for DR studies
- ResStock includes both; building library will filter to single-stage archetypes
- Variable-speed equipment may be added in a future release

**Limitation**: Results may be less representative for newer high-efficiency homes with variable-speed heat pumps.

### 7.4 Warm-Up Handling

**Decision**: Automatically run a hidden 7-day warm-up period before each episode.

**Rationale**:
- EnergyPlus requires warm-up to initialize thermal mass state
- Warm-up is transparent to the user; they specify episode start date
- System internally runs simulation from (start - 7 days)
- Warm-up period is excluded from output data
- For batch generation, warm-up can be reused across episodes starting on the same day

### 7.5 Humidity

**Decision**: Exclude humidity from initial release.

**Rationale**:
- Ecobee setpoint_responses.parquet does not include humidity
- Humidity control is secondary for most DR applications; temperature is primary
- Latent loads complicate HVAC modeling significantly
- May be added as optional observation in future release

**Limitation**: Results may be less accurate for hot-humid climates (IECC Zone 2A) where latent loads are significant.

---

## Appendix A: Setpoint Response Schema

To ensure compatibility with existing training pipelines, generated episodes shall include:

```
home_id              : string   # Building identifier
episode_id           : int      # Unique episode identifier
mode                 : string   # 'heat_increase' or 'cool_decrease'
timestep             : int      # Timestep within episode (0, 1, 2, ...)
timestamp            : datetime # Absolute timestamp
indoor_temp          : float    # Indoor temperature (°F)
outdoor_temp         : float    # Outdoor temperature (°F)
setpoint             : float    # Active setpoint (heating or cooling) (°F)
hvac_power_kw        : float    # HVAC electric power
hvac_on              : bool     # HVAC running status
hour                 : int      # Hour of day (0-23)
month                : int      # Month (1-12)
time_to_target_min   : float    # Minutes to reach setpoint (first row only)
```

---

## Appendix B: Comparison with CityLearn

| Requirement | CityLearn | ThermalGym |
|-------------|-----------|------------|
| Thermal model | LSTM (learned) | EnergyPlus (physics) |
| Control interface | Power fraction [0,1] | Setpoint temperature |
| HVAC behavior | Instantaneous response | Thermostat cycling |
| Building source | Pre-trained per building | ResStock archetypes |
| Training data | Not primary purpose | Core feature |
| DR evaluation | Supported | Supported |
