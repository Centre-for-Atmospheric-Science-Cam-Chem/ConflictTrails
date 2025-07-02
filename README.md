# ConflictTrails

An Emissions processing toolchain for calculating the impact of conflict on aviation emissions

## Overview

ConflictTrails integrates flight trajectory optimization from the FlightTrajectories submodule with advanced aircraft performance modeling and emissions analysis.

## Features

- **Flight Trajectory Optimization**: Based on the Zermelo method for optimal routing
- **Aircraft Performance Analysis**: Comprehensive modeling using ICAO engine data
- **Emissions Calculation**: Driver for generating detailed emissions profiles
- **Parallel Processing**: Benchmarking tools for serial vs parallel computation
- **Data Processing**: Tools for handling large-scale flight trajectory datasets

## Installation

### Prerequisites

```bash
conda install GEOS cartopy
```

### Install FlightTrajectories submodule

```bash
pip install ./FlightTrajectories/
```

### Additional Dependencies

Install required Python packages for the main project (see `environment.yml` for full environment setup).

## Key Components

### Core Scripts

- `driver.ipynb` - Main driver notebook for emissions analysis
- `driver_emissions_generator.py` - Automated emissions calculation pipeline
- `benchmark_performance.py` - Performance analysis tools
- `benchmark_serial_vs_parallel.py` - Parallel processing benchmarks

### Data Processing

- `download_day.py` / `download_day_script.py` - Flight data acquisition
- `bffm2.py` - Advanced flight modeling algorithms
- `boucher_et_al.py` - Implementation of Boucher et al. methodology

### Utilities

- `ft_to_hPa.py` - Altitude to pressure conversion
- `ft_to_m.py` - Imperial to metric conversion
- `ftmin_to_ms.py` - Flight speed unit conversion

### Data Files

- `aircraft_performance_table.csv` - Aircraft performance database
- `engine_data_icao.csv` - ICAO engine specifications
- Various `.pkl` files for processed performance data

## Usage

1. **Basic Trajectory Analysis**:
   ```python
   # See driver.ipynb for comprehensive examples
   ```

2. **Emissions Generation**:
   ```bash
   python driver_emissions_generator.py
   ```

3. **Performance Benchmarking**:
   ```bash
   python benchmark_performance.py
   ```

## FlightTrajectories Integration

This project incorporates the [FlightTrajectories](FlightTrajectories/) submodule for trajectory optimization. See [FlightTrajectories/README.md](FlightTrajectories/README.md) for specific documentation on:

- Trajectory optimization using the Zermelo method
- IAGOS flight data processing
- ERA5 meteorological data integration

## License

- Main project: [Specify your license]
- FlightTrajectories submodule: GNU General Public License v3.0

## Authors

- [Your name/organization]
- FlightTrajectories: Olivier Boucher, Ed Gryspeerdt & Julien Karadayi

## Contributing

[Add contribution guidelines if applicable]

## Citation

If you use this work in your research, please cite:

[Add citation information]
