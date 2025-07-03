# ConflictTrails

An emissions processing toolchain for calculating the impact of conflict on aviation emissions.

## Overview

ConflictTrails integrates flight trajectory optimization from the FlightTrajectories submodule with advanced aircraft performance modeling and emissions analysis.


## Dependencies

Install required Python packages for the main project (see `environment.yml` for full environment setup).


## Core Scripts

- `driver.ipynb` - Main driver notebook for emissions analysis

## FlightTrajectories Integration

This project incorporates the [FlightTrajectories](FlightTrajectories/) submodule for trajectory optimization. See [FlightTrajectories/README.md](FlightTrajectories/README.md) for specific documentation on:

- Trajectory optimization using the Zermelo method
- IAGOS flight data processing
- ERA5 meteorological data integration

## License

- Main project: GNU General Public License v3.0
- FlightTrajectories submodule: GNU General Public License v3.0

## Authors

- Owen Graham
- FlightTrajectories: Olivier Boucher, Ed Gryspeerdt & Julien Karadayi
