import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from generate_flightpath import generate_flightpath
import os
from multiprocessing import Pool, cpu_count
from geographiclib.geodesic import Geodesic


# Prepare cruise altitude and speed lookup from performance_and_emissions_model
perf_df = pd.read_pickle('performance_and_emissions_model.pkl').set_index('typecode')[0:1000]
SECONDS_PER_MONTH = 31 * 24 * 3600  # January
REMOVAL_TIMESCALE_S = 2 * 24 * 3600  # 2 days

# Function to get cruise parameters
def get_cruise_params(typecode):
        try:
            cruise_alt_ft = perf_df.loc[typecode, 'cruise_Ceiling'] * 100 if not pd.isnull(perf_df.loc[typecode, 'cruise_Ceiling']) else 35000
            cruise_speed_ms = perf_df.loc[typecode, 'cruise_TAS'] * 0.514444 if not pd.isnull(perf_df.loc[typecode, 'cruise_TAS']) else 250
            if cruise_alt_ft <= 0 or np.isnan(cruise_alt_ft):
                cruise_alt_ft = 35000
            if cruise_speed_ms <= 0 or np.isnan(cruise_speed_ms):
                cruise_speed_ms = 250
            return cruise_alt_ft, cruise_speed_ms
        except Exception:
            return 35000, 250
    # Prepare arguments for each process
def process_flight(args):
        row, xgb_models, perf_df, lat_bins, lon_bins, alt_bins_ft, nlat, nlon, nalt = args
        typecode = row['typecode']
        model = xgb_models.get(typecode)
        if model is None:
            return []
        try:
            cruise_alt_ft, cruise_speed_ms = get_cruise_params(typecode)
            fp = generate_flightpath(typecode, row['gc_FEAT_km'], None)
            cruise_alt_ft = fp.get('cruise', {}).get('cruise_altitude_ft', cruise_alt_ft)
        except Exception:
            cruise_alt_ft, cruise_speed_ms = get_cruise_params(typecode)
        features = np.array([[row['gc_FEAT_km'], cruise_alt_ft]])
        mean_nox_flux = model.predict(features)[0]
        cruise_distance_m = row['gc_FEAT_km'] * 1000
        cruise_time_s = cruise_distance_m / cruise_speed_ms
        total_nox_g = mean_nox_flux * cruise_time_s
        total_nox_kg = total_nox_g / 1000

        n_segments = int(np.ceil(cruise_distance_m / 10000))
        geod = Geodesic.WGS84
        line = geod.InverseLine(row['estdeparturelat'], row['estdeparturelong'],
                                row['estarrivallat'], row['estarrivallong'])
        ds = cruise_distance_m / n_segments
        lats, lons = [], []
        for i in range(n_segments):
            s = min(ds * i, line.s13)
            pos = line.Position(s)
            lats.append(pos['lat2'])
            lons.append(pos['lon2'])
        alts = np.full(n_segments, cruise_alt_ft)

        box_fraction = REMOVAL_TIMESCALE_S / (SECONDS_PER_MONTH + REMOVAL_TIMESCALE_S)
        nox_per_segment = total_nox_kg / n_segments * box_fraction

        updates = []
        for i in range(n_segments):
            lat_idx = np.searchsorted(lat_bins, lats[i], side='right') - 1
            lon_idx = np.searchsorted(lon_bins, lons[i], side='right') - 1
            alt_idx = np.searchsorted(alt_bins_ft, alts[i], side='right') - 1
            if 0 <= lat_idx < nlat and 0 <= lon_idx < nlon and 0 <= alt_idx < nalt:
                updates.append((lat_idx, lon_idx, alt_idx, nox_per_segment))
        return updates

def process_month_emissions(month_start_time_str: str,
                  output_dir: str = "/scratch/omg28/Data/no_track2023/emissions/",
                  performance_and_emissions_model: pd.DataFrame = pd.read_pickle('performance_and_emissions_model.pkl')):

    start_time_str_loop = pd.to_datetime(month_start_time_str)
    stop_time_str_loop = (start_time_str_loop + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
    start_time_simple_loop = pd.to_datetime(start_time_str_loop).strftime("%Y-%m-%d")
    stop_time_simple_loop = pd.to_datetime(stop_time_str_loop).strftime("%Y-%m-%d")

    # Load flights data
    monthly_flights = pd.read_pickle(f'{output_dir}/{start_time_simple_loop}_to_{stop_time_simple_loop}_filtered.pkl')[0:10000]
    model_dir = 'saved_models_nox_flux'
    typecodes = monthly_flights['typecode'].unique()

    # Load all xgboost models into memory for speed
    xgb_models = {}
    for typecode in typecodes:
        model_path = os.path.join(model_dir, f'xgb_{typecode}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                xgb_models[typecode] = pickle.load(f)
   

    # Define grid
    lat_bins = np.arange(-90, 90.1, 0.5)
    lon_bins = np.arange(-180, 180.1, 0.5)
    alt_bins_ft = np.arange(0, 55001, 1000)
    alt_bins_m = alt_bins_ft * 0.3048
    nlat, nlon, nalt = len(lat_bins)-1, len(lon_bins)-1, len(alt_bins_m)-1
    nox_grid = np.zeros((nlat, nlon, nalt), dtype=np.float64)





    # Prepare arguments for pool
    pool_args = [
        (row, xgb_models, perf_df, lat_bins, lon_bins, alt_bins_ft, nlat, nlon, nalt)
        for _, row in monthly_flights.iterrows()
    ]

    # Multiprocessing pool
    with Pool(processes=round(cpu_count()/2)) as pool:
        results = list(pool.imap_unordered(process_flight, pool_args))

    # Aggregate results
    for updates in results:
        for lat_idx, lon_idx, alt_idx, nox in updates:
            nox_grid[lat_idx, lon_idx, alt_idx] += nox

    # Optionally: Save as NetCDF or CSV for further analysis
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(f'{output_dir}/emissions', exist_ok=True)
    filename = os.path.join(output_dir, f'emissions/{start_time_simple_loop}_to_{stop_time_simple_loop}_NOx_nowar.npy')
    np.save(filename, nox_grid)
    return filename
