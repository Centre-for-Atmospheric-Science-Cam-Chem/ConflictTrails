import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, shape
import geopandas as gpd
from geopy.distance import great_circle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

start_time_str       = '2023-01-01T00:00:00Z'
stop_time_str        = '2023-12-31T23:59:59Z'
query_limit          = 15e4
send_notification    = True
make_plot            = True
output_dir           = "/scratch/omg28/Data/"

# Convert start and stop times to datetime objects
start_time_simple = pd.to_datetime(start_time_str).strftime("%Y-%m-%d")
stop_time_simple = pd.to_datetime(stop_time_str).strftime("%Y-%m-%d")
analysis_year = pd.to_datetime(start_time_str).year

# Define grid
lat_bins = np.arange(-90, 90.1, 0.5)
lon_bins = np.arange(-180, 180.1, 0.5)
alt_bins_ft = np.arange(0, 55001, 1000)
alt_bins_m = alt_bins_ft * 0.3048
nlat, nlon, nalt = len(lat_bins)-1, len(lon_bins)-1, len(alt_bins_m)-1

def label_conflict(flight_df: pd.DataFrame):

    # Load country boundaries for Russia and Ukraine
    # Download the shapefile from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    # and set the correct path below

    world = gpd.read_file('ne_110m_admin_0_countries.zip')
    # Some versions use 'NAME', others use 'name'
    name_col = 'NAME' if 'NAME' in world.columns else 'name'
    russia = world[world[name_col] == 'Russia'].geometry.unary_union
    ukraine = world[world[name_col] == 'Ukraine'].geometry.unary_union

    def interpolate_great_circle(lon1, lat1, lon2, lat2, num=100):
        # Interpolate points along the great circle
        lons = np.linspace(lon1, lon2, num)
        lats = np.linspace(lat1, lat2, num)
        return list(zip(lons, lats))

    def crosses_conflict_airspace(row):
        lon1, lat1 = row['estdeparturelong'], row['estdeparturelat']
        lon2, lat2 = row['estarrivallong'], row['estarrivallat']
        path_points = interpolate_great_circle(lon1, lat1, lon2, lat2)
        line = LineString([Point(lon, lat) for lon, lat in path_points])
        if line.intersects(russia) or line.intersects(ukraine):
            return True
        return False
    
    # Create a DataFrame for monthly flights with progress bar
    tqdm.pandas(desc="Processing flights")
    flight_df['conflict'] = flight_df.progress_apply(crosses_conflict_airspace, axis=1)
    return flight_df

def process_month_conflict(start_time_str_loop):
    stop_time_str_loop = (start_time_str_loop + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
    start_time_simple_loop = pd.to_datetime(start_time_str_loop).strftime("%Y-%m-%d")
    stop_time_simple_loop = pd.to_datetime(stop_time_str_loop).strftime("%Y-%m-%d")
    
    filepath = f'/scratch/omg28/Data/{start_time_simple_loop}_to_{stop_time_simple_loop}_filtered.pkl'
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    print(f"Processing conflict labeling for {start_time_simple_loop} to {stop_time_simple_loop}")
    flights = pd.read_pickle(filepath)
    flights_labeled = label_conflict(flights)
    
    # Save the labeled flights
    output_filepath = f'/scratch/omg28/Data/{start_time_simple_loop}_to_{stop_time_simple_loop}_filtered_conflict.pkl'
    flights_labeled.to_pickle(output_filepath)
    
    return output_filepath

# Process all months in parallel
month_starts = pd.date_range(start=pd.to_datetime(start_time_str), end=pd.to_datetime(stop_time_str), freq='MS', tz='UTC')

with Pool(cpu_count() // 2) as pool:
    results = pool.map(process_month_conflict, month_starts)

print("Conflict labeling completed for all months")
print("Results:", [r for r in results if r is not None])