import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from geographiclib.geodesic import Geodesic
import os
from multiprocessing import Pool, cpu_count

# User Inputs:
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


def label_conflict(flight_df: pd.DataFrame,
                   conflict_countries: list = ['Russia', 'Ukraine', 'Libya', 'Syria', 'Sudan', 'Yemen'],
                   buffer_degrees: float = 1.0):
    """
    Label flights that cross conflict airspace.
    
    Parameters:
    flight_df: DataFrame with flight data
    conflict_countries: list of country names (default: ['Russia', 'Ukraine', 'Libya', 'Syria', 'Sudan', 'Yemen'])
    buffer_degrees: buffer size in degrees around conflict countries (default: 1.0)
    """

    # Load country boundaries
    world = gpd.read_file('ne_110m_admin_0_countries.zip')
    name_col = 'NAME' if 'NAME' in world.columns else 'name'

    # Get geometries for conflict countries
    conflict_geometries = []
    for country in conflict_countries:
        country_geom = world[world[name_col] == country].geometry
        if not country_geom.empty:
            conflict_geometries.append(country_geom.union_all().buffer(buffer_degrees))
    
    # Create a union of all conflict areas
    if conflict_geometries:
        conflict_areas = conflict_geometries[0]
        for geom in conflict_geometries[1:]:
            conflict_areas = conflict_areas.union(geom)
        conflict_areas_buffered = conflict_areas.buffer(buffer_degrees)
    else:
        # If no valid countries found, create empty geometry
        conflict_areas_buffered = Point(0, 0).buffer(0)

    def interpolate_great_circle(lon1, lat1, lon2, lat2, num=100):
        """
        Interpolate points along a great circle path using geographiclib.
        """
        geod = Geodesic.WGS84
        line = geod.InverseLine(lat1, lon1, lat2, lon2)
        
        points = []
        for i in range(num):
            s = i * line.s13 / (num - 1)
            g = line.Position(s)
            points.append((g['lon2'], g['lat2']))
        
        return points

    def crosses_conflict_airspace(row):
        lon1, lat1 = row['estdeparturelong'], row['estdeparturelat']
        lon2, lat2 = row['estarrivallong'], row['estarrivallat']
        path_points = interpolate_great_circle(lon1, lat1, lon2, lat2)
        line = LineString([Point(lon, lat) for lon, lat in path_points])
        return line.intersects(conflict_areas_buffered)
    
    # Apply the conflict labeling to the dataframe
    flight_df['crosses_conflict'] = flight_df.apply(crosses_conflict_airspace, axis=1)
    
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
    flights = pd.read_pickle(filepath)[0:100]
    flights_labeled = label_conflict(flights)
    
    # Save the labeled flights
    output_filepath = f'/scratch/omg28/Data/{start_time_simple_loop}_to_{stop_time_simple_loop}_labeled.pkl'
    flights_labeled.to_pickle(output_filepath)
    
    return output_filepath
# Extract month start dates from the previously defined month_args
month_starts = pd.date_range(start=pd.to_datetime(start_time_str), end=pd.to_datetime(stop_time_str), freq='MS', tz='UTC')

# Process all months in parallel
with Pool(cpu_count() // 2) as pool:
    results = pool.map(process_month_conflict, month_starts)

print("Conflict labeling completed for all months")