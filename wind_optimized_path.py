import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, shape
import geopandas as gpd
from geopy.distance import great_circle
def label_conflict(flight_df: pd.DataFrame):

    flight_df['conflict'] = flight_df.apply(crosses_conflict_airspace, axis=1)
    return flight_df
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    russia = world[world['name'] == 'Russia'].geometry.unary_union
    ukraine = world[world['name'] == 'Ukraine'].geometry.unary_union

    def interpolate_great_circle(lon1, lat1, lon2, lat2, num=100):
        # Interpolate points along the great circle
        lons = np.linspace(lon1, lon2, num)
        lats = np.linspace(lat1, lat2, num)
        return list(zip(lons, lats))

    def crosses_conflict_airspace(row):
        lon1, lat1 = row['origin_lon'], row['origin_lat']
        lon2, lat2 = row['dest_lon'], row['dest_lat']
        path_points = interpolate_great_circle(lon1, lat1, lon2, lat2)
        line = LineString([Point(lon, lat) for lon, lat in path_points])
        if line.intersects(russia) or line.intersects(ukraine):
            return True
        return False

    monthly_flights['conflict'] = monthly_flights.apply(crosses_conflict_airspace, axis=1)
