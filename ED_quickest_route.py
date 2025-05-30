import os
import sys
proj_path1 = os.path.abspath(os.path.join("FlightTrajectories"))
proj_path2 = os.path.abspath(os.path.join("FlightTrajectories/FlightTrajectories"))
if proj_path1 not in sys.path:
  sys.path.append(proj_path1)
if proj_path2 not in sys.path:
  sys.path.append(proj_path2)
from FlightTrajectories.optimalrouting import ZermeloLonLat
from FlightTrajectories.misc_geo import nearest
from FlightTrajectories.minimization import cost_time, wind
import numpy as np
import time
import xarray as xr
import netCDF4
from sklearn.metrics import mean_squared_error
import pandas as pd
from geographiclib.geodesic import Geodesic
import bisect
from ft_to_hPa import ft_to_hPa
import numpy as np
import pandas as pd
import pickle
from generate_flightpath import generate_flightpath
import os
from multiprocessing import Pool, cpu_count
from geographiclib.geodesic import Geodesic
from xgboost import XGBRegressor
# This script computes the quickest route between two points using the Zermelo method

SECONDS_PER_MONTH = 31 * 24 * 3600  # January
REMOVAL_TIMESCALE_S = 2 * 24 * 3600  # 2 days

# From the ipynb driver, user input: ###################################
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

# Define grid
lat_bins = np.arange(-90, 90.1, 0.5)
lon_bins = np.arange(-180, 180.1, 0.5)
alt_bins_ft = np.arange(0, 55001, 1000)
alt_bins_m = alt_bins_ft * 0.3048
nlat, nlon, nalt = len(lat_bins)-1, len(lon_bins)-1, len(alt_bins_m)-1
########################################################################
era5_file = f"/scratch/omg28/Data/winddb/era5_wind_{analysis_year}.nc"
ds_era5 = xr.open_dataset(era5_file).load()
print(ds_era5)

# cruise_alt_ft = 35000  # feet example value
# cruise_speed_ms = 250  # m/s, example value
#################################################################



def get_cruise_params(typecode, perf_df):
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

def process_flight(args):
    row, xgb_models, perf_df, lat_bins, lon_bins, alt_bins_ft, nlat, nlon, nalt, xr_u, xr_v, lons_wind, lats_wind, levels_wind = args
    typecode = row['typecode']
    model = xgb_models.get(typecode)
    if model is None:
      return []
    
    # Check if the flight crosses conflict airspace
    if row['crosses_conflict']:
        # Get cruise altitude and speed from performance data
        cruise_alt_ft, cruise_speed_ms = get_cruise_params(typecode, perf_df) # skip generating flight path since distance will change and thus we cannot generate a flight path to get a plausible altitude
      
        #################
        optim_level,pressure_ind_closest=nearest(levels_wind,ft_to_hPa(cruise_alt_ft))
        
        lon_p1, lat_p1 = row['estdeparturelong'], row['estdeparturelat']
        lon_p2, lat_p2 = row['estarrivallong'], row['estarrivallat']
        
        p1 = (lon_p1, lat_p1)  # Departure point
        p2 = (lon_p2, lat_p2)  # Destination point

        airspeed = cruise_speed_ms  # m/s

        dist = row['gc_km'] * 1000  # in meters

        nbmeters = 50000. # Number of meters to split the route into segments


        npoints = int(dist // nbmeters)
        n_segments = int(dist // nbmeters) + 1  # Number of segments for the route

        geod = Geodesic.WGS84
        line = geod.InverseLine(row['estdeparturelat'], row['estdeparturelong'],
                                row['estarrivallat'], row['estarrivallong'])
        ds = dist / n_segments
        lon_shortest, lat_shortest = [], []
        for i in range(n_segments):
            s = min(ds * i, line.s13)
            pos = line.Position(s)
            lat_shortest.append(pos['lat2'])
            lon_shortest.append(pos['lon2'])
        lat_quickest = lat_shortest.copy()  # Assuming lat_quickest is the same as lat_shortest for now, lat_quickest is only used for comparison
        lat_iagos_cruising = lat_shortest.copy()  # Assuming lat_iagos_cruising is the same as lat_shortest for now, used for comparison
        npoints = int(dist // nbmeters)
        
        # Reduce the wind field to the relevant longitudes
        _idx_p1 = bisect.bisect_right(lons_wind, lon_p1) - 1 
        _idx_p2 = bisect.bisect_right(lons_wind, lon_p2) + 1
        print(_idx_p1, _idx_p2)
        
        lons_wind_reduced = lons_wind
        # lons_wind_reduced = lons_wind[_idx_p1:_idx_p2]
        
        lons_z = xr.DataArray(lons_wind_reduced, dims="z")
        
        # this is probably slower than just using the whole grid - this will result in loading in the wind field many times 
        xr_u200=xr_u.sel(pressure_level=optim_level).load()
        xr_u200_reduced=xr_u200.sel(longitude=lons_z, latitude=lats_wind).load()

        xr_v200=xr_v.sel(pressure_level=optim_level).load()
        xr_v200_reduced=xr_v200.sel(longitude=lons_z,latitude=lats_wind).load()
        
        rmse_shortest, rmse_quickest, lat_max_shortest, lat_max_quickest, lon_ed_LD, lat_ed_LD, dt_ed_LD,                   \
                                time_elapsed_EG, lon_ed, lat_ed, dt_ed_HD, solution =                                                 \
                                ED_quickest_route(p1, p2, airspeed, lon_p1, lon_p2, lat_p1, lat_p2,                                   \
                                lat_shortest, lat_quickest, lat_iagos_cruising, lons_wind_reduced, lats_wind, xr_u200_reduced, xr_v200_reduced, npoints)
      #################
        if solution:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            # Create a figure with a globe projection
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(
                central_longitude=(lon_p1 + lon_p2) / 2,
                central_latitude=(lat_p1 + lat_p2) / 2
            ))
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.OCEAN, color='lightblue')
            ax.add_feature(cfeature.LAND, color='lightgray')
            
            # Plot the optimal route
            ax.plot(lon_ed, lat_ed, 'r-', linewidth=2, label='Optimal Route', 
                    transform=ccrs.PlateCarree())
            
            # Plot departure and arrival points
            ax.plot(lon_p1, lat_p1, 'go', markersize=8, label='Departure', 
                    transform=ccrs.PlateCarree())
            ax.plot(lon_p2, lat_p2, 'ro', markersize=8, label='Arrival', 
                    transform=ccrs.PlateCarree())
            
            # Plot the great circle route for comparison
            ax.plot(lon_shortest, lat_shortest, 'b--', linewidth=1, label='Great Circle', 
                    transform=ccrs.PlateCarree())
            
            ax.legend()
            ax.set_title(f'Flight Path Optimization\nFlight Time: {dt_ed_HD:.2f} hours')
            plt.tight_layout()
            plt.show()
            plt.close()
            return[]

    else:
        try:
            cruise_alt_ft, cruise_speed_ms = get_cruise_params(typecode, perf_df)
            fp = generate_flightpath(typecode, row['gc_FEAT_km'], None)
            cruise_alt_ft = fp.get('cruise', {}).get('cruise_altitude_ft', cruise_alt_ft)
        except Exception:
            cruise_alt_ft, cruise_speed_ms = get_cruise_params(typecode, perf_df)
        # Handle non-conflict case - use geodesic for cruise path
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


def process_month_emissions_conflict(
    month_start_time_str: str,
    output_dir: str = "/scratch/omg28/Data/no_track2023/emissions/",
    performance_and_emissions_model: pd.DataFrame = pd.read_pickle('performance_and_emissions_model.pkl')
):
    start_time_str_loop = pd.to_datetime(month_start_time_str)
    stop_time_str_loop = (start_time_str_loop + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
    start_time_simple_loop = pd.to_datetime(start_time_str_loop).strftime("%Y-%m-%d")
    stop_time_simple_loop = pd.to_datetime(stop_time_str_loop).strftime("%Y-%m-%d")

    # Load flights data
    monthly_flights = pd.read_pickle(f'{output_dir}/{start_time_simple_loop}_to_{stop_time_simple_loop}_labeled.pkl').iloc[[1132522, 1052457, 783169, 907852, 783170]]
    model_dir = 'saved_models_nox_flux'
    typecodes = monthly_flights['typecode'].unique()

    # load wind data
    era5_month = ds_era5.sel(valid_time=start_time_simple_loop, method='nearest')

    xr_u=era5_month.drop_vars('v')
    xr_v=era5_month.drop_vars('u')
    
    #--Extract coordinates
    levels_wind=list(xr_u['pressure_level'].values)
    lons_wind=xr_u['longitude'].values
    lats_wind=xr_u['latitude'].values
    
    # Load all xgboost models into memory for speed
    xgb_models = {}
    for typecode in typecodes:
        model_path = os.path.join(model_dir, f'xgb_{typecode}.ubj')
        if os.path.exists(model_path):
            model = XGBRegressor()
            model.load_model(model_path)
            xgb_models[typecode] = model
            
    # Prepare cruise altitude and speed lookup from performance_and_emissions_model
    perf_df = performance_and_emissions_model.set_index('typecode')

    # Define grid
    lat_bins = np.arange(-90, 90.1, 0.5)
    lon_bins = np.arange(-180, 180.1, 0.5)
    alt_bins_ft = np.arange(0, 55001, 1000)
    alt_bins_m = alt_bins_ft * 0.3048
    nlat, nlon, nalt = len(lat_bins)-1, len(lon_bins)-1, len(alt_bins_m)-1
    nox_grid = np.zeros((nlat, nlon, nalt), dtype=np.float64)

      # Prepare arguments for loop
    pool_args = [
        (row, xgb_models, perf_df, lat_bins, lon_bins, alt_bins_ft, nlat, nlon, nalt, xr_u, xr_v, lons_wind, lats_wind, levels_wind)
        for _, row in monthly_flights.iterrows()
    ]

    # Process flights and aggregate results
    for args in pool_args:
        updates = process_flight(args)
        for lat_idx, lon_idx, alt_idx, nox in updates:
            nox_grid[lat_idx, lon_idx, alt_idx] += nox
          
    # Optionally: Save as NetCDF or CSV for further analysis
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(f'{output_dir}/emissions', exist_ok=True)
    filename = os.path.join(output_dir, f'emissions/{start_time_simple_loop}_to_{stop_time_simple_loop}_NOx_nowar.npy')
    # np.save(filename, nox_grid)
    return filename
  

def ED_quickest_route(p1, p2, airspeed, lon_p1, lon_p2, lat_p1, lat_p2, 
                      lat_shortest, lat_quickest, lat_iagos_cruising, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, npoints):
    
    # p1: (lon_p1, lat_p1)
    start_time = time.time()
    # Create the zermelo solver. Note the default values
    #--max_dest_dist is in metres
    #--sub_factor: number of splits for next round if solution is bounded by pairs of trajectories
    #--psi_range: +/- angle for the initial bearing
    #--psi_res: resolution within psi_range bounds, could try 0.2 instead
    zermelolonlat = ZermeloLonLat(cost_func=lambda x, y, z: np.ones(np.atleast_1d(x).shape),
                                  wind_func=wind, timestep=60, psi_range=60, psi_res=0.5,
                                  length_factor=1.4, max_dest_distance=75000., sub_factor=80)
    
    initial_psi = zermelolonlat.bearing_func(*p1, *p2)
    psi_vals = np.linspace(initial_psi-60, initial_psi+60, 30)
    #--This prodcues a series of Zermelo trajectories for the given initial directions
    zloc, zpsi, zcost = zermelolonlat.zermelo_path(np.repeat(np.array(p1)[:, None], len(psi_vals), axis=-1),lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, 
                        # This 90 is due to an internal conversion between bearings and angles
                        # - which is obviously a bad idea... noramlly it is hidden internally
                        ##   90-psi_vals, nsteps=800, airspeed=250, dtime=dep_time_iagos)
                        90-psi_vals, nsteps=800, airspeed=airspeed, dtime=0) #--modif OB
    
     # np.repeat(np.array(p1)[:, None] creates many instances of the p1 point for each psi value and passes it to the 'departure location' argument.
     # lons_wind uses 
    
    # This identifies the optimal route
    solution, fpst, ftime, flocs, fcost = zermelolonlat.route_optimise(np.array(p1), np.array(p2),  lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed=airspeed, dtime=0)
    #--if solution was found
    if solution: 
        lon_ed=flocs[:,0]
        lat_ed=flocs[:,1]
        #
        #--compute Ed's time by stretching slightly the trajectory to the same endpoints
        npoints_ed=len(lon_ed)
        print('npoints_ed=',npoints_ed)
        lon_ed=lon_ed+(lon_p2-lon_ed[-1])*np.arange(npoints_ed)/float(npoints_ed-1)
        lat_ed=lat_ed+(lat_p2-lat_ed[-1])*np.arange(npoints_ed)/float(npoints_ed-1)
        #--compute corresponding time 
        dt_ed_HD=cost_time(lon_ed, lat_ed, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, dtprint=False)
        print('Cruising flight time ED (high res) =',"{:6.4f}".format(dt_ed_HD),'hours')
        lon_ed_LD=np.append(lon_ed[::npoints_ed//npoints],[lon_ed[-1]])
        lat_ed_LD=np.append(lat_ed[::npoints_ed//npoints],[lat_ed[-1]])
        dt_ed_LD=cost_time(lon_ed_LD, lat_ed_LD, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, dtprint=False)
        print('Cruising flight time ED (low res) =',"{:6.4f}".format(dt_ed_LD),'hours')
    else: 
        print('No solution found by Zermelo')  
        lon_ed=float('inf')
        lat_ed=float('inf')
        dt_ed_HD=float('inf')
        lon_ed_LD=float('inf')
        lat_ed_LD=float('inf')
        dt_ed_LD=float('inf')
    end_time = time.time()
    time_elapsed_EG=end_time-start_time
    print('Time elapsed for Zermelo method=',"{:3.1f}".format(time_elapsed_EG),'s')
    
    #--computing indices of quality of fit
    rmse_shortest, rmse_quickest, lat_max_shortest, lat_max_quickest = 0, 0, 0, 0
    #rmse_shortest=mean_squared_error(lat_shortest,lat_iagos_cruising)**0.5
    #rmse_quickest=mean_squared_error(lat_quickest,lat_iagos_cruising)**0.5
    #lat_max_shortest=np.max(np.abs(lat_shortest-lat_iagos_cruising))
    #lat_max_quickest=np.max(np.abs(lat_quickest-lat_iagos_cruising))
    print('rmse and lat max=',rmse_shortest,rmse_quickest,lat_max_shortest,lat_max_quickest)
    
    return rmse_shortest, rmse_quickest, lat_max_shortest, lat_max_quickest, lon_ed_LD, lat_ed_LD, dt_ed_LD, time_elapsed_EG, lon_ed, lat_ed, dt_ed_HD, solution




'''
lon_p1, lat_p1 = row['estdeparturelon'], row['estdeparturelat']
lon_p2, lat_p2 = row['estarrivallon'], row['estarrivallat']

p1 = (lon_p1, lat_p1)  # Departure point
p2 = (lon_p2, lat_p2)  # Destination point

airspeed = cruise_speed_ms  # m/s

dist = row['gc_km'] * 1000  # in meters

nbmeters = 50000. # Number of meters to split the route into segments


npoints = int(dist // nbmeters)
n_segments = int(dist // nbmeters) + 1  # Number of segments for the route

geod = Geodesic.WGS84
line = geod.InverseLine(row['estdeparturelat'], row['estdeparturelong'],
                        row['estarrivallat'], row['estarrivallong'])
ds = dist / n_segments
lon_shortest, lat_shortest = [], []
for i in range(n_segments):
    s = min(ds * i, line.s13)
    pos = line.Position(s)
    lat_shortest.append(pos['lat2'])
    lon_shortest.append(pos['lon2'])
lat_quickest = lat_shortest.copy()  # Assuming lat_quickest is the same as lat_shortest for now, lat_quickest is only used for comparison
lat_iagos_cruising = lat_shortest.copy()  # Assuming lat_iagos_cruising is the same as lat_shortest for now, used for comparison

npoints = int(dist // nbmeters)


_idx_p1 = bisect.bisect_right(lons_wind, lon_p1) - 1 
_idx_p2 = bisect.bisect_right(lons_wind, lon_p2) + 1

lons_wind_reduced = lons_wind[_idx_p1:_idx_p2]


rmse_shortest, rmse_quickest, lat_max_shortest, lat_max_quickest, lon_ed_LD, lat_ed_LD, dt_ed_LD,                   \
                      time_elapsed_EG, lon_ed, lat_ed, dt_ed_HD, solution =                                                 \
                      ED_quickest_route(p1, p2, airspeed, lon_p1, lon_p2, lat_p1, lat_p2,                                   \
                      lat_shortest, lat_quickest, lat_iagos_cruising, lons_wind_reduced, lats_wind, xr_u200_reduced, xr_v200_reduced, npoints)
'''
                      
                      
                      