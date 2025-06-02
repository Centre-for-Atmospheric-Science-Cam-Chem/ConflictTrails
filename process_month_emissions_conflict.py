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
from FlightTrajectories.minimization import cost_time, wind, cost_squared
from scipy.optimize import minimize
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
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.patches import Patch
from telegram_notifier import send_telegram_notification
# This script computes the quickest route between two points using the Zermelo method

SECONDS_PER_MONTH = 31 * 24 * 3600  # January
REMOVAL_TIMESCALE_S = 2 * 24 * 3600  # 2 days
'''
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

# cruise_alt_ft = 35000  # feet example value
# cruise_speed_ms = 250  # m/s, example value
#################################################################
'''
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
    
    # Get common parameters
    cruise_alt_ft, cruise_speed_ms = get_cruise_params(typecode, perf_df)
    
    # Check if the flight crosses conflict airspace
    if row['crosses_conflict']:
        # Do optimal routing for conflict flights
        optim_level, pressure_ind_closest = nearest(levels_wind, ft_to_hPa(cruise_alt_ft))
        
        lon_p1, lat_p1 = row['estdeparturelong'], row['estdeparturelat']
        lon_p2, lat_p2 = row['estarrivallong'], row['estarrivallat']
        
        p1 = (lon_p1, lat_p1)
        p2 = (lon_p2, lat_p2)
        airspeed = cruise_speed_ms
        dist = row['gc_km'] * 1000
        
        # boucher et at al. 2023 use 50km segments
        # we will use 10 segments for all flights to minimize computation time 
        # nbmeters = 500000
        # npoints = int(dist // nbmeters)
        # n_segments = int(dist // nbmeters) + 1
        
        npoints = 12  # Number of points for optimization
        n_segments = npoints + 1  # Number of segments for the optimized route
        
        # Create great circle route as baseline
        geod1 = Geodesic.WGS84
        line = geod1.InverseLine(row['estdeparturelat'], row['estdeparturelong'],
                                row['estarrivallat'], row['estarrivallong'])
        ds = dist / n_segments
        
        lon_shortest = np.zeros(n_segments)
        lat_shortest = np.zeros(n_segments)
        for i in range(n_segments):
            s = min(ds * i, line.s13)
            pos = line.Position(s)
            lat_shortest[i] = pos['lat2']
            lon_shortest[i] = pos['lon2']
        
        # Get wind data
        lons_wind_reduced = lons_wind  # Use full grid for now
        lons_z = xr.DataArray(lons_wind_reduced, dims="z")
        
        xr_u200 = xr_u.sel(pressure_level=optim_level).load()
        xr_u200_reduced = xr_u200.sel(longitude=lons_z, latitude=lats_wind).load()
        xr_v200 = xr_v.sel(pressure_level=optim_level).load()
        xr_v200_reduced = xr_v200.sel(longitude=lons_z, latitude=lats_wind).load()
        
        # Run optimization
        try:
            lat_quickest = lat_shortest.copy()
            lat_iagos_cruising = lat_shortest.copy()
            
            method_i_am_using = 2  # 1 for quickest_route, 2 for quickest_route_fast, 3 for zermelo method

            if method_i_am_using == 1: #FIXME - add condition to choose optimization method
                method = 'SLSQP'
                disp = False # True displays optimization output, False suppresses it
                maxiter = 100
                
                
                lon_quickest, lat_quickest, dt_quickest = quickest_route(p1, p2, npoints, lat_iagos_cruising, lons_wind_reduced, lats_wind,
                                                                        xr_u200_reduced, xr_v200_reduced, airspeed, method, disp, maxiter)
                
                lon_ed_LD = lon_quickest
                lat_ed_LD = lat_quickest
                solution = True
            elif method_i_am_using == 2: #FIXME - add condition to choose optimization method
                method = 'SLSQP'
                disp = False # True displays optimization output, False suppresses it
                maxiter = 100
                nbest = 1 # The performance speedup is made by considering only the best initial guess for optimization.
                # fewer initial guesses means faster optimization, but lower likelihood of finding the global minimum.

                lon_quickest, lat_quickest, dt_quickest = quickest_route_fast(p1, p2, npoints, nbest, lat_iagos_cruising, lons_wind_reduced, lats_wind,
                                                                         xr_u200_reduced, xr_v200_reduced, airspeed, method, disp, maxiter)

                lon_ed_LD = lon_quickest
                lat_ed_LD = lat_quickest
                
                solution = True
            else: # zermelo method
                rmse_shortest, rmse_quickest, lat_max_shortest, lat_max_quickest, lon_ed_LD, lat_ed_LD, dt_ed_LD, \
                time_elapsed_EG, lon_ed, lat_ed, dt_ed_HD, solution = \
                ED_quickest_route(p1, p2, airspeed, lon_p1, lon_p2, lat_p1, lat_p2,
                            lat_shortest, lat_quickest, lat_iagos_cruising, lons_wind_reduced, 
                            lats_wind, xr_u200_reduced, xr_v200_reduced, npoints)
        

            
            if False:
            #if solution:
                # Use optimized route
                lats = lat_ed_LD
                lons = lon_ed_LD
                alts = np.full(len(lats), cruise_alt_ft)
                n_segments = len(lats)
                
                # Calculate cruise distance as sum of great circle distances between successive points
                if len(lats) > 1:
                    # Calculate distances between successive points using a loop
                    geod = Geodesic.WGS84
                    total_distance = 0
                    
                    for i in range(len(lats) - 1):
                        result = geod.Inverse(lats[i], lons[i], lats[i+1], lons[i+1])
                        total_distance += result['s12']
                    
                    cruise_distance_m = total_distance
                    print(cruise_distance_m)
                else:
                    cruise_distance_m = 0

                # Create visualization showing flight paths on 2D globe representation
                import matplotlib.pyplot as plt
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                
                # Create figure with map projection
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                
                # Add map features
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
                ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
                
                # Load and highlight conflict countries
                conflict_countries = ['Russia', 'Ukraine', 'Libya', 'Syria', 'Sudan', 'Yemen']
                try:
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                except (AttributeError, KeyError):
                    world = gpd.read_file('ne_110m_admin_0_countries.zip')
                
                name_col = 'NAME' if 'NAME' in world.columns else 'name'
                conflict_zones = world[world[name_col].isin(conflict_countries)]
                
                # Add conflict zones to map
                for idx, country in conflict_zones.iterrows():
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), 
                                    facecolor='red', alpha=0.3, edgecolor='darkred', 
                                    linewidth=1.5)
                
                # Set map extent based on flight path
                lon_margin = max(10, abs(lon_p2 - lon_p1) * 0.2)
                lat_margin = max(5, abs(lat_p2 - lat_p1) * 0.2)
                ax.set_extent([min(lon_p1, lon_p2) - lon_margin, 
                              max(lon_p1, lon_p2) + lon_margin,
                              min(lat_p1, lat_p2) - lat_margin, 
                              max(lat_p1, lat_p2) + lat_margin], 
                              crs=ccrs.PlateCarree())
                
                # Plot great circle route
                ax.plot([lon_p1, lon_p2], [lat_p1, lat_p2], 
                        color='blue', linewidth=3, label='Great Circle Route', 
                        transform=ccrs.Geodetic(), alpha=0.8)
                
                # Plot optimized route
                ax.plot(lons, lats, 
                        color='green', linewidth=3, label='Wind-Optimized Route', 
                        transform=ccrs.Geodetic(), alpha=0.8)
                
                # Mark start and end points
                ax.plot(lon_p1, lat_p1, 'go', markersize=10, label='Departure', 
                        transform=ccrs.PlateCarree(), markeredgecolor='black', markeredgewidth=1)
                ax.plot(lon_p2, lat_p2, 'ro', markersize=10, label='Arrival', 
                        transform=ccrs.PlateCarree(), markeredgecolor='black', markeredgewidth=1)
                
                # Add gridlines
                ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                            linewidth=0.5, alpha=0.5)
                
                # Create custom legend
                legend_elements = [
                    plt.Line2D([0], [0], color='blue', linewidth=3, label='Great Circle Route'),
                    plt.Line2D([0], [0], color='green', linewidth=3, label='Wind-Optimized Route'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                              markersize=10, label='Departure', markeredgecolor='black'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', 
                              markersize=10, label='Arrival', markeredgecolor='black'),
                    Patch(facecolor='red', alpha=0.3, edgecolor='darkred', label='Conflict Zones')
                ]
                
                # Set title and legend
                distance_saved = (cruise_distance_m/1000) - (row['gc_km'])
                ax.set_title(f'Flight Path Optimization\nDistance: GC={row["gc_km"]:.0f}km, Optimized={cruise_distance_m/1000:.0f}km\nDifference: {distance_saved:.1f}km', 
                           fontsize=12)
                ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
                
                plt.tight_layout()
                plt.show()


            else:
                # Fall back to great circle
                lats = lat_shortest
                lons = lon_shortest  
                alts = np.full(len(lats), cruise_alt_ft)
                n_segments = len(lats)
                cruise_distance_m = row['gc_FEAT_km'] * 1000

        except Exception as e:
            print(f"Optimization failed: {e}")
            # Fall back to great circle
            lats = lat_shortest
            lons = lon_shortest
            alts = np.full(len(lats), cruise_alt_ft)
            n_segments = len(lats)
            cruise_distance_m = row['gc_FEAT_km'] * 1000
            
        # Calculate emissions for conflict case
        features = np.array([[cruise_distance_m/1000, cruise_alt_ft]])
        mean_nox_flux = model.predict(features)[0]
        cruise_time_s = cruise_distance_m / cruise_speed_ms
        total_nox_g = mean_nox_flux * cruise_time_s
        total_nox_kg = total_nox_g / 1000
    else: # Non-conflict case - use great circle route
        
        cruise_distance_m = row['gc_FEAT_km'] * 1000
        
        try: # calculates cruise parameters from flight path if possible.
            fp = generate_flightpath(typecode, row['gc_FEAT_km'], None)
            cruise_alt_ft = fp.get('cruise', {}).get('cruise_altitude_ft', cruise_alt_ft)
        except Exception:
            pass
        
        # Non-conflict case - use geodesic for cruise path
        geod = Geodesic.WGS84
        line = geod.InverseLine(row['estdeparturelat'], row['estdeparturelong'],
                                row['estarrivallat'], row['estarrivallong'])
        n_segments = int(np.ceil(cruise_distance_m / 10000))
        ds = cruise_distance_m / n_segments
        
        lats = np.zeros(n_segments)
        lons = np.zeros(n_segments)
        for i in range(n_segments):
            s = min(ds * i, line.s13)
            pos = line.Position(s)
            lats[i] = pos['lat2']
            lons[i] = pos['lon2']
        alts = np.full(n_segments, cruise_alt_ft)

        # Calculate emissions
        features = np.array([[row['gc_FEAT_km'], cruise_alt_ft]])
        mean_nox_flux = model.predict(features)[0]
        cruise_time_s = cruise_distance_m / cruise_speed_ms
        total_nox_g = mean_nox_flux * cruise_time_s
        total_nox_kg = total_nox_g / 1000

    # Calculate emissions distribution
    box_fraction = REMOVAL_TIMESCALE_S / (SECONDS_PER_MONTH + REMOVAL_TIMESCALE_S)
    nox_per_segment = total_nox_kg / n_segments * box_fraction

    updates = np.zeros((n_segments, 4), dtype=np.float64)
    valid_count = 0
    for i in range(n_segments):
        lat_idx = np.searchsorted(lat_bins, lats[i], side='right') - 1
        lon_idx = np.searchsorted(lon_bins, lons[i], side='right') - 1
        alt_idx = np.searchsorted(alt_bins_ft, alts[i], side='right') - 1
        if 0 <= lat_idx < nlat and 0 <= lon_idx < nlon and 0 <= alt_idx < nalt:
            updates[valid_count] = [lat_idx, lon_idx, alt_idx, nox_per_segment]
            valid_count += 1
    updates = updates[:valid_count]  # Trim to actual valid entries
    
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
    analysis_year = start_time_str_loop.year
    
    # Load flights data
    monthly_flights = pd.read_pickle(f'{output_dir}/{start_time_simple_loop}_to_{stop_time_simple_loop}_labeled.pkl')
    model_dir = 'saved_models_nox_flux'
    typecodes = monthly_flights['typecode'].unique()

    # load wind data
    era5_file = f"/scratch/omg28/Data/winddb/era5_wind_{analysis_year}.nc"
    ds_era5 = xr.open_dataset(era5_file).load()
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
    total_flights = len(pool_args)
    for i, args in enumerate(pool_args):
        updates = process_flight(args)
        for lat_idx, lon_idx, alt_idx, nox in updates:
            nox_grid[int(lat_idx), int(lon_idx), int(alt_idx)] += nox
        
        # Progress update every 100,000 flights
        if (i + 1) % 100000 == 0:
            progress_pct = ((i + 1) / total_flights) * 100
            print(f"Progress: {i + 1:,}/{total_flights:,} flights processed ({progress_pct:.1f}%)")
            
            # Send telegram notification
            try:
                send_telegram_notification(f"Progress update: {i + 1:,}/{total_flights:,} flights processed ({progress_pct:.1f}%)")
            except Exception as e:
                print(f"Failed to send telegram notification: {e}")
          
    # Optionally: Save as NetCDF or CSV for further analysis
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(f'{output_dir}/emissions', exist_ok=True)
    filename = os.path.join(output_dir, f'emissions/{start_time_simple_loop}_to_{stop_time_simple_loop}_NOx_war.npy')
    np.save(filename, nox_grid)
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

def quickest_route(dep_loc, arr_loc, npoints, lat_iagos, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, method, disp, maxiter):
    """Compute the quickest route from dep_loc to arr_loc"""
    
    # Fix coordinate order - ensure consistent lon, lat format
    if len(dep_loc) == 2 and len(arr_loc) == 2:
        lon_dep, lat_dep = dep_loc
        lon_arr, lat_arr = arr_loc
    else:
        raise ValueError("dep_loc and arr_loc must be (lon, lat) tuples")
    
    # Bounds for latitude optimization (more restrictive near poles)
    lat_min = max(-89.9, min(lat_dep, lat_arr) - 20)
    lat_max = min(89.9, max(lat_dep, lat_arr) + 20)
    bnds = tuple((lat_min, lat_max) for i in range(npoints))
    
    def shortest_route(dep_loc, arr_loc, npoints):
        """Compute great circle route using proper geodesic calculations"""
        lon_dep, lat_dep = dep_loc
        lon_arr, lat_arr = arr_loc
        
        # Use geodesic for proper great circle calculation
        geod = Geodesic.WGS84
        line = geod.InverseLine(lat_dep, lon_dep, lat_arr, lon_arr)  # Note: lat, lon for geod
        
        # Calculate points along the great circle
        n_segments = npoints
        lons = np.zeros(n_segments)
        lats = np.zeros(n_segments)
        
        for i in range(n_segments):
            s = line.s13 * i / (n_segments - 1)  # Distance along the line
            pos = line.Position(s)
            lats[i] = pos['lat2']
            lons[i] = pos['lon2']
        
        return lons, lats

    x0, y0 = shortest_route(dep_loc, arr_loc, npoints)
    
    # Initial optimization
    res = minimize(cost_squared, y0[1:-1], 
                   args=(x0[1:-1], lon_dep, lat_dep, lon_arr, lat_arr, 
                         lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed),
                   method=method, bounds=bnds[1:-1], 
                   options={'maxiter': maxiter, 'disp': disp})
    
    y = np.append(np.insert(res['x'], 0, lat_dep), lat_arr)
    quickest_time = cost_time(x0, y, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, dtprint=False)
    
    # Multiple optimization attempts with latitude shifts
    # Scale shifts based on the flight distance to avoid excessive deviations
    flight_distance_deg = np.sqrt((lon_arr - lon_dep)**2 + (lat_arr - lat_dep)**2)
    max_shift = min(15.0, flight_distance_deg * 0.3)  # Adaptive maximum shift
     
    n = len(x0)
    shift_values = np.linspace(-max_shift, max_shift, 15)  # More reasonable shifts
    
    for dymax in shift_values:
        if abs(dymax) < 1e-6:  # Skip near-zero shifts
            continue
            
        for imid in [n//2, n//3, 2*n//3]:
            if imid <= 0 or imid >= n:
                continue
                
            # Create latitude shift profile
            dy = np.zeros(n)
            for i in range(imid):
                dy[i] = dymax * float(i) / float(imid)
            for i in range(imid, n):
                dy[i] = dymax * float(n - i) / float(n - imid)
            
            y0p = y0 + dy
            
            # Ensure shifted points are within bounds
            y0p = np.clip(y0p, lat_min, lat_max)
            
            try:
                res = minimize(cost_squared, y0p[1:-1],
                             args=(x0[1:-1], lon_dep, lat_dep, lon_arr, lat_arr,
                                   lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed),
                             method=method, bounds=bnds[1:-1],
                             options={'maxiter': maxiter, 'disp': disp})
                
                y_2 = np.append(np.insert(res['x'], 0, lat_dep), lat_arr)
                quickest_time_2 = cost_time(x0, y_2, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed)
                
                if quickest_time_2 < quickest_time:
                    quickest_time = quickest_time_2
                    y = y_2
                    
            except Exception as e:
                print(f"Optimization failed for shift {dymax}: {e}")
                continue
    
    return (x0, y, quickest_time)


def quickest_route_fast(dep_loc, arr_loc, npoints, nbest, lat_iagos, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, method, disp, maxiter):
    """Compute the quickest route from dep_loc to arr_loc, faster but less accurate version with restricted airspace avoidance"""
    
    # Fix coordinate order - ensure consistent lon, lat format
    if len(dep_loc) == 2 and len(arr_loc) == 2:
        lon_dep, lat_dep = dep_loc
        lon_arr, lat_arr = arr_loc
    else:
        raise ValueError("dep_loc and arr_loc must be (lon, lat) tuples")
    
    # Bounds for latitude optimization (more restrictive near poles)
    lat_min = max(-89.9, min(lat_dep, lat_arr) - 20)
    lat_max = min(89.9, max(lat_dep, lat_arr) + 20)
    bnds = tuple((lat_min, lat_max) for i in range(npoints))

    # Load conflict zones for restricted airspace checking
    if not hasattr(quickest_route_fast, '_conflict_zones'):
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except (AttributeError, KeyError):
            world = gpd.read_file('ne_110m_admin_0_countries.zip')
        
        conflict_countries = ['Russia', 'Ukraine', 'Libya', 'Syria', 'Sudan', 'Yemen']
        name_col = 'NAME' if 'NAME' in world.columns else 'name'
        conflict_zones = world[world[name_col].isin(conflict_countries)]
        conflict_buffered = conflict_zones.geometry.buffer(1.0)
        quickest_route_fast._conflict_zones = conflict_buffered

    def is_route_in_restricted_airspace(lons, lats):
        """Check if route crosses restricted airspace"""
        for i in range(len(lons)):
            point = Point(lons[i], lats[i])
            for conflict_zone in quickest_route_fast._conflict_zones:
                if conflict_zone.contains(point):
                    return True
        return False

    #--List of possible solutions
    y_list=[]
    dtime_list=[]
    #
    def shortest_route(dep_loc, arr_loc, npoints):
        """Compute great circle route using proper geodesic calculations"""
        lon_dep, lat_dep = dep_loc
        lon_arr, lat_arr = arr_loc
        
        # Use geodesic for proper great circle calculation
        geod = Geodesic.WGS84
        line = geod.InverseLine(lat_dep, lon_dep, lat_arr, lon_arr)  # Note: lat, lon for geod
        
        # Calculate points along the great circle
        n_segments = npoints
        lons = np.zeros(n_segments)
        lats = np.zeros(n_segments)
        
        for i in range(n_segments):
            s = line.s13 * i / (n_segments - 1)  # Distance along the line
            pos = line.Position(s)
            lats[i] = pos['lat2']
            lons[i] = pos['lon2']
        
        return lons, lats
    #--First compute shortest route
    x0, y0 = shortest_route(dep_loc, arr_loc, npoints)
    #
    #--Length of longitude vector
    n=len(x0)
    #
    #--Test how good a first guess this is (using cost_squared which includes conflict penalty)
    dtime=cost_squared(y0[1:-1], x0[1:-1], lon_dep, lat_dep, lon_arr, lat_arr, 
                       lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed)
    y_list.append(y0) ; dtime_list.append(dtime)
    #
    #--More possible first guesses with enhanced avoidance of restricted zones
    shift_values = [-21.-18.,-12.,-6.,-3.,3.,6.,12.,21.]

    # Add additional shifts if great circle crosses restricted airspace
    if is_route_in_restricted_airspace(x0, y0):
        # Add larger shifts to avoid restricted zones
        shift_values.extend([-90., -60., -45., -30., -25., 25., 30., 45., 60., 90.])
    
    for dymax in shift_values:
      for imid in [n//2, n//3, 2*n//3]:
         dy=[dymax*float(i)/float(imid) for i in range(imid)]+[dymax*float(n-i)/float(n-imid) for i in range(imid,n)]
         y0p=y0+np.array(dy)
         
         # Ensure shifted points are within bounds
         y0p = np.clip(y0p, lat_min, lat_max)
         
         # Use cost_squared which includes conflict penalty
         dtime=cost_squared(y0p[1:-1], x0[1:-1], lon_dep, lat_dep, lon_arr, lat_arr,
                           lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed)
         y_list.append(y0p) ; dtime_list.append(dtime)
    #
    #--find the nbest first guesses
    idx=np.argpartition(dtime_list,nbest)
    y_list_to_minimize=[y_list[i] for i in idx[:nbest]]
    #
    #--initialise y to one of value (it does not matter which one)
    quickest_y=y_list[0]
    quickest_time=dtime_list[0]
    #
    #--loop on selected best first guesses
    for y in y_list_to_minimize:
       #--Minimization with y as initial conditions (cost_squared includes conflict avoidance)
       res = minimize(cost_squared,y[1:-1],args=(x0[1:-1],*dep_loc,*arr_loc, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed ),\
                      method=method,bounds=bnds[1:-1],options={'maxiter':maxiter,'disp':disp} )
       y_2 = np.append(np.insert(res['x'],0,dep_loc[1]),arr_loc[1])
       quickest_time_2=cost_squared(y_2[1:-1], x0[1:-1], lon_dep, lat_dep, lon_arr, lat_arr,
                                   lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed)
       if quickest_time_2 < quickest_time:
          quickest_time = quickest_time_2
          quickest_y = y_2   #--new best minimum
    #
    #--Solution to optimal route
    return (x0, quickest_y, quickest_time)



def cost_squared(y, x0, lon1, lat1, lon2, lat2, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, dtprint=False):
    """
    Cost function for optimization.
    Calculates the squared cost of flight time based on wind data and conflict zones.
    
    Parameters:
    - y: vector of latitudes to be optimized (excluding departure and arrival)
    - x0: vector of longitudes corresponding to y
    - lon1, lat1: departure longitude and latitude
    - lon2, lat2: arrival longitude and latitude
    - lons_wind, lats_wind: wind data coordinates
    - xr_u200_reduced, xr_v200_reduced: reduced wind data for u and v components
    - airspeed: cruise speed in m/s
    - dtprint: whether to print debug information
    
    Returns:
    - Total cost as the sum of squared flight time and conflict penalties.
    """
    #--y is the vector to be optimized (excl. departure and arrival)
    #--x0 is the coordinate (vector of longitudes corresponding to the y)
    #--lons vector including dep and arr
    lons = np.array([lon1] + list(x0) + [lon2])
    #--lats vector including dep and arr
    lats = np.array([lat1] + list(y) + [lat2])

    # Define conflict countries and create penalty zones using proper country boundaries
    conflict_countries = ['Russia', 'Ukraine', 'Libya', 'Syria', 'Sudan', 'Yemen']

    # Load world boundaries (cache this globally for performance)
    if not hasattr(cost_squared, '_conflict_buffered'):
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except (AttributeError, KeyError):
            world = gpd.read_file('ne_110m_admin_0_countries.zip')
        
        # Create conflict zone masks
        name_col = 'NAME' if 'NAME' in world.columns else 'name'
        conflict_zones = world[world[name_col].isin(conflict_countries)]
        
        # Create 1 degree buffers for conflict countries
        conflict_buffered = conflict_zones.geometry.buffer(1.0)
        
        # Cache the buffered zones for performance
        cost_squared._conflict_buffered = conflict_buffered

    # Calculate base flight time
    base_cost = cost_time(lons, lats, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, dtprint=dtprint)

    # Calculate conflict penalty by counting segments that cross conflict zones
    conflict_penalty = 0
    
    for i in range(len(lons) - 1):
        # Create points for segment endpoints
        point1 = Point(lons[i], lats[i])
        point2 = Point(lons[i+1], lats[i+1])
        
        # Check if either endpoint is in conflict zones
        segment_in_conflict = False
        for conflict_buffered in cost_squared._conflict_buffered:
            if (conflict_buffered.contains(point1) or 
                conflict_buffered.contains(point2)):
                segment_in_conflict = True
                break
        
        if segment_in_conflict:
            # Add fixed penalty per segment crossing conflict zone
            # This avoids geodesic calculations while still penalizing conflict crossings
            conflict_penalty += 100000  # Fixed penalty per segment

    #--return cost time squared plus conflict penalty
    return (base_cost**2.0) + conflict_penalty
                        
                        
                        