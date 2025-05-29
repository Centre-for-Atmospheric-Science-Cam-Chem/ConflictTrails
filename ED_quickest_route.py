import os
import sys
import numpy as np
import time
import xarray as xr
import netCDF4
from sklearn.metrics import mean_squared_error
import pandas as pd
from geographiclib.geodesic import Geodesic

proj_path1 = os.path.abspath(os.path.join("FlightTrajectories"))
proj_path2 = os.path.abspath(os.path.join("FlightTrajectories/FlightTrajectories"))
if proj_path1 not in sys.path:
    sys.path.append(proj_path1)
if proj_path2 not in sys.path:
    sys.path.append(proj_path2)
from FlightTrajectories.optimalrouting import ZermeloLonLat
from FlightTrajectories.misc_geo import nearest
from FlightTrajectories.minimization import cost_time, wind

cruise_alt_ft = 35000  # in feet
cruise_speed = 250 #m/s


file_u=pathERA5+'u.'+stryr+'.GLOBAL.nc'
file_v=pathERA5+'v.'+stryr+'.GLOBAL.nc'
xr_u=xr.open_dataset(file_u)
xr_v=xr.open_dataset(file_v)

levels_wind=list(xr_u['level'].values)

optim_level,pressure_ind_closest=nearest(levels_wind,ft_to_hPa(cruise_alt_ft))


lons_wind=xr_u['longitude'].values
lats_wind=xr_u['latitude'].values
# this is probably slower than just using the whole grid - this will result in loading in the wind field many times 
xr_u200=xr_u.sel(level=optim_level,time=times_to_extract).load()
xr_u200_reduced=xr_u200.sel(longitude=lons_z,time=times_z,latitude=lats_wind).load()

xr_v200=xr_v.sel(level=optim_level,time=times_to_extract).load()
xr_v200_reduced=xr_v200.sel(longitude=lons_z,time=times_z,latitude=lats_wind).load()

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
    rmse_shortest=mean_squared_error(lat_shortest,lat_iagos_cruising)**0.5
    rmse_quickest=mean_squared_error(lat_quickest,lat_iagos_cruising)**0.5
    lat_max_shortest=np.max(np.abs(lat_shortest-lat_iagos_cruising))
    lat_max_quickest=np.max(np.abs(lat_quickest-lat_iagos_cruising))
    print('rmse and lat max=',rmse_shortest,rmse_quickest,lat_max_shortest,lat_max_quickest)
    
    return rmse_shortest, rmse_quickest, lat_max_shortest, lat_max_quickest, lon_ed_LD, lat_ed_LD, dt_ed_LD, time_elapsed_EG, lon_ed, lat_ed, dt_ed_HD, solution


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

flights = pd.read_pickle('/scratch/omg28/Data/2023-01-01_to_2023-01-31_labeled.pkl')
row = flights.iloc[244256] # long flight from EGLL to NZCH
era5_file = f"/scratch/omg28/Data/winddb/era5_wind_{analysis_year}.nc"
ds_era5 = xr.open_dataset(era5_file)
print(ds_era5)

cruise_alt_ft = 35000  # feet example value
cruise_speed_ms = 250  # m/s, example value
#################################################################


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

npoints = int(dist // nbmeters)

# NEXT UP: VERIFY lat_quickest is being loaded in properly.


rmse_shortest, rmse_quickest, lat_max_shortest, lat_max_quickest, lon_ed_LD, lat_ed_LD, dt_ed_LD,                   \
                      time_elapsed_EG, lon_ed, lat_ed, dt_ed_HD, solution =                                                 \
                      ED_quickest_route(p1, p2, airspeed, lon_p1, lon_p2, lat_p1, lat_p2,                                   \
                      lat_shortest, lat_quickest, lat_iagos_cruising, lons_wind_reduced, lats_wind, xr_u200_reduced, xr_v200_reduced, npoints):