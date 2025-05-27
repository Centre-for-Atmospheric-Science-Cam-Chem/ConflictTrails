# Import necessary libraries
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cartopy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'FlightTrajectories'))
from FlightTrajectories.optimalrouting.zermelo_lonlat import ZermeloLonLat
from math import radians, degrees, atan2, sin, cos, sqrt, asin
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import cartopy.feature as cfeat
import cartopy.crs as ccrs

# Path to wind data directory
wind_dir = '/scratch/omg28/Data/winddb'

# Open the ERA5 wind dataset for 2023
wind_data = xr.open_dataset(f'{wind_dir}/era5_wind_2023.nc', engine='netcdf4')

# Explore the dimensions and variables of the NetCDF file
print(wind_data)

# Create a datetime object for January 1, 2023 at 00:00 UTC
jan1 = datetime.datetime(2023, 1, 1, 0, 0, 0)

# Select wind data for Jan 1, 2023 at the 300 hPa pressure level
nc_jan = wind_data.sel(valid_time=jan1, pressure_level=300)

# Extract u and v wind components, longitude, and latitude
u = nc_jan['u']
v = nc_jan['v']
lon = nc_jan['longitude']
lat = nc_jan['latitude']

# Create a figure and axis with PlateCarree projection
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeat.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True)

# Create 2D meshgrid for longitude and latitude
lon2d, lat2d = np.meshgrid(lon, lat)

# Skip every 5th arrow for clarity in the quiver plot
skip = (slice(None, None, 5), slice(None, None, 5))

# Plot wind vectors using quiver
plt.quiver(
    lon2d[skip], lat2d[skip], u.values[skip], v.values[skip],
    scale=700, width=0.0007, headwidth=3, headlength=4,
    transform=ccrs.PlateCarree()
)

# Add a title and display the plot
plt.title('Wind Vectors at 300 hPa (Jan 1, 2023, 00:00 UTC)')
plt.show(block=False)

# Define start and end coordinates (lon, lat in degrees)
start_lon, start_lat = -0.4614, 51.4775  # London Heathrow
end_lon, end_lat = -73.7789, 40.6397     # JFK

# Convert to radians for calculations
start_lat_rad = radians(start_lat)
start_lon_rad = radians(start_lon)
end_lat_rad = radians(end_lat)
end_lon_rad = radians(end_lon)

# Calculate great circle initial heading
d_lon = end_lon_rad - start_lon_rad
x = sin(d_lon) * cos(end_lat_rad)
y = cos(start_lat_rad) * sin(end_lat_rad) - sin(start_lat_rad) * cos(end_lat_rad) * cos(d_lon)
initial_gc_heading = (degrees(atan2(x, y)) + 360) % 360

# Heading sweep: 45 deg clockwise/counterclockwise from great circle
heading_range = np.arange(initial_gc_heading - 45, initial_gc_heading + 46, 1) % 360

# Aircraft parameters
cruise_speed = 240  # m/s
dt = 60  # seconds per step
# Helper: Haversine distance (meters)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2*R*asin(sqrt(a))

# Interpolators for wind
u_interp = RegularGridInterpolator((lat, lon), u.values, bounds_error=False, fill_value=np.nan)
v_interp = RegularGridInterpolator((lat, lon), v.values, bounds_error=False, fill_value=np.nan)


# Create an instance of ZermeloLonLat for position updates
zermelo = ZermeloLonLat()
zermelo.timestep = dt  # Set the timestep attribute (required by position_update_func)

# Simulate all headings with progress bar
results = []
for heading in tqdm(heading_range, desc="Simulating headings"):
    pos = np.array([start_lat, start_lon])
    total_time = 0
    path = [pos.copy()]
    while haversine(pos[1], pos[0], end_lon, end_lat) > 100000:  # 10 km threshold
        wind_u = u_interp(pos).item()
        wind_v = v_interp(pos).item()
        # Update position using ZermeloLonLat
        # position_update_func returns (new_lon, new_lat)
        new_lon, new_lat = zermelo.position_update_func(
            pos, heading, cruise_speed, wind_u, wind_v
        )
        # Store as [lat, lon] for consistency
        pos = np.array([new_lat, new_lon])
        total_time += dt
        path.append(pos.copy())
        # Prevent infinite loops
        if total_time > 12*3600:  # 12 hours
            break
    if haversine(pos[1], pos[0], end_lon, end_lat) <= 10000:
        results.append({'heading': heading, 'time': total_time, 'path': np.array(path)})

# Find optimal path
if results:
    best = min(results, key=lambda x: x['time'])
    print(f"Optimal initial heading: {best['heading']:.1f} deg, Time: {best['time']/3600:.2f} hours")
    # Plot optimal path
    ax.plot(best['path'][:,1], best['path'][:,0], color='red', linewidth=2, label='Optimal Path')
    ax.scatter([start_lon, end_lon], [start_lat, end_lat], color='black', marker='o', zorder=5)
    ax.legend()
    plt.show(block=False)
else:
    print("No valid path found within constraints.")
    # Always create a new figure and axis for plotting attempted paths
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = plt.axes(projection=ccrs.PlateCarree())
    ax2.set_global()
    ax2.coastlines()
    ax2.add_feature(cfeat.BORDERS, linestyle=':')
    ax2.gridlines(draw_labels=True)
    for res in results:
        # Debug: print min/max of attempted path lons/lats
        print(f"Attempted path: min/max lat {res['path'][:,0].min()} / {res['path'][:,0].max()}, min/max lon {res['path'][:,1].min()} / {res['path'][:,1].max()}")
        ax2.plot(res['path'][:,1], res['path'][:,0], linewidth=1, alpha=0.5)
    ax2.scatter([start_lon, end_lon], [start_lat, end_lat], color='black', marker='o', zorder=5)
    ax2.set_title('All Attempted Paths (No Valid Path Found)')
    plt.show()
