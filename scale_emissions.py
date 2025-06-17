import pandas as pd
import numpy as np
import xarray as xr
import netCDF4
import os

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

# Define countries whose airspace we want to exclude
conflict_countries = ['Russia', 'Ukraine', 'Libya', 'Syria', 'Sudan', 'Yemen']

## scale emissions files to be close to the right order of magnitude with respect to GAIA emissions.
# load in the netCDF file with the GAIA emissions data
gaia_file = f"/scratch/omg28/Data/GAIA/2019-01-monthly.nc"
gaia = xr.open_dataset(gaia_file).load()
print(gaia)

for start_time_str_loop in pd.date_range(start=pd.to_datetime(start_time_str), end=pd.to_datetime(stop_time_str), freq='MS', tz='UTC'):
    stop_time_str_loop = (start_time_str_loop + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
    start_time_simple_loop = pd.to_datetime(start_time_str_loop).strftime("%Y-%m-%d")
    stop_time_simple_loop = pd.to_datetime(stop_time_str_loop).strftime("%Y-%m-%d")
    month_year = pd.to_datetime(start_time_str_loop).strftime('%B %Y')

    # Load the NOx emissions grid for this month
    # nox_grid = np.load(f'/home/omg28/nethome/Data/emissions/{start_time_simple_loop}_to_{stop_time_simple_loop}_NOx_war.npy')
    nox_grid = np.load(f'/home/omg28/nethome/Data/emissions/{start_time_simple_loop}_to_{stop_time_simple_loop}_NOx_war.npy')

    # print off the maximum value in the grid
    print(f'Max NOx emissions for {month_year}: {np.max(nox_grid)} kg')
    # remove the top 5000 feet of the grid
    nox_grid = nox_grid[:, :, :-5]  # Remove the top 5000 feet (5th index onwards)
    # move all of the emissions down by 5000 feet
    nox_grid = np.roll(nox_grid, shift=-5, axis=2)  # Shift down by 5 indices (5000 feet)
    # Define new grid coordinates (midpoints)
    
    new_lat_coords = np.arange(-89.5, 90, 1.0)  # -89.5, -88.5, ..., 88.5, 89.5
    new_lon_coords = np.arange(-179.5, 180, 1.0)  # -179.5, -178.5, ..., 178.5, 179.5
    new_alt_coords = np.arange(500, 50000, 1000)  # 500, 1500, ..., 48500, 49500 (50 levels)

    # Initialize new grid with aggregated emissions
    new_nox_grid = np.zeros((len(new_lat_coords), len(new_lon_coords), len(new_alt_coords)))

    # Aggregate emissions by summing neighboring cells
    for i, lat_mid in enumerate(new_lat_coords):
        for j, lon_mid in enumerate(new_lon_coords):
            for k, alt_mid in enumerate(new_alt_coords):
                # Find indices in original grid
                # For latitude: sum cells at indices 2*i and 2*i+1
                lat_idx1, lat_idx2 = 2*i, 2*i+1
                # For longitude: sum cells at indices 2*j and 2*j+1  
                lon_idx1, lon_idx2 = 2*j, 2*j+1
                # For altitude: use same index k (already excluded top 5 levels)
                alt_idx = k
                
                # Sum the 4 neighboring cells (2x2 in lat-lon, same altitude)
                if lat_idx2 < nox_grid.shape[0] and lon_idx2 < nox_grid.shape[1]:
                    new_nox_grid[i, j, k] = (nox_grid[lat_idx1, lon_idx1, alt_idx] + 
                                        nox_grid[lat_idx1, lon_idx2, alt_idx] + 
                                        nox_grid[lat_idx2, lon_idx1, alt_idx] + 
                                        nox_grid[lat_idx2, lon_idx2, alt_idx])

    # scale the new emissions grid to match the GAIA emissions grid
    # Calculate scaling factor based on total emissions
    '''
    total_nox_emissions = np.sum(new_nox_grid)
    total_gaia_emissions = np.sum(gaia['nox'].values)
    scaling_factor = total_gaia_emissions / total_nox_emissions if total_nox_emissions > 0 else 1.0
    print(f'Scaling factor for {month_year}: {scaling_factor}')
    '''
    # Scale the new emissions grid by 30 - a bit of an underestimate, but should be close enough
    # new_nox_grid *= 30 * 0.7 used in nowar case to adjust down to make emissions in war and nowar cases more similar
    
    new_nox_grid *= 30 # used in war case to match GAIA emissions more closely.
    
    # Create netCDF file
    output_filename = f'/scratch/omg28/Data/emissions/{start_time_simple_loop}_to_{stop_time_simple_loop}_NOx_war_1deg.nc'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Create time coordinate for this month
    month_time = pd.to_datetime(stop_time_str_loop).replace(day=pd.to_datetime(stop_time_str_loop).days_in_month, hour=0, minute=0, second=0)
    
    with netCDF4.Dataset(output_filename, 'w', format='NETCDF4') as nc:
        # Create dimensions to match GAIA format
        nc.createDimension('longitude', len(new_lon_coords))
        nc.createDimension('latitude', len(new_lat_coords)) 
        nc.createDimension('altitude_ft', len(new_alt_coords))
        nc.createDimension('time', 1)
        
        # Create coordinate variables
        lon_var = nc.createVariable('longitude', 'f8', ('longitude',))
        lat_var = nc.createVariable('latitude', 'f8', ('latitude',))
        alt_var = nc.createVariable('altitude_ft', 'f8', ('altitude_ft',))
        time_var = nc.createVariable('time', 'f8', ('time',))
        
        # Create data variables with same dimensions as GAIA
        seg_length_var = nc.createVariable('seg_length_km', 'f8', ('longitude', 'latitude', 'altitude_ft', 'time'))
        fuel_burn_var = nc.createVariable('fuel_burn', 'f8', ('longitude', 'latitude', 'altitude_ft', 'time'))
        nox_var = nc.createVariable('nox', 'f8', ('longitude', 'latitude', 'altitude_ft', 'time'))
        co_var = nc.createVariable('co', 'f8', ('longitude', 'latitude', 'altitude_ft', 'time'))
        hc_var = nc.createVariable('hc', 'f8', ('longitude', 'latitude', 'altitude_ft', 'time'))
        nvpm_mass_var = nc.createVariable('nvpm_mass', 'f8', ('longitude', 'latitude', 'altitude_ft', 'time'))
        nvpm_number_var = nc.createVariable('nvpm_number', 'f8', ('longitude', 'latitude', 'altitude_ft', 'time'))
        
        # Set coordinate values
        lon_var[:] = new_lon_coords
        lat_var[:] = new_lat_coords
        alt_var[:] = new_alt_coords
        time_var[:] = netCDF4.date2num(month_time, units='days since 1900-01-01', calendar='gregorian')
        
        # Calculate emissions data from NOx grid
        nox_grid_transposed = new_nox_grid.transpose(1, 0, 2)  # lon, lat, alt
        
        # Calculate fuel burn from NOx (NOx EI = 15.14 g/kg fuel)
        fuel_burn_grid = nox_grid_transposed * 1000 / 15.14  # Convert kg NOx to kg fuel
        
        # Calculate other emissions using emission indices
        co_grid = fuel_burn_grid * 3.61 / 1000  # kg CO
        hc_grid = fuel_burn_grid * 0.520 / 1000  # kg HC
        nvpm_mass_grid = fuel_burn_grid * 0.088 / 1000  # kg nvPM mass
        nvpm_number_grid = fuel_burn_grid * 1e15  # number of nvPM particles
        
        # Set data for all variables
        seg_length_var[:, :, :, 0] = np.zeros_like(nox_grid_transposed)
        fuel_burn_var[:, :, :, 0] = fuel_burn_grid
        nox_var[:, :, :, 0] = nox_grid_transposed
        co_var[:, :, :, 0] = co_grid
        hc_var[:, :, :, 0] = hc_grid
        nvpm_mass_var[:, :, :, 0] = nvpm_mass_grid
        nvpm_number_var[:, :, :, 0] = nvpm_number_grid
        
        # Add coordinate attributes to match GAIA
        lon_var.units = 'degrees_east'
        lat_var.units = 'degrees_north' 
        alt_var.units = 'feet'
        time_var.units = 'days since 1900-01-01'
        time_var.calendar = 'gregorian'
        
        # Add variable attributes
        seg_length_var.units = 'km'
        seg_length_var.long_name = 'Segment length'
        fuel_burn_var.units = 'kg'
        fuel_burn_var.long_name = 'Fuel burn'
        nox_var.units = 'kg'
        nox_var.long_name = 'NOx emissions'
        co_var.units = 'kg'
        co_var.long_name = 'CO emissions'
        hc_var.units = 'kg'
        hc_var.long_name = 'HC emissions'
        nvpm_mass_var.units = 'kg'
        nvpm_mass_var.long_name = 'nvPM mass emissions'
        nvpm_number_var.units = 'number'
        nvpm_number_var.long_name = 'nvPM number emissions'
        
        # Global attributes
        nc.title = f'Aircraft emissions for {month_year}'
        nc.description = 'Aggregated aircraft emissions on 1-degree grid'
        
        print(f"Created {output_filename}")
