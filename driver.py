from functions.download_day import download_day
from functions.load_saved_fd4 import load_saved_fd4

# User Inputs:
start_time_str       = '2024-02-01T00:00:00Z'
stop_time_str        = '2024-02-02T00:00:01Z'
# plane_callsign       = "EZY158T"
query_limit          = 5e4
send_notification    = True
make_plot            = False
output_dir           = "/scratch/omg28/Data/"

# Download data from OpenSky history database
download_day(start_time_str, stop_time_str, query_limit, send_notification, make_plot, output_dir)
