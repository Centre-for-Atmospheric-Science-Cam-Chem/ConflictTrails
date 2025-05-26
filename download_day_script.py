from download_day import download_day
import pandas as pd

# User Inputs:
start_time_str       = '2023-12-03T00:00:00Z'
stop_time_str        = '2023-12-31T23:59:59Z'
query_limit          = int(15e4)
send_notification    = True
make_plot            = False
output_dir           = "/scratch/omg28/Data/no_track2023"

# Convert start and stop times to datetime objects
start_time_simple = pd.to_datetime(start_time_str).strftime("%Y-%m-%d")
stop_time_simple = pd.to_datetime(stop_time_str).strftime("%Y-%m-%d")

download_day(start_time_str, stop_time_str, query_limit, send_notification, make_plot, output_dir)
