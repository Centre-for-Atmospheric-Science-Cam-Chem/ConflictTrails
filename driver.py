from download_day import download_day
from load_saved_fd4 import load_saved_fd4
from scrape_aircraft_list import scrape_aircraft_list
from get_perf_model_typecodes import get_perf_model_typecodes 
from match_icao_model import match_icao_model
import pandas as pd
from time import sleep

# User Inputs:
start_time_str       = '2024-01-01T00:00:00Z'
stop_time_str        = '2024-01-10T00:00:01Z'
query_limit          = 30e4
send_notification    = True
make_plot            = False
output_dir           = "/scratch/omg28/Data/no_track2024"
'''
# Download data from OpenSky history database
download_day(start_time_str, stop_time_str, query_limit, send_notification, make_plot, output_dir)
loaded_day = load_saved_fd4(stop_time_str, output_dir, query_limit)
'''

available_codes = get_perf_model_typecodes()
aircraft_performance_data_table = scrape_aircraft_list(available_codes)
aircraft_performance_data_table.to_pickle("aircraft_performance_data_table.pkl")
