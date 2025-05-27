
from process_month_emissions import process_month_emissions
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
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

performance_and_emissions_model = pd.read_pickle('performance_and_emissions_model.pkl')
# Prepare arguments for each month
month_args = []

def process_month_emissions_wrapper(args):
    return process_month_emissions(*args)

for start_time_str_loop in pd.date_range(start=pd.to_datetime(start_time_str), end=pd.to_datetime(stop_time_str), freq='MS', tz='UTC'):
    stop_time_str_loop = (start_time_str_loop + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
    month_args.append((start_time_str_loop, output_dir, performance_and_emissions_model))
    process_month_emissions(
        start_time_str_loop,
        output_dir=output_dir,
        performance_and_emissions_model=performance_and_emissions_model
    )
    
#with Pool() as pool:
#   list(tqdm(pool.imap(process_month_emissions_wrapper, month_args), total=len(month_args)))