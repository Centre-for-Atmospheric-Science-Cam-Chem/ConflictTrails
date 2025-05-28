from process_month_emissions import process_month_emissions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram_notifier import send_telegram_notification

# User Inputs:
start_time_str       = '2023-01-01T00:00:00Z'
stop_time_str        = '2023-10-31T23:59:59Z'
query_limit          = 15e4
send_notification    = True
make_plot            = True
output_dir           = "/scratch/omg28/Data/"

# Convert start and stop times to datetime objects
start_time_simple = pd.to_datetime(start_time_str).strftime("%Y-%m-%d")
stop_time_simple = pd.to_datetime(stop_time_str).strftime("%Y-%m-%d")

performance_and_emissions_model = pd.read_pickle('performance_and_emissions_model.pkl')

for start_time_str_loop in pd.date_range(start=pd.to_datetime(start_time_str), end=pd.to_datetime(stop_time_str), freq='MS', tz='UTC'):
    stop_time_str_loop = (start_time_str_loop + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
    process_month_emissions(
        start_time_str_loop,
        output_dir=output_dir,
        performance_and_emissions_model=performance_and_emissions_model
    )
    print(f"Generated emissions file for month: {start_time_str_loop.strftime('%Y-%m')}")
    telegram_message = f"Emissions file for {start_time_str_loop.strftime('%Y-%m')} generated successfully."
    send_telegram_notification(telegram_message)