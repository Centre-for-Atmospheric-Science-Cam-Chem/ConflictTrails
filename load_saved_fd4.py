import pandas as pd
import os
from datetime import datetime


def load_saved_fd4(timestamp: str,
                   output_dir: str = "~/Data",
                   query_limit: int = 5e4):
    """
    Loads data from the saved .pkl files for analysis, optionally choosing the file with a specific query limit.
    The function will look for files in the specified directory and load the one that matches the timestamp and query limit.
    The function will also print the path of the loaded file.
    
    Parameters:
    :param timestamp (str): The date from which data is to be extracted in ISO format (YYYY-MM-DDTHH:MM:SSZ).
    :data_dir_str (str): The directory from which data is to be extracted. Default is '~/Data', recommended to use '/scratch/username/Data'
    : query_limit (str): The size of the data to be extracted. Default is 
    """
    # Convert the timestamp to a datetime object
    timestamp_dt = pd.to_datetime(timestamp)
    # Convert the timestamp to a string in the format YYYY-MM-DD
    timestamp = timestamp_dt.strftime("%Y-%m-%d")
    filepath = os.path.join(output_dir, f"result_df_{timestamp}_{int(query_limit)}.pkl")
    loaded_data = pd.read_pickle(filepath)
    # print("Loaded data from ", filepath)
    return loaded_data