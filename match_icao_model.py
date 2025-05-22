import pandas as pd
import os
from datetime import datetime
from load_saved_fd4 import load_saved_fd4

def match_icao_model(start_time_str: str,
                    stop_time_str: str,
                    query_limit: int = 30e4,
                    aircraft_db_path: str = "/scratch/omg28/Data/aircraftdb/aircraft-database-complete-2025-02.csv",
                    flight_db_path: str = "/scratch/omg28/Data/no_track2024/",
                    output_dir: str = "/scratch/omg28/Data/aircraftdb/"):
    """
    Loads data from the saved .pkl files to extract icao24 bit transponder codes and matches them with plane ICAO 4 digit model number.
    The function will join the two tables on the icao24 column and save the result to a new .pkl file, creating a mapping between the icao24 and the model number.
    
    Parameters:
    :param start_time_str (str): The start time in ISO format (YYYY-MM-DDTHH:MM:SSZ).
    :param stop_time_str (str): The end time in ISO format (YYYY-MM-DDTHH:MM:SSZ).
    :param query_limit (int): The number of queries to be made in the opensky database for the information; defaults to 30e4.
    :param aircraft_db_path (str): The path to the aircraft database CSV file.
    :param flight_db_path (str): The path to the flight database directory.
    :param output_dir (str): The directory where the output file will be saved.
    """
    # load the aircraft database, removing all rows with more than 32 columns, keeping only the typecode and icao24 columns:
    aircraft_df = pd.read_csv(aircraft_db_path, low_memory=False, on_bad_lines='skip', usecols= ["'icao24'", "'typecode'"])
    # remove the single quotes from the column names
    aircraft_df.columns = aircraft_df.columns.str.replace("'", "")
    # remove the single quotes from the icao24 column
    aircraft_df['icao24'] = aircraft_df['icao24'].str.replace("'", "")
    
    # Convert the icao24 column to a string
    aircraft_df['icao24'] = aircraft_df['icao24'].astype(str)
    # drop all rows where typecode is "
    aircraft_df = aircraft_df[aircraft_df['typecode'] != "''"]
    
    
    print("There are " + str(len(aircraft_df['typecode'].unique())) + " unique typecodes in the aircraft database.")
    
    # load the flight database
    flight_df = pd.DataFrame()
    
    # Loop through the dates between start and end date
    time_range = pd.date_range(start_time_str, stop_time_str, freq='D')
    for current_day in time_range:
        # Convert the start and stop time to POSIX time
        start_time_posix    = int(current_day.timestamp())
        end_time_posix      = int(datetime.fromisoformat(stop_time_str).timestamp())
        
        flight_df = pd.concat([flight_df, load_saved_fd4(current_day, flight_db_path, query_limit)], axis=0, ignore_index=True)
    
    # remove all rows where estdepartureairport or estarrivalairport are <NA>
    flight_df = flight_df[flight_df['estdepartureairport'].notna() & flight_df['estarrivalairport'].notna()]

    # inner join the two dataframes on the icao24 column. this eliminates all rows where typecode cannot be matched to an icao24 code.
    flight_df = pd.merge(flight_df, aircraft_df, on='icao24', how='inner')

    # Determine the month for each flight and save by month
    if 'timestamp' in flight_df.columns:
        flight_df['month'] = pd.to_datetime(flight_df['timestamp']).dt.strftime('%m')
        flight_df['year'] = pd.to_datetime(flight_df['timestamp']).dt.strftime('%Y')
    elif 'day' in flight_df.columns:
        flight_df['month'] = pd.to_datetime(flight_df['day']).dt.strftime('%m')
        flight_df['year'] = pd.to_datetime(flight_df['day']).dt.strftime('%Y')
    else:
        flight_df['month'] = pd.NA
        flight_df['year'] = pd.NA

    # Save each month's flights to a separate folder
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    grouped = flight_df.groupby(['year', 'month'])
    saved_files = []
    for (year, month), group in grouped:
        if pd.isna(year) or pd.isna(month):
            continue
        month_folder = os.path.join(output_dir, f"{year}_{month}")
        os.makedirs(month_folder, exist_ok=True)
        start_time_dt = pd.to_datetime(start_time_str)
        stop_time_dt  = pd.to_datetime(stop_time_str)
        start_time_ts = start_time_dt.strftime("%Y-%m-%d")
        stop_time_ts  = stop_time_dt.strftime("%Y-%m-%d")
        filename = os.path.join(month_folder, f"{start_time_ts}_to_{stop_time_ts}_{int(query_limit)}_typecodes_added.pkl")
        group = group.drop(columns=['month', 'year'])
        group.to_pickle(filename)
        print(f"Saved {len(group)} flights to", filename)
        saved_files.append(filename)

    # Optionally return the list of saved files
    return saved_files