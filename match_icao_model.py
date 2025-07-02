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
    #########
    num_acft_in_metadata = len(aircraft_df)
    #######
    aircraft_df = aircraft_df[aircraft_df['typecode'] != "''"]
    
    #########
    num_acftmetadata_entries_after_filtering  = len(aircraft_df)
    ###########

    # load the flight database
    flight_df = pd.DataFrame()
    
    # Loop through the dates between start and end date
    time_range = pd.date_range(start_time_str, stop_time_str, freq='D')
    for current_day in time_range:
        # Convert the start and stop time to POSIX time
        start_time_posix    = int(current_day.timestamp())
        end_time_posix      = int(datetime.fromisoformat(stop_time_str).timestamp())
        
        flight_df = pd.concat([flight_df, load_saved_fd4(current_day, flight_db_path, query_limit)], axis=0, ignore_index=True)
    num_all_seen_flights = len(flight_df)
    #########
    num_no_arrival_airport = flight_df[flight_df['estarrivalairport'].isna() & flight_df['estdepartureairport'].notna()].shape[0]
    num_no_departure_airport = flight_df[flight_df['estdepartureairport'].isna() & flight_df['estarrivalairport'].notna()].shape[0]
    num_no_airport = flight_df[flight_df['estarrivalairport'].isna() & flight_df['estdepartureairport'].isna()].shape[0]
    ############
    num_removed_airports = num_no_arrival_airport + num_no_departure_airport + num_no_airport
    # remove all rows where estdepartureairport or estarrivalairport are <NA>
    flight_df = flight_df[flight_df['estdepartureairport'].notna() & flight_df['estarrivalairport'].notna()]

    num_flights_before_icao24_filtering = len(flight_df)
    num_acft_entries_no_matched_flight = aircraft_df[~aircraft_df['icao24'].isin(flight_df['icao24'])].shape[0]
    # count entries in the flight_df dataframe that will be removed due to missing ICAO24 code match with
    # inner join the two dataframes on the icao24 column. this eliminates all rows where typecode cannot be matched to an icao24 code.
    flight_df = pd.merge(flight_df, aircraft_df, on='icao24', how='inner')
    num_flights_after_icao24_filtering = len(flight_df)
    num_flights_lost_icao_filtering = num_flights_before_icao24_filtering - num_flights_after_icao24_filtering


    # Convert the timestamp to a datetime object
    start_time_dt = pd.to_datetime(start_time_str)
    stop_time_dt  = pd.to_datetime(stop_time_str)   
    # Convert the timestamp to a string in the format YYYY-MM-DD
    start_time_ts = start_time_dt.strftime("%Y-%m-%d")
    stop_time_ts  = stop_time_dt.strftime("%Y-%m-%d")
    
    # save the result to a pickle file
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{start_time_ts}_to_{stop_time_ts}_{int(query_limit)}_typecodes_added.pkl")
    flight_df.to_pickle(filename)
    print("Saved result_df to", filename)
    
    #return the result dataframe
    return flight_df, num_acft_in_metadata, num_acftmetadata_entries_after_filtering, num_all_seen_flights, num_no_arrival_airport, num_no_departure_airport, num_no_airport, num_removed_airports, num_flights_before_icao24_filtering, num_flights_after_icao24_filtering, num_flights_lost_icao_filtering, num_acft_entries_no_matched_flight