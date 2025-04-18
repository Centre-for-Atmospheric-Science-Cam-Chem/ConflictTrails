import pandas as pd
import os

def process_airport_list(input_csv: str = "/scratch/omg28/Data/airportdb/airports.csv", output_dir: str = "/scratch/omg28/Data/airportdb/"):
    """
    Processes the airport CSV file to extract relevant columns and save it as a new CSV and pickle file.
    
    Parameters:
    :param input_csv (str): The path to the input CSV file.
    :param output_dir (str): The directory to save the processed CSV file.
    
    Output:
    :return: airports_df (pandas DataFrame): The processed DataFrame containing relevant airport data.
    """
    # Load the CSV file
    airports_df = pd.read_csv(input_csv)
    
    # Select relevant columns
    airports_df = airports_df[['id', 'ident', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'gps_code']]
    
    # Save the result to a new CSV file
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "processed_airports.csv")
    airports_df.to_csv(filename, index=False)
    airports_df.to_pickle(filename.replace('.csv', '.pkl'))
    print("Saved processed airports to", filename)
    
    return airports_df