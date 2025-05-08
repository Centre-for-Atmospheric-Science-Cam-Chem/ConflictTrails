import os
import pandas as pd
def get_engine_data(input_directory: str = "/scratch/omg28/Data/engine_data",
                    output_directory: str = "/scratch/omg28/Data/engine_data",
                    remove_superseded: bool = False) -> pd.DataFrame:
    """
    Loads data from the saved engine data .csv files for analysis. Saves this dataframe to a .pkl file, and also returns the loaded data as a DataFrame.

    Parameters:
    :param input_directory (str): The directory from which data is to be extracted. Default is '/scratch/username/Data'
    :param output_directory (str): The directory where the data will be saved. Default is '/scratch/username/Data'
    """
    # Load the data from the CSV file
    filepath = os.path.join(input_directory, "engine_data_icao.csv")
    loaded_data = pd.read_csv(filepath, index_col=0)
    # Save the data to a .pkl file
    output_filepath = os.path.join(output_directory, "engine_data_icao.pkl")
    
    if remove_superseded:
        # Remove superseded engines
        loaded_data = loaded_data[~loaded_data['Data Superseded'].str.contains('Yes', na=False)]
    
    loaded_data.to_pickle(output_filepath)
    # Return the loaded data
    print(f"Data loaded from {filepath} and saved to {output_filepath}")
    return loaded_data