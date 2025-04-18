import requests
from lxml import html
import pandas as pd
import os

def get_perf_model_typecodes(start_typecode: str = "A10",
                             output_dir: str = "/scratch/omg28/Data/aircraftdb/"):
    """
    Fetches the available typecodes from the Eurocontrol Aircraft Performance database.
    Returns a DataFrame containing the typecodes.
    """
    # Load the first page in the database
    # This page contains the dropdown menu with all available models
    url = f"https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO={start_typecode}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to load {url}")
        return None
    tree = html.fromstring(response.content)

    # Get the available models in the database
    typecodes_available = tree.xpath('//select[@id="wsGroupDropDownList"]/option/@value')
    
    # Convert to a pandas dataframe
    typecodes_available = pd.DataFrame(typecodes_available, columns=["typecode"])
    
    # save the result to a pickle file
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename_csv = os.path.join(output_dir, f"performance_models_typecodes.csv")
    filename_pkl = os.path.join(output_dir, f"performance_models_typecodes.pkl")
    typecodes_available.to_csv(filename_csv, index=False)
    typecodes_available.to_pickle(filename_pkl)
    print("Saved typecodes_available to", filename_csv)
    return typecodes_available