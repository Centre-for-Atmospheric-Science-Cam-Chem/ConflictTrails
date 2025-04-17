import pandas as pd
from parse_aircraft_page import parse_aircraft_page
from time import sleep
import os
import numpy as np

def scrape_aircraft_list(typecodes,
                         output_dir: str = "~/ConflictTrails"):
    """
    Scrapes the aircraft performance database for the given typecodes.
    Parameters:
    :param typecodes (pandas Dataframe): The typecodes to scrape.
    :param output_dir (str): The directory to save the scraped data.
    
    Output:
    :return records (pandas DataFrame): The scraped data.
    """
    records = []
    for typecode in typecodes['typecode']:
        print(f"Scraping typecode: {typecode}")
        row = parse_aircraft_page(typecode)
        if row:
            records.append(row)
        # sleep for a random time between 1 and 1.1 seconds to avoid overloading the server
        sleep(0.1 +  1 * np.random.rand())
    
    records = pd.DataFrame(records)
    # save the result to a csv file
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"aircraft_performance_table.csv")
    records.to_csv(filename, index=False)
    print("Saved result_df to", filename)
    
    return records