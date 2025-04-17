import pandas as pd
from parse_aircraft_page import parse_aircraft_page
from time import sleep
import os

def scrape_aircraft_list(typecodes,
                         output_dir: str = "~/nethome"):
    records = []
    for typecode in typecodes:
        print(f"Scraping typecode: {typecode}")
        row = parse_aircraft_page(typecode)
        if row:
            records.append(row)
    
    records = pd.DataFrame(records)
    # save the result to a csv file
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"aircraft_performance_table.csv")
    records.to_csv(filename, index=False)
    print("Saved result_df to", filename)
    
    return records