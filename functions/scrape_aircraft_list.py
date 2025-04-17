import pandas as pd
from parse_aircraft_page import parse_aircraft_page

def scrape_aircraft_list(typecodes):
    records = []
    for typecode in typecodes:
        row = parse_aircraft_page(typecode)
        if row:
            records.append(row)
    return pd.DataFrame(records)