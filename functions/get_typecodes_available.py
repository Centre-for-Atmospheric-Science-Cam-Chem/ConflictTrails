import requests
from lxml import html
import pandas as pd

def get_typecodes_available():
    """
    Fetches the available typecodes from the Eurocontrol Aircraft Performance database.
    Returns a DataFrame containing the typecodes.
    """
    # Load the first page in the database
    # This page contains the dropdown menu with all available models
    page = requests.get("https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=A10")
    tree = html.fromstring(page.content)

    # Get the available models in the database
    typecodes_available = tree.xpath('//select[@id="wsGroupDropDownList"]/option/@value')
    
    # Convert to a pandas dataframe
    typecodes_available = pd.DataFrame(typecodes_available, columns=["typecode"])
    
    return typecodes_available
