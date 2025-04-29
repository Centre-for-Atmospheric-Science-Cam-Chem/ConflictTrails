import pdfplumber
import re
import pandas as pd
import os

def extract_engine_data(pdf_path):
    """
    Extract key engine parameters from a given PDF file.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: Dictionary with extracted information.
    """
    data = {}
    # Open the PDF file with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # Read text from all pages
        full_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

    # Use regular expressions to extract values.
    # Extract engine identification and bypass ratio.
    m = re.search(r'ENGINE IDENTIFICATION:\s*([\w-]+).*BYPASS RATIO:\s*([\d.]+)', full_text)
    if m:
        data["Engine Identification"] = m.group(1)
        data["Bypass Ratio"] = float(m.group(2))
    
    # Extract Unique ID Number and Pressure Ratio (poo)
    m = re.search(r'UNIQUE ID NUMBER:\s*(\w+).*PRESSURE RATIO\s*\(poo\):\s*([\d.]+)', full_text, re.DOTALL)
    if m:
        data["Unique ID"] = m.group(1)
        data["Pressure Ratio"] = float(m.group(2))
    
    # Extract combustor and engine type. Use a non-greedy match to stop at line breaks.
    m = re.search(r'COMBUSTOR:\s*(.+)', full_text)
    if m:
        # Stop at end of line (if multiple fields are on the same line, split by double space)
        combustor = m.group(1).strip().split("  ")[0]
        data["Combustor"] = combustor
    
    m = re.search(r'ENGINE TYPE:\s*(.+)', full_text)
    if m:
        engine_type = m.group(1).strip().split("  ")[0]
        data["Engine Type"] = engine_type

    # Optionally, extract additional data.
    # For example, suppose you want to extract the rated thrust value using a pattern.
    m = re.search(r'RATED THRUST \(Foo\) \(kN\):\s*([\d.]+)', full_text)
    if m:
        data["Rated Thrust (kN)"] = float(m.group(1))
    
    # Extract one row of measured emission data as an example: TAKE-OFF row.
    # The regex below assumes the row format is: Mode, Time, Fuel Flow, HC, CO, NOX, SMOKE, NUMBER and that values are separated by spaces.
    # You might need to adjust this to match your table's layout.
    m = re.search(r'TAKE-OFF\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', full_text)
    if m:
        data["Takeoff Fuel Flow (kg/s)"] = float(m.group(1))
        data["Takeoff HC"] = float(m.group(2))
        data["Takeoff CO"] = float(m.group(3))
        data["Takeoff NOX"] = float(m.group(4))
        # If the table includes more columns, add similar parsing lines.
    
    return data

# Folder or list of PDF file names
pdf_files = [
    "2GE048-CF6-80C2B6F, 1862M39 Combustor (22.01.2021).pdf",
    "01P06AL031-AE3007A1P, Type 3 (reduced emissions) Combustor (22.01.2021).pdf"
]

results = []
for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        extracted = extract_engine_data(pdf_file)
        extracted["File Name"] = pdf_file
        results.append(extracted)
    else:
        print(f"File not found: {pdf_file}")

# Create a pandas DataFrame with the extracted data
df = pd.DataFrame(results)
print(df)

# Optionally, save to CSV
df.to_csv("engine_performance_data.csv", index=False)
