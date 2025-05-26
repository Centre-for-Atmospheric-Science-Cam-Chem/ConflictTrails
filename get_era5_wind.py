import os
import cdsapi

def get_era5_wind(year: int = 2023, output_dir: str = None):
    """
    Downloads ERA5 wind data for specified pressure levels and months in 2023.
    The data is downloaded in GRIB format and unarchived.
    """
    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "pressure_level": [
            "175", "200", "225",
            "250", "300"
        ],
        "year": [str(year)],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"winddb/era5_wind_{year}.nc")
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(out_path)