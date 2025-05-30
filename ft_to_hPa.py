from ambiance import Atmosphere
from ft_to_m import ft_to_m
import numpy as np

# coverts from altitude above mean sea level (AMSL) in feet to hectopascals (hPa)
def ft_to_hPa(alt_ft: float) -> float:
    altitude = Atmosphere(ft_to_m(alt_ft))
    hPa = altitude.pressure / 100
    return hPa