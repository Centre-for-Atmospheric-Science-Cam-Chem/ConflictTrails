from ambiance import Atmosphere
from ft_to_m import ft_to_m
import numpy as np
# coverts from altitude above mean sea level (AMSL) in feet to hectopascals (hPa)
def ft_to_hPa_bin(alt_ft: float, hPa_choices: list) -> float:
    altitude = Atmosphere(ft_to_m(alt_ft))
    hpa = altitude.pressure / 100
    binned_hpa = np.floor(hpa / 5) * 5
    return closest(hPa_choices, binned_hpa)

def closest(lst, K):
    """
    Find the closest value to K in a list.
    """
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]