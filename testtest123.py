import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'FlightTrajectories'))
from FlightTrajectories.misc_geo import el_foeew

def test_el_foeew():
    # Test with 20°C (celsius=True)
    Tair = 20.0
    result = el_foeew(Tair, celsius=True)
    print(f"Saturation vapor pressure at {Tair}°C: {result} Pa")

if __name__ == "__main__":
    test_el_foeew()