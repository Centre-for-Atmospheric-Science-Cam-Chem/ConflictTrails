import pandas as pd
from kts_to_ms import kts_to_ms
from ft_to_m import ft_to_m
from ftmin_to_ms import ftmin_to_ms
from icet import icet
from ambiance import Atmosphere
def generate_flightpath(typecode,
                        gc_dist: float = 200,
                        performance_data: pd.DataFrame = pd.DataFrame()):
    """
    Generates a flight path for a given aircraft type and distance.
    creates a mission profile with climb, cruise, and descent phases.
    Args:
        typecode (str): The aircraft type code.
        gc_dist (float): The great circle distance in km.
        performance_data_path (str): Path to the performance data file.
    Returns:
        dict: A dictionary containing the flight path information.
    """
    passenger_load_factor = 0.835 # iata 2024 average seat load factor
    passenger_freight_factor = 0.851 # ICAO 2024 average freight load factor
    
    total_distance = 0
    # Get aircraft performance data for the given typecode
    aircraft_data = performance_data[performance_data['typecode'] == typecode]
    alt_max = ft_to_m(aircraft_data['cruise_Ceiling']*1e2) # m
        
    # Takeoff phase
    m_to = aircraft_data['take-off_MTOW'] * 0.85 # FIX - make far more robust, in kg
    v_to = kts_to_ms(aircraft_data['take-off_V2_IAS']) # m/s
    s_to = aircraft_data['take-off_Distance'] # m
    
    # Initial Climb to 5000 ft
    alt_start = 0 # m
    if alt_max > ft_to_m(5000):
        alt_end   = ft_to_m(5000) # m
        
        v_climb_5 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start-alt_end)/2)[0] # m/s
        dh_climb_5 = ftmin_to_ms(aircraft_data['initial_climb_to_5000ft_ROC']) # m/s
        gs_climb_5 = (v_climb_5**2 - dh_climb_5**2) ** 0.5 # ground speed and thus distance covered
    else:
        alt_end = alt_max
        v_climb_5 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start-alt_end)/2)[0]
    # Climb to FL150
    if alt_max > ft_to_m(15000):
        alt_start = alt_end
        alt_end   = ft_to_m(15000) # m
        
        v_climb_15 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start - alt_end)/2)[0] # m/s
        dh_climb_15 = ftmin_to_ms(aircraft_data['climb_to_fl_150_ROC']) # m/s
        gs_climb_15 = (v_climb_15**2 - dh_climb_15**2) ** 0.5 # ground speed and thus distance covered
        
        # descent to 10000 ft
        
    
    # Climb to FL240 and beyond
    if alt_max > ft_to_m(24000):
        # Climb to FL240
        alt_start = alt_end
        alt_end   = ft_to_m(24000) # m
        
        v_climb_24 = icet(kts_to_ms(aircraft_data['climb_to_fl_240_IAS']), (alt_start - alt_end)/2)[0] # m/s
        dh_climb_24 = ftmin_to_ms(aircraft_data['climb_to_fl_240_ROC']) # m/s
        gs_climb_24 = (v_climb_24**2 - dh_climb_24**2) ** 0.5 # ground speed and thus distance covered
        
        # Mach climb to cruise alitude
        alt_start = alt_end
        alt_end   = alt_max # m
        
        v_climb_ceil = aircraft_data['mach_climb_MACH'] * Atmosphere((alt_start-alt_end)/2).speed_of_sound # m/s
        dh_climb_ceil = ftmin_to_ms(aircraft_data['mach_climb_ROC']) # m/s
        gs_climb_ceil = (v_climb_ceil**2 - dh_climb_ceil**2) ** 0.5 # ground speed and thus distance covered
        
        # Mach descent to 24000 ft
        alt_start = alt_end
        alt_end   = ft_to_m(24000)
        
        v_descent_ceil = aircraft_data['initial_descent_to_fl_240_MACH'] * Atmosphere((alt_start-alt_end)/2).speed_of_sound # m/s
        dh_descent_ceil = ftmin_to_ms(aircraft_data['initial_descent_to_fl_240_ROD']) # m/s
        gs_descent_ceil = (v_descent_ceil**2 - dh_descent_ceil**2) ** 0.5
        
    # Descent phase:
    # Descent from cruise altitude to 5000 ft
    
    
    
    
    
    
    # calculate the takeoff leg distance, time, and 

    # Create the flight path dictionary
    flight_path = {
        "typecode": typecode,
        "gc_dist": gc_dist
        "climb": climb_phase,
        "cruise": cruise_phase,
        "descent": descent_phase
    }

    return flight_path