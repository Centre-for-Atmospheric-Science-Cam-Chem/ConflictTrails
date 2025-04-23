import pandas as pd
from kts_to_ms import kts_to_ms
from ft_to_m import ft_to_m
from ftmin_to_ms import ftmin_to_ms
from icet import icet
from ambiance import Atmosphere
def generate_flightpath(typecode,
                        gc_dist: float = 200 * 1e3, # m
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
    flight_path = dict()
    
    # Constants
    min_cruise_duration = 600 # s, 10 minutes
    passenger_load_factor = 0.835 # iata 2024 average seat load factor
    passenger_freight_factor = 0.851 # ICAO 2024 average freight load factor
    
    # Get aircraft performance data for the given typecode
    aircraft_data = performance_data[performance_data['typecode'] == typecode]
    alt_max = ft_to_m(aircraft_data['cruise_Ceiling']*1e2) # m
        
    # Takeoff phase
    m_to = aircraft_data['take-off_MTOW'] * 0.85 # FIX - make far more robust, in kg
    v_to = icet(kts_to_ms(aircraft_data['take-off_V2_IAS']), 0)[0] # m/s, TAS
    s_to = aircraft_data['take-off_Distance'] # m
    
    t_total = 0
    s_total = 0
    
    # build flight 0 to 5000 ft
    alt_start = 0
    if alt_max <= ft_to_m(5000): # if ceiling is between 0 and 5000 ft
        alt_end = alt_max # m
        # Build climb 0-ceiling feet
        if aircraft_data['initial_climb_to_5000ft_IAS'] == 'no data':
            v_climb_0_5 = 0
            w_climb_0_5 = 0
        else: 
            v_climb_0_5 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_end/2))[0] # m/s, TAS
            w_climb_0_5 = ftmin_to_ms(aircraft_data['initial_climb_to_5000ft_ROC']) # m/s
        gs_climb_0_5 = (v_climb_0_5**2 - w_climb_0_5**2) ** 0.5 # ground speed and thus distance covered
        
        t_climb_0_5 = (alt_end - alt_start) / w_climb_0_5 # s
        s_climb_0_5 = gs_climb_0_5 * t_climb_0_5 # m
        t_total += t_climb_0_5 # s
        s_total += s_climb_0_5 # m
        
        # Build Descent ceiling-0 feet
        if aircraft_data['approach_IAS'] == 'no data':
            v_descent_5_0 = 0
            w_descent_5_0 = 0
        else:
            v_descent_5_0 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_end/2))[0] # m/s, TAS
            w_descent_5_0 = ftmin_to_ms(aircraft_data['approach_ROD'])
        gs_descent_5_0 = (v_descent_5_0**2 - w_descent_5_0**2) ** 0.5
        
        t_descent_5_0 = (alt_end - alt_start) / w_descent_5_0
        s_descent_5_0 = gs_descent_5_0 * t_descent_5_0
        t_total += t_descent_5_0
        s_total += s_descent_5_0

    elif alt_max > ft_to_m(5000): # if ceiling is greater than 5000 ft
        alt_start = 0 # m
        alt_end = ft_to_m(5000) # m
        
        # Build Climb 0-5000 ft
        if aircraft_data['initial_climb_to_5000ft_IAS'] == 'no data':
            v_climb_0_5 = 0
            w_climb_0_5 = 0
        else: 
            v_climb_0_5 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_end/2))[0] # m/s, TAS
            w_climb_0_5 = ftmin_to_ms(aircraft_data['initial_climb_to_5000ft_ROC']) # m/s
        gs_climb_0_5 = (v_climb_0_5**2 - w_climb_0_5**2) ** 0.5 # ground speed and thus distance covered
        
        t_climb_0_5 = (alt_end - alt_start) / w_climb_0_5
        s_climb_0_5 = gs_climb_0_5 * t_climb_0_5
        t_total += t_climb_0_5
        s_total += s_climb_0_5
        
        
        # Build Descent 5000-0 ft ft
        if aircraft_data['approach_IAS'] == 'no data':
            v_descent_5_0 = 0
            w_descent_5_0 = 0
        else:
            v_descent_5_0 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_end/2))[0] # m/s, TAS
            w_descent_5_0 = ftmin_to_ms(aircraft_data['approach_ROD'])
        gs_descent_5_0 = (v_descent_5_0**2 - w_descent_5_0**2) ** 0.5
        
        t_descent_5_0 = (alt_end - alt_start) / w_descent_5_0
        s_descent_5_0 = gs_descent_5_0 * t_descent_5_0
        t_total += t_descent_5_0
        s_total += s_descent_5_0
        
        if alt_max <= ft_to_m(10000): # if ceiling is between 5000 and 10000 ft
            alt_start = ft_to_m(5000) # m
            alt_end= alt_max # m
            # Build Climb 5000-ceiling feet
            if aircraft_data['climb_to_fl_150_IAS'] == 'no data':
                v_climb_5_10 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start + alt_end) /2)[0]
                w_climb_5_10 = w_climb_0_5
            else:
                v_climb_5_10 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                w_climb_5_10 = ftmin_to_ms(aircraft_data['climb_to_fl_150_ROC'])
            gs_climb_5_15 = (v_climb_5_10**2 - w_climb_5_10**2) ** 0.5 # ground speed and thus distance covered
            
            t_climb_5_10 = (alt_end - alt_start) / w_climb_5_10 # s
            s_climb_5_10 = gs_climb_5_10 * t_climb_5_10
            t_total += t_climb_5_10 # s
            s_total += s_climb_5_10 # m
            
            # Build Descent ceiling-5000 ft
            if aircraft_data['approach_IAS'] == 'no data':
                v_descent_10_5 = 0
                w_descent_10_5 = 0
            else:
                v_descent_10_5 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end)/2)[0]
                w_descent_10_5 = ftmin_to_ms(aircraft_data['approach_ROD'])
            gs_descent_10_5 = (v_descent_10_5**2 - w_descent_10_5**2) ** 0.5 # ground speed and thus distance covered
            
            t_descent_10_5 = (alt_end - alt_start) / w_descent_10_5 # s
            s_descent_10_5 = gs_descent_10_5 * t_descent_10_5
            t_total += t_descent_10_5
            s_total += s_descent_10_5
        
        else: # if ceiling is greater than 10000 ft
            alt_start = ft_to_m(5000) # m
            alt_end= ft_to_m(10000) # m
            
            # Build Climb 5000-10000 ft
            if aircraft_data['climb_to_fl_150_IAS'] == 'no data':
                v_climb_5_10 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start + alt_end) /2)[0]
                w_climb_5_10 = w_climb_0_5
            else:
                v_climb_5_10 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                w_climb_5_10 = ftmin_to_ms(aircraft_data['climb_to_fl_150_ROC'])
            gs_climb_5_10 = (v_climb_5_10**2 - w_climb_5_10**2) ** 0.5 # ground speed and thus distance covered
            
            t_climb_5_10 = (alt_end - alt_start) / w_climb_5_10 # s
            s_climb_5_10 = gs_climb_5_10 * t_climb_5_10
            t_total += t_climb_5_10 # s
            s_total += s_climb_5_10 # m
            
            # Build Descent 10000-5000 ft
            if aircraft_data['approach_IAS'] == 'no data':
                v_descent_10_5 = 0
                w_descent_10_5 = 0
            else:
                v_descent_10_5 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end) /2)[0]
                w_descent_10_5 = ftmin_to_ms(aircraft_data['approach_ROD'])
            gs_descent_10_5 = (v_descent_10_5**2 - w_descent_10_5**2) ** 0.5 # ground speed and thus distance covered
            
            t_descent_10_5 = (alt_end - alt_start) / w_descent_10_5
            s_descent_10_5 = gs_descent_10_5 * t_descent_10_5
            t_total += t_descent_10_5
            s_total += s_descent_10_5
            
            if alt_max <= ft_to_m(15000): # if ceiling is between 10000 and 15000 ft
                alt_start = ft_to_m(10000) # m
                alt_end   = alt_max # m
                
                # Build Climb 10000-ceiling feet
                if aircraft_data['climb_to_fl_150_IAS'] == 'no data':
                    v_climb_10_15 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start + alt_end) /2)[0]
                    w_climb_10_15 = w_climb_5_10
                else:
                    v_climb_10_15 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                    w_climb_10_15 = ftmin_to_ms(aircraft_data['climb_to_fl_150_ROC'])
                gs_climb_10_15 = (v_climb_10_15**2 - w_climb_10_15**2) ** 0.5
                
                t_climb_10_15 = (alt_end - alt_start) / w_climb_10_15 # s
                s_climb_10_15 = gs_climb_10_15 * t_climb_10_15
                t_total += t_climb_10_15
                s_total += s_climb_10_15
                
                # Build Descent ceiling-10000 ft
                if aircraft_data['descent_to_fl_100_IAS'] == 'no data':
                    v_descent_15_10 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end)/2)[0]
                    w_descent_15_10 = w_descent_10_5
                else:
                    v_descent_15_10 = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_start + alt_end)/2)[0]
                    w_descent_15_10 = ftmin_to_ms(aircraft_data['descent_to_fl_100_ROD'])
                gs_descent_15_10 = (v_descent_15_10**2 - w_descent_15_10**2) ** 0.5
                
                t_descent_15_10 = (alt_end - alt_start) / w_descent_15_10 # s
                s_descent_15_10 = gs_descent_15_10 * t_descent_15_10
                t_total += t_descent_15_10
                s_total += s_descent_15_10
                
                
            else: # if ceiling is greater than 15000 ft
                alt_start = ft_to_m(10000) # m
                alt_end   = ft_to_m(15000) # m
                # Build Climb 10000-15000 ft
                if aircraft_data['climb_to_fl_150_IAS'] == 'no data':
                    v_climb_10_15 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start + alt_end) /2)[0]
                    w_climb_10_15 = w_climb_0_5
                else:
                    v_climb_10_15 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                    w_climb_10_15 = ftmin_to_ms(aircraft_data['climb_to_fl_150_ROC'])
                gs_climb_10_15 = (v_climb_10_15**2 - w_climb_10_15**2) ** 0.5
                
                t_climb_10_15 = (alt_end - alt_start) / w_climb_10_15 # s
                s_climb_10_15 = gs_climb_10_15 * t_climb_10_15
                t_total += t_climb_10_15
                s_total += s_climb_10_15
                
                # Build Descent 15000-10000 ft
                if aircraft_data['descent_to_fl_100_IAS'] == 'no data':
                    v_descent_15_10 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end)/2)[0]
                    w_descent_15_10 = w_descent_10_5
                else:
                    v_descent_15_10 = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_start + alt_end)/2)[0]
                    w_descent_15_10 = ftmin_to_ms(aircraft_data['descent_to_fl_100_IAS'])
                gs_descent_15_10 = (v_descent_15_10**2 - w_descent_15_10**2) ** 0.5
                
                t_descent_15_10 = (alt_end - alt_start) / w_descent_15_10 # s
                s_descent_15_10 = gs_descent_15_10 * t_descent_15_10
                t_total += t_descent_15_10
                s_total += s_descent_15_10
                
                if alt_max <= ft_to_m(24000): # if ceiling is between 15000 and 24000 ft
                    alt_start = ft_to_m(15000) # m
                    alt_end   = alt_max # m
                    # Build Climb 15000-ceiling feet
                    if aircraft_data['climb_to_fl_240_IAS'] == 'no data':
                        v_climb_15_24 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                        w_climb_15_24 = w_climb_10_15
                    else:
                        v_climb_15_24 = icet(kts_to_ms(aircraft_data['climb_to_fl_240_IAS']), (alt_start + alt_end) /2)[0]
                        w_climb_15_24 = ftmin_to_ms(aircraft_data['climb_to_fl_240_ROC'])
                    gs_climb_15_24 = (v_climb_15_24**2 - w_climb_15_24**2) ** 0.5
                    
                    t_climb_15_24 = (alt_end - alt_start) / w_climb_15_24 # s
                    s_climb_15_24 = gs_climb_15_24 * t_climb_15_24
                    t_total += t_climb_15_24
                    s_total += s_climb_15_24
                        
                    # Build Descent ceiling-15000 ft
                    if aircraft_data['descent_to_fl_100_IAS'] == 'no data':
                        v_descent_24_15 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end)/2)[0]
                        w_descent_24_15 = w_descent_15_10
                    else:
                        v_descent_24_15 = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_start + alt_end)/2)[0]
                        w_descent_24_15 = ftmin_to_ms(aircraft_data['descent_to_fl_100_ROD'])
                    gs_descent_24_15 = (v_descent_24_15**2 - w_descent_24_15**2) ** 0.5
                    
                    t_descent_24_15 = (alt_end - alt_start) / w_descent_24_15
                    s_descent_24_15 = gs_descent_24_15 * t_descent_24_15
                    t_total += t_descent_24_15
                    s_total += s_descent_24_15
                else: # if ceiling is greater than 24000 ft
                    alt_start = ft_to_m(15000) # m
                    alt_end   = ft_to_m(24000)
                    # Build Climb 15000-24000 ft
                    if aircraft_data['climb_to_fl_240_IAS'] == 'no data':
                        v_climb_15_24 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                        w_climb_15_24 = w_climb_10_15
                    else:
                        v_climb_15_24 = icet(kts_to_ms(aircraft_data['climb_to_fl_240_IAS']), (alt_start + alt_end) /2)[0]
                        w_climb_15_24 = ftmin_to_ms(aircraft_data['climb_to_fl_240_ROC'])
                    gs_climb_15_24 = (v_climb_15_24**2 - w_climb_15_24**2) ** 0.5
                    t_climb_15_24 = (alt_end - alt_start) / w_climb_15_24 # s
                    s_climb_15_24 = gs_climb_15_24 * t_climb_15_24
                    t_total += t_climb_15_24
                    s_total += s_climb_15_24
                    
                    # Build Descent 24000-15000 ft
                    if aircraft_data['descent_to_fl_100_IAS'] == 'no data':
                        v_descent_24_15 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end)/2)[0]
                        w_descent_24_15 = w_descent_15_10
                    else:
                        v_descent_24_15 = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_start + alt_end)/2)[0]
                        w_descent_24_15 = ftmin_to_ms(aircraft_data['descent_to_fl_100_ROD'])
                    gs_descent_24_15 = (v_descent_24_15**2 - w_descent_24_15**2) ** 0.5
                    t_descent_24_15 = (alt_end - alt_start) / w_descent_24_15
                    s_descent_24_15 = gs_descent_24_15 * t_descent_24_15
                    t_total += t_descent_24_15
                    s_total += s_descent_24_15
                    
                    # Build Climb 24000-ceiling feet
                    alt_start = ft_to_m(24000)
                    alt_end   = alt_max # m
                    if aircraft_data['mach_climb_MACH'] == 'no data':
                        v_climb_ceil = icet(kts_to_ms(aircraft_data['climb_to_fl_240_IAS']), (alt_start + alt_end) /2)[0]
                        w_climb_ceil = w_climb_15_24
                    else:
                        v_climb_ceil = aircraft_data['mach_climb_MACH'] * Atmosphere((alt_start + alt_end)/2).speed_of_sound # m/s, TAS
                        w_climb_ceil = ftmin_to_ms(aircraft_data['mach_climb_ROC'])
                    gs_climb_ceil = (v_climb_ceil**2 - w_climb_ceil**2) ** 0.5 # ground speed and thus distance covered
                    
                    t_climb_ceil = (alt_end - alt_start) / w_climb_ceil # s
                    s_climb_ceil = gs_climb_ceil * t_climb_ceil
                    t_total += t_climb_ceil
                    s_total += s_climb_ceil
                    
                    
                    # Build Descent ceiling-24000 ft
                    if aircraft_data['initial_descent_to_fl_240_MACH'] == 'no data':
                        v_descent_ceil = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_start + alt_end)/2)[0]
                        w_descent_ceil = w_descent_24_15
                    else:
                        v_descent_ceil = aircraft_data['initial_descent_to_fl_240_MACH'] * Atmosphere((alt_start + alt_end)/2).speed_of_sound
                        w_descent_ceil = ftmin_to_ms(aircraft_data['initial_descent_to_fl_240_ROD'])
                    gs_descent_ceil = (v_descent_ceil**2 - w_descent_ceil**2) ** 0.5
                    
                    t_descent_ceil = (alt_end - alt_start) / w_descent_ceil
                    s_descent_ceil = gs_descent_ceil * t_descent_ceil
                    t_total += t_descent_ceil
                    s_total += s_descent_ceil

    # Build cruise phase
    v_cruise = kts_to_ms(aircraft_data['cruise_TAS']) # m/s, TAS
    w_cruise = 0 # assumes no climbing 
    gs_cruise = (v_cruise**2 - w_cruise**2) ** 0.5 # ground speed and thus distance covered
    t_cruise = (gc_dist * 1e3 - s_total) / gs_cruise # s
    s_cruise = gc_dist * 1e3 - s_total # m
    


generate_flightpath('A320', performance_data=pd.read_pickle('aircraft_performance_data_table.pkl'))

# generate a flightpath that allows one to 