import pandas as pd
from kts_to_ms import kts_to_ms
from ft_to_m import ft_to_m
from ftmin_to_ms import ftmin_to_ms
from icet import icet
from ambiance import Atmosphere
import re
from bffm2 import bffm2
def generate_flightpath(typecode,
                        gc_dist: float = 200, # km: converted to km in function
                        performance_and_emissions_model: pd.DataFrame = pd.DataFrame(),
                        cruise_altitude_ft: float = None):
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
    # Create flight path dictionary
    flight_path = dict()
    
    # initialize to zero values:
    
    flight_path['climb'] = {
        # path 
        's_climb_0_5': 0,
        't_climb_0_5': 0,
        'M_climb_0_5': 0,
        'h_climb_0_5_start': 0,
        'h_climb_0_5_end': 0,
        's_climb_5_10': 0,
        't_climb_5_10': 0,
        'M_climb_5_10': 0,
        'h_climb_5_10_start': 0,
        'h_climb_5_10_end': 0,
        's_climb_10_15': 0,
        't_climb_10_15': 0,
        'M_climb_10_15': 0,
        'h_climb_10_15_start': 0,
        'h_climb_10_15_end': 0,
        's_climb_15_24': 0,
        't_climb_15_24': 0,
        'M_climb_15_24': 0,
        'h_climb_15_24_start': 0,
        'h_climb_15_24_end': 0,
        's_climb_ceil': 0,
        't_climb_ceil': 0,
        'M_climb_ceil': 0,
        'h_climb_ceil_start': 0,
        'h_climb_ceil_end': 0,
        # emissions 
        'HC_climb_0_5': 0,
        'CO_climb_0_5': 0,
        'NOx_climb_0_5': 0,
        'SN_climb_0_5': 0,
        'HC_climb_5_10': 0,
        'CO_climb_5_10': 0,
        'NOx_climb_5_10': 0,
        'SN_climb_5_10': 0,
        'HC_climb_10_15': 0,
        'CO_climb_10_15': 0,
        'NOx_climb_10_15': 0,
        'SN_climb_10_15': 0,
        'HC_climb_15_24': 0,
        'CO_climb_15_24': 0,
        'NOx_climb_15_24': 0,
        'SN_climb_15_24': 0,
        'HC_climb_ceil': 0,
        'CO_climb_ceil': 0,
        'NOx_climb_ceil': 0,
        'SN_climb_ceil': 0
    }
    
    
    flight_path['cruise'] = {
        # path
        's_cruise': 0,
        't_cruise': 0,
        'M_cruise': 0,
        'h_cruise_start': 0,
        'h_cruise_end': 0,
        # emissions
        'HC_cruise': 0,
        'CO_cruise': 0,
        'NOx_cruise': 0,
        'SN_Cruise': 0
    }

    flight_path['descent'] = {
        # path
        's_descent_ceil': 0,
        't_descent_ceil': 0,
        'M_descent_ceil': 0,
        'h_descent_ceil_start': 0,
        'h_descent_ceil_end': 0,
        's_descent_24_15': 0,
        't_descent_24_15': 0,
        'M_descent_24_15': 0,
        'h_descent_24_15_start': 0,
        'h_descent_24_15_end': 0,
        's_descent_15_10': 0,
        't_descent_15_10': 0,
        'M_descent_15_10': 0,
        'h_descent_15_10_start': 0,
        'h_descent_15_10_end': 0,
        's_descent_10_5': 0,
        't_descent_10_5': 0,
        'M_descent_10_5': 0,
        'h_descent_10_5_start': 0,
        'h_descent_10_5_end': 0,
        's_descent_5_0': 0,
        't_descent_5_0': 0,
        'M_descent_5_0': 0,
        'h_descent_5_0_start': 0,
        'h_descent_5_0_end': 0,
        # emissions
        'HC_descent_ceil': 0,
        'CO_descent_ceil': 0,
        'NOx_descent_ceil': 0,
        'SN_descent_ceil': 0,
        'HC_descent_24_15': 0,
        'CO_descent_24_15': 0,
        'NOx_descent_24_15': 0,
        'SN_descent_24_15': 0,
        'HC_descent_15_10': 0,
        'CO_descent_15_10': 0,
        'NOx_descent_15_10': 0,
        'SN_descent_15_10': 0,
        'HC_descent_10_5': 0,
        'CO_descent_10_5': 0,
        'NOx_descent_10_5': 0,
        'SN_descent_10_5': 0,
        'HC_descent_5_0': 0,
        'CO_descent_5_0': 0,
        'NOx_descent_5_0': 0,
        'SN_descent_5_0': 0  
    }   
    
    # convert gc_dist to m
    gc_dist = gc_dist * 1e3 # km to m
        
    # Constants
    min_cruise_duration = 600 # s, 10 minutes
    s_tolerance = 1000 # m, tolerance for distance when creating flight paths
    alt_decrement = 1000 # ft, since FLs are in kft. try 0.95 of original cruising altitude for cruise altitude for geometric progression
    min_cruise_altitude = 0 # m, minimum cruise altitude
    passenger_load_factor = 0.835 # iata 2024 average seat load factor
    passenger_freight_factor = 0.851 # ICAO 2024 average freight load factor
    
    # Get aircraft performance data for the given typecode
    aircraft_data_df = performance_and_emissions_model[performance_and_emissions_model['typecode'] == typecode]
    
    # use regex to convert all string numbers to floats
    pattern = re.compile(r'^-?\d+(\.\d+)?$')
    def to_number(val):
        if isinstance(val, str) and pattern.fullmatch(val):
            return float(val) if '.' in val else int(val)
        return val
    aircraft_data = aircraft_data_df.map(to_number).squeeze()

    # start at the cruise ceiling
    alt_max = ft_to_m(aircraft_data['cruise_Ceiling']*1e2) # m
    
    # Takeoff phase
    m_to = aircraft_data['take-off_MTOW'] * 0.85 # FIX - make far more robust, in kg
        
    # Generate the default 10 minute cruise time flight profile at the maximum cruise altitude:
    if cruise_altitude_ft is not None:
        alt_cruise = ft_to_m(cruise_altitude_ft) # m, cruise altitude
    else:
        alt_cruise = alt_max * 0.95
    t_cruise = min_cruise_duration # s, cruise time
    
    # Build climb and descent phases 
    if alt_cruise <= ft_to_m(5000): # if cruise altitude is between 0 and 5000 ft
        alt_start = ft_to_m(0) # m
        alt_end = alt_cruise # m
        # Build climb 0-cruise feet
        if aircraft_data['initial_climb_to_5000ft_IAS'] == 'no data':
            v_climb_0_5 = 0
            w_climb_0_5 = 0
        else: 
            v_climb_0_5 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_end/2))[0] # m/s, TAS
            w_climb_0_5 = ftmin_to_ms(aircraft_data['initial_climb_to_5000ft_ROC']) # m/s
        gs_climb_0_5 = (v_climb_0_5**2 - w_climb_0_5**2) ** 0.5 # ground speed and thus distance covered
        
        t_climb_0_5 = (alt_end - alt_start) / w_climb_0_5 # s
        s_climb_0_5 = gs_climb_0_5 * t_climb_0_5
        M_climb_0_5 = v_climb_0_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
        flight_path['climb']['M_climb_0_5'] = M_climb_0_5 # m/s
        flight_path['climb']['t_climb_0_5'] = t_climb_0_5 # s
        flight_path['climb']['s_climb_0_5'] = s_climb_0_5 # m
        flight_path['climb']['h_climb_0_5_start'] = alt_start # m
        flight_path['climb']['h_climb_0_5_end'] = alt_end # m

        
        # Build Descent cruise-0 feet
        if aircraft_data['approach_IAS'] == 'no data':
            v_descent_5_0 = 0
            w_descent_5_0 = 0
        else:
            v_descent_5_0 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_end/2))[0] # m/s, TAS
            w_descent_5_0 = ftmin_to_ms(aircraft_data['approach_ROD'])
        gs_descent_5_0 = (v_descent_5_0**2 - w_descent_5_0**2) ** 0.5
        
        t_descent_5_0 = (alt_end - alt_start) / w_descent_5_0
        s_descent_5_0 = gs_descent_5_0 * t_descent_5_0
        M_descent_5_0 = v_descent_5_0 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
        flight_path['descent']['M_descent_5_0'] = M_descent_5_0 # m/s
        flight_path['descent']['t_descent_5_0'] = t_descent_5_0 # s
        flight_path['descent']['s_descent_5_0'] = s_descent_5_0 # m
        flight_path['descent']['h_descent_5_0_start'] = alt_end # m
        flight_path['descent']['h_descent_5_0_end'] = alt_start # m

        
        # assign remaining climb and descent altitudes to the cruise altitude:
        flight_path['climb']['h_climb_5_10_start'] = alt_end # m
        flight_path['climb']['h_climb_5_10_end'] = alt_end # m
        flight_path['climb']['h_climb_10_15_start'] = alt_end # m
        flight_path['climb']['h_climb_10_15_end'] = alt_end # m.
        flight_path['climb']['h_climb_15_24_start'] = alt_end # m
        flight_path['climb']['h_climb_15_24_end'] = alt_end # m
        flight_path['climb']['h_climb_ceil_start'] = alt_end # m
        flight_path['climb']['h_climb_ceil_end'] = alt_end # m
        flight_path['descent']['h_descent_ceil_start'] = alt_end # m
        flight_path['descent']['h_descent_ceil_end'] = alt_end # m
        flight_path['descent']['h_descent_24_15_start'] = alt_end # m
        flight_path['descent']['h_descent_24_15_end'] = alt_end
        flight_path['descent']['h_descent_15_10_start'] = alt_end # m
        flight_path['descent']['h_descent_15_10_end'] = alt_end # m
        flight_path['descent']['h_descent_10_5_start'] = alt_end # m
        flight_path['descent']['h_descent_10_5_end'] = alt_end # m


    elif alt_cruise > ft_to_m(5000): # if cruise altitude is greater than 5000 ft
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
        M_climb_0_5 = v_climb_0_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
        flight_path['climb']['M_climb_0_5'] = M_climb_0_5 # m/s
        flight_path['climb']['t_climb_0_5'] = t_climb_0_5 # s
        flight_path['climb']['s_climb_0_5'] = s_climb_0_5 # m
        flight_path['climb']['h_climb_0_5_start'] = alt_start # m
        flight_path['climb']['h_climb_0_5_end'] = alt_end # m

                
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
        M_descent_5_0 = v_descent_5_0 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
        flight_path['descent']['M_descent_5_0'] = M_descent_5_0 # m/s
        flight_path['descent']['t_descent_5_0'] = t_descent_5_0 # s
        flight_path['descent']['s_descent_5_0'] = s_descent_5_0 # m
        flight_path['descent']['h_descent_5_0_start'] = alt_end # m
        flight_path['descent']['h_descent_5_0_end'] = alt_start

        
        if alt_cruise <= ft_to_m(10000): # if cruise altitude is between 5000 and 10000 ft
            alt_start = ft_to_m(5000) # m
            alt_end= alt_cruise # m
            # Build Climb 5000-ceiling feet
            if aircraft_data['climb_to_fl_150_IAS'] == 'no data':
                v_climb_5_10 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start + alt_end) /2)[0]
                w_climb_5_10 = w_climb_0_5
            else:
                v_climb_5_10 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                w_climb_5_10 = ftmin_to_ms(aircraft_data['climb_to_fl_150_ROC'])
            gs_climb_5_10 = (v_climb_5_10**2 - w_climb_5_10**2) ** 0.5 # ground speed and thus distance covered
            
            t_climb_5_10 = (alt_end - alt_start) / w_climb_5_10 # s
            s_climb_5_10 = gs_climb_5_10 * t_climb_5_10
            M_climb_5_10 = v_climb_5_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
            flight_path['climb']['M_climb_5_10'] = M_climb_5_10 # m/s
            flight_path['climb']['t_climb_5_10'] = t_climb_5_10 # s
            flight_path['climb']['s_climb_5_10'] = s_climb_5_10 # m
            flight_path['climb']['h_climb_5_10_start'] = alt_start # m
            flight_path['climb']['h_climb_5_10_end'] = alt_end # m

            
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
            M_descent_10_5 = v_descent_10_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
            flight_path['descent']['M_descent_10_5'] = M_descent_10_5 # m/s
            flight_path['descent']['t_descent_10_5'] = t_descent_10_5 # s
            flight_path['descent']['s_descent_10_5'] = s_descent_10_5 # m
            flight_path['descent']['h_descent_10_5_start'] = alt_end # m
            flight_path['descent']['h_descent_10_5_end'] = alt_start # m

        # assign remaining climb and descent altitudes to the cruise altitude:
            flight_path['climb']['h_climb_10_15_start'] = alt_end # m
            flight_path['climb']['h_climb_10_15_end'] = alt_end # m.
            flight_path['climb']['h_climb_15_24_start'] = alt_end # m
            flight_path['climb']['h_climb_15_24_end'] = alt_end # m
            flight_path['climb']['h_climb_ceil_start'] = alt_end # m
            flight_path['climb']['h_climb_ceil_end'] = alt_end # m
            flight_path['descent']['h_descent_ceil_start'] = alt_end # m
            flight_path['descent']['h_descent_ceil_end'] = alt_end # m
            flight_path['descent']['h_descent_24_15_start'] = alt_end # m
            flight_path['descent']['h_descent_24_15_end'] = alt_end
            flight_path['descent']['h_descent_15_10_start'] = alt_end # m
            flight_path['descent']['h_descent_15_10_end'] = alt_end # m
        
        else: # if cruise altitude is greater than 10000 ft
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
            M_climb_5_10 = v_climb_5_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound
            flight_path['climb']['M_climb_5_10'] = M_climb_5_10 # m/s
            flight_path['climb']['t_climb_5_10'] = t_climb_5_10 # s
            flight_path['climb']['s_climb_5_10'] = s_climb_5_10 # m
            flight_path['climb']['h_climb_5_10_start'] = alt_start # m
            flight_path['climb']['h_climb_5_10_end'] = alt_end # m

            
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
            M_descent_10_5 = v_descent_10_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound
            flight_path['descent']['M_descent_10_5'] = M_descent_10_5 # m/s
            flight_path['descent']['t_descent_10_5'] = t_descent_10_5 # s
            flight_path['descent']['s_descent_10_5'] = s_descent_10_5 # m
            flight_path['descent']['h_descent_10_5_start'] = alt_end # m
            flight_path['descent']['h_descent_10_5_end'] = alt_start # m

            
            if alt_cruise <= ft_to_m(15000): # if ceiling is between 10000 and 15000 ft
                alt_start = ft_to_m(10000) # m
                alt_end   = alt_cruise # m
                
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
                M_climb_10_15 = v_climb_10_15 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['climb']['M_climb_10_15'] = M_climb_10_15 # m/s
                flight_path['climb']['t_climb_10_15'] = t_climb_10_15 # s
                flight_path['climb']['s_climb_10_15'] = s_climb_10_15 # m
                flight_path['climb']['h_climb_10_15_start'] = alt_start # m
                flight_path['climb']['h_climb_10_15_end'] = alt_end # m

                
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
                M_descent_15_10 = v_descent_15_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['descent']['M_descent_15_10'] = M_descent_15_10 # m/s
                flight_path['descent']['t_descent_15_10'] = t_descent_15_10 # s
                flight_path['descent']['s_descent_15_10'] = s_descent_15_10 # m
                flight_path['descent']['h_descent_15_10_start'] = alt_end # m
                flight_path['descent']['h_descent_15_10_end'] = alt_start # m

                # assign remaining climb and descent altitudes to the cruise altitude:
                flight_path['climb']['h_climb_15_24_start'] = alt_end # m
                flight_path['climb']['h_climb_15_24_end'] = alt_end # m
                flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                flight_path['climb']['h_climb_ceil_end'] = alt_end # m
                flight_path['descent']['h_descent_ceil_start'] = alt_end # m
                flight_path['descent']['h_descent_ceil_end'] = alt_end # m
                flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                flight_path['descent']['h_descent_24_15_end'] = alt_end

                
                
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
                M_climb_10_15 = v_climb_10_15 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['climb']['M_climb_10_15'] = M_climb_10_15 # m/s
                flight_path['climb']['t_climb_10_15'] = t_climb_10_15 # s
                flight_path['climb']['s_climb_10_15'] = s_climb_10_15 # m
                flight_path['climb']['h_climb_10_15_start'] = alt_start # m
                flight_path['climb']['h_climb_10_15_end'] = alt_end # m

                
                # Build Descent 15000-10000 ft
                if aircraft_data['descent_to_fl_100_IAS'] == 'no data':
                    v_descent_15_10 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end)/2)[0]
                    w_descent_15_10 = w_descent_10_5
                else:
                    v_descent_15_10 = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_start + alt_end)/2)[0]
                    w_descent_15_10 = ftmin_to_ms(aircraft_data['descent_to_fl_100_ROD'])
                gs_descent_15_10 = (v_descent_15_10**2 - w_descent_15_10**2) ** 0.5
                
                t_descent_15_10 = (alt_end - alt_start) / w_descent_15_10 # s
                s_descent_15_10 = gs_descent_15_10 * t_descent_15_10
                M_descent_15_10 = v_descent_15_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['descent']['M_descent_15_10'] = M_descent_15_10 # m/s
                flight_path['descent']['t_descent_15_10'] = t_descent_15_10 # s
                flight_path['descent']['s_descent_15_10'] = s_descent_15_10 # m
                flight_path['descent']['h_descent_15_10_start'] = alt_end # m
                flight_path['descent']['h_descent_15_10_end'] = alt_start # m

                
                if alt_cruise <= ft_to_m(24000): # if ceiling is between 15000 and 24000 ft
                    alt_start = ft_to_m(15000) # m
                    alt_end   = alt_cruise # m
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
                    M_climb_15_24 = v_climb_15_24 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                    flight_path['climb']['M_climb_15_24'] = M_climb_15_24 # m/s
                    flight_path['climb']['t_climb_15_24'] = t_climb_15_24 # s
                    flight_path['climb']['s_climb_15_24'] = s_climb_15_24 # m
                    flight_path['climb']['h_climb_15_24_start'] = alt_start # m
                    flight_path['climb']['h_climb_15_24_end'] = alt_end # m

                        
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
                    M_descent_24_15 = v_descent_24_15 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                    flight_path['descent']['M_descent_24_15'] = M_descent_24_15 # m/s
                    flight_path['descent']['t_descent_24_15'] = t_descent_24_15 # s
                    flight_path['descent']['s_descent_24_15'] = s_descent_24_15 # m
                    flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                    flight_path['descent']['h_descent_24_15_end'] = alt_start # m

                    
                    # assign remaining climb and descent altitudes to the cruise altitude:
                    flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                    flight_path['climb']['h_climb_ceil_end'] = alt_end # m
                    flight_path['descent']['h_descent_ceil_start'] = alt_end # m
                    flight_path['descent']['h_descent_ceil_end'] = alt_end # m
                    
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
                    M_climb_15_24 = v_climb_15_24 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                    flight_path['climb']['M_climb_15_24'] = M_climb_15_24 # m/s
                    flight_path['climb']['t_climb_15_24'] = t_climb_15_24 # s
                    flight_path['climb']['s_climb_15_24'] = s_climb_15_24 # m
                    flight_path['climb']['h_climb_15_24_start'] = alt_start # m
                    flight_path['climb']['h_climb_15_24_end'] = alt_end # m

                    
                    # Build Climb 24000-ceiling feet
                    if aircraft_data['mach_climb_MACH'] == 'no data':
                        v_climb_ceil = icet(kts_to_ms(aircraft_data['climb_to_fl_240_IAS']), (alt_end + alt_cruise) /2)[0]
                        w_climb_ceil = w_climb_15_24
                    else:
                        v_climb_ceil = aircraft_data['mach_climb_MACH'] * Atmosphere((alt_end + alt_cruise)/2).speed_of_sound # m/s, TAS
                        w_climb_ceil = ftmin_to_ms(aircraft_data['mach_climb_ROC'])
                    gs_climb_ceil = (v_climb_ceil**2 - w_climb_ceil**2) ** 0.5 # ground speed and thus distance covered
                    
                    t_climb_ceil = (alt_cruise - alt_end) / w_climb_ceil # s
                    s_climb_ceil = gs_climb_ceil * t_climb_ceil
                    M_climb_ceil = v_climb_ceil / Atmosphere((alt_end + alt_cruise)/2).speed_of_sound # m/s
                    flight_path['climb']['M_climb_ceil'] = M_climb_ceil # m/s
                    flight_path['climb']['t_climb_ceil'] = t_climb_ceil # s
                    flight_path['climb']['s_climb_ceil'] = s_climb_ceil # m
                    flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                    flight_path['climb']['h_climb_ceil_end'] = alt_cruise # m

                    
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
                    M_descent_24_15 = v_descent_24_15 / Atmosphere((alt_start + alt_end)/2).speed_of_sound # m/s
                    flight_path['descent']['M_descent_24_15'] = M_descent_24_15 # m/s
                    flight_path['descent']['t_descent_24_15'] = t_descent_24_15 # s
                    flight_path['descent']['s_descent_24_15'] = s_descent_24_15 # m
                    flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                    flight_path['descent']['h_descent_24_15_end'] = alt_start # m

                    
                    # Build Descent ceiling-24000 ft
                    if aircraft_data['initial_descent_to_fl_240_MACH'] == 'no data':
                        v_descent_ceil = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_end + alt_cruise)/2)[0]
                        w_descent_ceil = w_descent_24_15
                    else:
                        v_descent_ceil = aircraft_data['initial_descent_to_fl_240_MACH'] * Atmosphere((alt_end + alt_cruise)/2).speed_of_sound
                        w_descent_ceil = ftmin_to_ms(aircraft_data['initial_descent_to_fl_240_ROD'])
                    gs_descent_ceil = (v_descent_ceil**2 - w_descent_ceil**2) ** 0.5 # ground speed and thus distance covered
                    
                    t_descent_ceil = (alt_cruise - alt_end) / w_descent_ceil # s
                    s_descent_ceil = gs_descent_ceil * t_descent_ceil
                    M_descent_ceil = v_descent_ceil / Atmosphere((alt_end + alt_cruise)/2).speed_of_sound # m/s
                    flight_path['descent']['M_descent_ceil'] = M_descent_ceil # m/s
                    flight_path['descent']['t_descent_ceil'] = t_descent_ceil # s
                    flight_path['descent']['s_descent_ceil'] = s_descent_ceil # m
                    flight_path['descent']['h_descent_ceil_start'] = alt_cruise # m
                    flight_path['descent']['h_descent_ceil_end'] = alt_end # m

                    
    # Build cruise phase
    # only use mach cruise above 24kft, otherwise use TAS
    if alt_cruise > ft_to_m(24000): # if cruise altitude is greater than 24000 ft, prefer mach
        if aircraft_data['cruise_MACH'] == 'no data':
            v_cruise = kts_to_ms(aircraft_data['cruise_TAS']) # m/s, TAS
        else:
            v_cruise = aircraft_data['cruise_MACH'] * Atmosphere(alt_cruise).speed_of_sound # m/s, TAS
    else: # if cruise altitude is less than 24000 ft, prefer TAS
        if aircraft_data['cruise_TAS'] == 'no data':
            v_cruise = aircraft_data['cruise_MACH'] * Atmosphere(ft_to_m(24000)).speed_of_sound # m/s, TAS
        else:
            v_cruise = kts_to_ms(aircraft_data['cruise_TAS'])
    w_cruise = 0 # assumes no climbing 
    gs_cruise = (v_cruise**2 - w_cruise**2) ** 0.5 # ground speed and thus distance covered
    s_cruise = gs_cruise * t_cruise # m
    
    s_tol = (sum(value for key, value in flight_path['climb'].items() if key.startswith('s_')) +
             sum(value for key, value in flight_path['descent'].items() if key.startswith('s_')))
    t_tol = (sum(value for key, value in flight_path['climb'].items() if key.startswith('t_')) +
             sum(value for key, value in flight_path['descent'].items() if key.startswith('t_')))
    s_total = s_tol + s_cruise # m
    t_total = t_tol + t_cruise # s
    
    
    # if the 10 minute cruise profile results in a distance longer than the requested great circle distance, throw an error
    if s_cruise > gc_dist:
        raise ValueError(f"Requested mission distance of {gc_dist} m is shorter than a 10 minute cruise, which will go {s_cruise} m. Please request a total distance longer than {s_cruise} m.")
    # if the default cruise profile results in a distance shorter than the requested great circle distance, decrease the cruise altitude
    elif s_total < gc_dist:
        s_cruise = gc_dist - s_tol # m
        t_cruise = s_cruise / gs_cruise
        s_total = s_tol + s_cruise # m
        t_total = t_tol + t_cruise
    # otherwise, if the default cruise profile results in a distance too long, decrease the cruise altitude
    else:
        counter = 0
        while s_total - gc_dist > s_tolerance and counter < 100: # not wrapping in absolute value to terminate loop if we get to an undershoot, will get skipped if we are already under the tolerance
            # initialize to zero values:
            flight_path['climb'] = {
                's_climb_0_5': 0,
                't_climb_0_5': 0,
                'M_climb_0_5': 0,
                'h_climb_0_5_start': 0,
                'h_climb_0_5_end': 0,
                's_climb_5_10': 0,
                't_climb_5_10': 0,
                'M_climb_5_10': 0,
                'h_climb_5_10_start': 0,
                'h_climb_5_10_end': 0,
                's_climb_10_15': 0,
                't_climb_10_15': 0,
                'M_climb_10_15': 0,
                'h_climb_10_15_start': 0,
                'h_climb_10_15_end': 0,
                's_climb_15_24': 0,
                't_climb_15_24': 0,
                'M_climb_15_24': 0,
                'h_climb_15_24_start': 0,
                'h_climb_15_24_end': 0,
                's_climb_ceil': 0,
                't_climb_ceil': 0,
                'M_climb_ceil': 0,
                'h_climb_ceil_start': 0,
                'h_climb_ceil_end': 0
            }
            
            flight_path['cruise'] = {
                's_cruise': 0,
                't_cruise': 0,
                'M_cruise': 0,
                'h_cruise_start': 0,
                'h_cruise_end': 0
            }

            flight_path['descent'] = {
                's_descent_ceil': 0,
                't_descent_ceil': 0,
                'M_descent_ceil': 0,
                'h_descent_ceil_start': 0,
                'h_descent_ceil_end': 0,
                's_descent_24_15': 0,
                't_descent_24_15': 0,
                'M_descent_24_15': 0,
                'h_descent_24_15_start': 0,
                'h_descent_24_15_end': 0,
                's_descent_15_10': 0,
                't_descent_15_10': 0,
                'M_descent_15_10': 0,
                'h_descent_15_10_start': 0,
                'h_descent_15_10_end': 0,
                's_descent_10_5': 0,
                't_descent_10_5': 0,
                'M_descent_10_5': 0,
                'h_descent_10_5_start': 0,
                'h_descent_10_5_end': 0,
                's_descent_5_0': 0,
                't_descent_5_0': 0,
                'M_descent_5_0': 0,
                'h_descent_5_0_start': 0,
                'h_descent_5_0_end': 0
            }
            
            # alt_cruise *= ft_to_m(alt_decrement)  # decrease the cruise alitude by 1kft. may be faster/more accurate if use amount of overshoot as a scaling factor
            alt_cruise *= 0.99
            # alt_cruise -= ft_to_m(alt_decrement)
            # build new flight at given altitude
            # Build climb and descent phases 
            if alt_cruise <= ft_to_m(5000): # if cruise altitude is between 0 and 5000 ft
                alt_start = ft_to_m(0) # m
                alt_end = alt_cruise # m
                # Build climb 0-cruise feet
                if aircraft_data['initial_climb_to_5000ft_IAS'] == 'no data':
                    v_climb_0_5 = 0
                    w_climb_0_5 = 0
                else: 
                    v_climb_0_5 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_end/2))[0] # m/s, TAS
                    w_climb_0_5 = ftmin_to_ms(aircraft_data['initial_climb_to_5000ft_ROC']) # m/s
                gs_climb_0_5 = (v_climb_0_5**2 - w_climb_0_5**2) ** 0.5 # ground speed and thus distance covered
                
                t_climb_0_5 = (alt_end - alt_start) / w_climb_0_5 # s
                s_climb_0_5 = gs_climb_0_5 * t_climb_0_5
                M_climb_0_5 = v_climb_0_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['climb']['M_climb_0_5'] = M_climb_0_5 # m/s
                flight_path['climb']['t_climb_0_5'] = t_climb_0_5 # s
                flight_path['climb']['s_climb_0_5'] = s_climb_0_5 # m
                flight_path['climb']['h_climb_0_5_start'] = alt_start # m
                flight_path['climb']['h_climb_0_5_end'] = alt_end # m

                
                # Build Descent cruise-0 feet
                if aircraft_data['approach_IAS'] == 'no data':
                    v_descent_5_0 = 0
                    w_descent_5_0 = 0
                else:
                    v_descent_5_0 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_end/2))[0] # m/s, TAS
                    w_descent_5_0 = ftmin_to_ms(aircraft_data['approach_ROD'])
                gs_descent_5_0 = (v_descent_5_0**2 - w_descent_5_0**2) ** 0.5
                
                t_descent_5_0 = (alt_end - alt_start) / w_descent_5_0
                s_descent_5_0 = gs_descent_5_0 * t_descent_5_0
                M_descent_5_0 = v_descent_5_0 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['descent']['M_descent_5_0'] = M_descent_5_0 # m/s
                flight_path['descent']['t_descent_5_0'] = t_descent_5_0 # s
                flight_path['descent']['s_descent_5_0'] = s_descent_5_0 # m
                flight_path['descent']['h_descent_5_0_start'] = alt_end # m
                flight_path['descent']['h_descent_5_0_end'] = alt_start # m

                
                # assign remaining climb and descent altitudes to the cruise altitude:
                flight_path['climb']['h_climb_5_10_start'] = alt_end # m
                flight_path['climb']['h_climb_5_10_end'] = alt_end # m
                flight_path['climb']['h_climb_10_15_start'] = alt_end # m
                flight_path['climb']['h_climb_10_15_end'] = alt_end # m.
                flight_path['climb']['h_climb_15_24_start'] = alt_end # m
                flight_path['climb']['h_climb_15_24_end'] = alt_end # m
                flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                flight_path['climb']['h_climb_ceil_end'] = alt_end # m
                flight_path['descent']['h_descent_ceil_start'] = alt_end # m
                flight_path['descent']['h_descent_ceil_end'] = alt_end # m
                flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                flight_path['descent']['h_descent_24_15_end'] = alt_end
                flight_path['descent']['h_descent_15_10_start'] = alt_end # m
                flight_path['descent']['h_descent_15_10_end'] = alt_end # m
                flight_path['descent']['h_descent_10_5_start'] = alt_end # m
                flight_path['descent']['h_descent_10_5_end'] = alt_end # m


            elif alt_cruise > ft_to_m(5000): # if cruise altitude is greater than 5000 ft
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
                M_climb_0_5 = v_climb_0_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['climb']['M_climb_0_5'] = M_climb_0_5 # m/s
                flight_path['climb']['t_climb_0_5'] = t_climb_0_5 # s
                flight_path['climb']['s_climb_0_5'] = s_climb_0_5 # m
                flight_path['climb']['h_climb_0_5_start'] = alt_start # m
                flight_path['climb']['h_climb_0_5_end'] = alt_end # m

                        
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
                M_descent_5_0 = v_descent_5_0 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                flight_path['descent']['M_descent_5_0'] = M_descent_5_0 # m/s
                flight_path['descent']['t_descent_5_0'] = t_descent_5_0 # s
                flight_path['descent']['s_descent_5_0'] = s_descent_5_0 # m
                flight_path['descent']['h_descent_5_0_start'] = alt_end # m
                flight_path['descent']['h_descent_5_0_end'] = alt_start

                
                if alt_cruise <= ft_to_m(10000): # if cruise altitude is between 5000 and 10000 ft
                    alt_start = ft_to_m(5000) # m
                    alt_end= alt_cruise # m
                    # Build Climb 5000-ceiling feet
                    if aircraft_data['climb_to_fl_150_IAS'] == 'no data':
                        v_climb_5_10 = icet(kts_to_ms(aircraft_data['initial_climb_to_5000ft_IAS']), (alt_start + alt_end) /2)[0]
                        w_climb_5_10 = w_climb_0_5
                    else:
                        v_climb_5_10 = icet(kts_to_ms(aircraft_data['climb_to_fl_150_IAS']), (alt_start + alt_end) /2)[0]
                        w_climb_5_10 = ftmin_to_ms(aircraft_data['climb_to_fl_150_ROC'])
                    gs_climb_5_10 = (v_climb_5_10**2 - w_climb_5_10**2) ** 0.5 # ground speed and thus distance covered
                    
                    t_climb_5_10 = (alt_end - alt_start) / w_climb_5_10 # s
                    s_climb_5_10 = gs_climb_5_10 * t_climb_5_10
                    M_climb_5_10 = v_climb_5_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                    flight_path['climb']['M_climb_5_10'] = M_climb_5_10 # m/s
                    flight_path['climb']['t_climb_5_10'] = t_climb_5_10 # s
                    flight_path['climb']['s_climb_5_10'] = s_climb_5_10 # m
                    flight_path['climb']['h_climb_5_10_start'] = alt_start # m
                    flight_path['climb']['h_climb_5_10_end'] = alt_end # m

                    
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
                    M_descent_10_5 = v_descent_10_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                    flight_path['descent']['M_descent_10_5'] = M_descent_10_5 # m/s
                    flight_path['descent']['t_descent_10_5'] = t_descent_10_5 # s
                    flight_path['descent']['s_descent_10_5'] = s_descent_10_5 # m
                    flight_path['descent']['h_descent_10_5_start'] = alt_end # m
                    flight_path['descent']['h_descent_10_5_end'] = alt_start # m

                # assign remaining climb and descent altitudes to the cruise altitude:
                    flight_path['climb']['h_climb_10_15_start'] = alt_end # m
                    flight_path['climb']['h_climb_10_15_end'] = alt_end # m.
                    flight_path['climb']['h_climb_15_24_start'] = alt_end # m
                    flight_path['climb']['h_climb_15_24_end'] = alt_end # m
                    flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                    flight_path['climb']['h_climb_ceil_end'] = alt_end # m
                    flight_path['descent']['h_descent_ceil_start'] = alt_end # m
                    flight_path['descent']['h_descent_ceil_end'] = alt_end # m
                    flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                    flight_path['descent']['h_descent_24_15_end'] = alt_end
                    flight_path['descent']['h_descent_15_10_start'] = alt_end # m
                    flight_path['descent']['h_descent_15_10_end'] = alt_end # m
                
                else: # if cruise altitude is greater than 10000 ft
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
                    M_climb_5_10 = v_climb_5_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound
                    flight_path['climb']['M_climb_5_10'] = M_climb_5_10 # m/s
                    flight_path['climb']['t_climb_5_10'] = t_climb_5_10 # s
                    flight_path['climb']['s_climb_5_10'] = s_climb_5_10 # m
                    flight_path['climb']['h_climb_5_10_start'] = alt_start # m
                    flight_path['climb']['h_climb_5_10_end'] = alt_end # m

                    
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
                    M_descent_10_5 = v_descent_10_5 / Atmosphere((alt_start+alt_end)/2).speed_of_sound
                    flight_path['descent']['M_descent_10_5'] = M_descent_10_5 # m/s
                    flight_path['descent']['t_descent_10_5'] = t_descent_10_5 # s
                    flight_path['descent']['s_descent_10_5'] = s_descent_10_5 # m
                    flight_path['descent']['h_descent_10_5_start'] = alt_end # m
                    flight_path['descent']['h_descent_10_5_end'] = alt_start # m

                    
                    if alt_cruise <= ft_to_m(15000): # if ceiling is between 10000 and 15000 ft
                        alt_start = ft_to_m(10000) # m
                        alt_end   = alt_cruise # m
                        
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
                        M_climb_10_15 = v_climb_10_15 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                        flight_path['climb']['M_climb_10_15'] = M_climb_10_15 # m/s
                        flight_path['climb']['t_climb_10_15'] = t_climb_10_15 # s
                        flight_path['climb']['s_climb_10_15'] = s_climb_10_15 # m
                        flight_path['climb']['h_climb_10_15_start'] = alt_start # m
                        flight_path['climb']['h_climb_10_15_end'] = alt_end # m

                        
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
                        M_descent_15_10 = v_descent_15_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                        flight_path['descent']['M_descent_15_10'] = M_descent_15_10 # m/s
                        flight_path['descent']['t_descent_15_10'] = t_descent_15_10 # s
                        flight_path['descent']['s_descent_15_10'] = s_descent_15_10 # m
                        flight_path['descent']['h_descent_15_10_start'] = alt_end # m
                        flight_path['descent']['h_descent_15_10_end'] = alt_start # m

                        # assign remaining climb and descent altitudes to the cruise altitude:
                        flight_path['climb']['h_climb_15_24_start'] = alt_end # m
                        flight_path['climb']['h_climb_15_24_end'] = alt_end # m
                        flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                        flight_path['climb']['h_climb_ceil_end'] = alt_end # m
                        flight_path['descent']['h_descent_ceil_start'] = alt_end # m
                        flight_path['descent']['h_descent_ceil_end'] = alt_end # m
                        flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                        flight_path['descent']['h_descent_24_15_end'] = alt_end

                        
                        
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
                        M_climb_10_15 = v_climb_10_15 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                        flight_path['climb']['M_climb_10_15'] = M_climb_10_15 # m/s
                        flight_path['climb']['t_climb_10_15'] = t_climb_10_15 # s
                        flight_path['climb']['s_climb_10_15'] = s_climb_10_15 # m
                        flight_path['climb']['h_climb_10_15_start'] = alt_start # m
                        flight_path['climb']['h_climb_10_15_end'] = alt_end # m

                        
                        # Build Descent 15000-10000 ft
                        if aircraft_data['descent_to_fl_100_IAS'] == 'no data':
                            v_descent_15_10 = icet(kts_to_ms(aircraft_data['approach_IAS']), (alt_start + alt_end)/2)[0]
                            w_descent_15_10 = w_descent_10_5
                        else:
                            v_descent_15_10 = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_start + alt_end)/2)[0]
                            w_descent_15_10 = ftmin_to_ms(aircraft_data['descent_to_fl_100_ROD'])
                        gs_descent_15_10 = (v_descent_15_10**2 - w_descent_15_10**2) ** 0.5
                        
                        t_descent_15_10 = (alt_end - alt_start) / w_descent_15_10 # s
                        s_descent_15_10 = gs_descent_15_10 * t_descent_15_10
                        M_descent_15_10 = v_descent_15_10 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                        flight_path['descent']['M_descent_15_10'] = M_descent_15_10 # m/s
                        flight_path['descent']['t_descent_15_10'] = t_descent_15_10 # s
                        flight_path['descent']['s_descent_15_10'] = s_descent_15_10 # m
                        flight_path['descent']['h_descent_15_10_start'] = alt_end # m
                        flight_path['descent']['h_descent_15_10_end'] = alt_start # m

                        
                        if alt_cruise <= ft_to_m(24000): # if ceiling is between 15000 and 24000 ft
                            alt_start = ft_to_m(15000) # m
                            alt_end   = alt_cruise # m
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
                            M_climb_15_24 = v_climb_15_24 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                            flight_path['climb']['M_climb_15_24'] = M_climb_15_24 # m/s
                            flight_path['climb']['t_climb_15_24'] = t_climb_15_24 # s
                            flight_path['climb']['s_climb_15_24'] = s_climb_15_24 # m
                            flight_path['climb']['h_climb_15_24_start'] = alt_start # m
                            flight_path['climb']['h_climb_15_24_end'] = alt_end # m

                                
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
                            M_descent_24_15 = v_descent_24_15 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                            flight_path['descent']['M_descent_24_15'] = M_descent_24_15 # m/s
                            flight_path['descent']['t_descent_24_15'] = t_descent_24_15 # s
                            flight_path['descent']['s_descent_24_15'] = s_descent_24_15 # m
                            flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                            flight_path['descent']['h_descent_24_15_end'] = alt_start # m

                            
                            # assign remaining climb and descent altitudes to the cruise altitude:
                            flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                            flight_path['climb']['h_climb_ceil_end'] = alt_end # m
                            flight_path['descent']['h_descent_ceil_start'] = alt_end # m
                            flight_path['descent']['h_descent_ceil_end'] = alt_end # m
                            
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
                            M_climb_15_24 = v_climb_15_24 / Atmosphere((alt_start+alt_end)/2).speed_of_sound # m/s
                            flight_path['climb']['M_climb_15_24'] = M_climb_15_24 # m/s
                            flight_path['climb']['t_climb_15_24'] = t_climb_15_24 # s
                            flight_path['climb']['s_climb_15_24'] = s_climb_15_24 # m
                            flight_path['climb']['h_climb_15_24_start'] = alt_start # m
                            flight_path['climb']['h_climb_15_24_end'] = alt_end # m

                            
                            # Build Climb 24000-ceiling feet
                            if aircraft_data['mach_climb_MACH'] == 'no data':
                                v_climb_ceil = icet(kts_to_ms(aircraft_data['climb_to_fl_240_IAS']), (alt_end + alt_cruise) /2)[0]
                                w_climb_ceil = w_climb_15_24
                            else:
                                v_climb_ceil = aircraft_data['mach_climb_MACH'] * Atmosphere((alt_end + alt_cruise)/2).speed_of_sound # m/s, TAS
                                w_climb_ceil = ftmin_to_ms(aircraft_data['mach_climb_ROC'])
                            gs_climb_ceil = (v_climb_ceil**2 - w_climb_ceil**2) ** 0.5 # ground speed and thus distance covered
                            
                            t_climb_ceil = (alt_cruise - alt_end) / w_climb_ceil # s
                            s_climb_ceil = gs_climb_ceil * t_climb_ceil
                            M_climb_ceil = v_climb_ceil / Atmosphere((alt_end + alt_cruise)/2).speed_of_sound # m/s
                            flight_path['climb']['M_climb_ceil'] = M_climb_ceil # m/s
                            flight_path['climb']['t_climb_ceil'] = t_climb_ceil # s
                            flight_path['climb']['s_climb_ceil'] = s_climb_ceil # m
                            flight_path['climb']['h_climb_ceil_start'] = alt_end # m
                            flight_path['climb']['h_climb_ceil_end'] = alt_cruise # m

                            
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
                            M_descent_24_15 = v_descent_24_15 / Atmosphere((alt_start + alt_end)/2).speed_of_sound # m/s
                            flight_path['descent']['M_descent_24_15'] = M_descent_24_15 # m/s
                            flight_path['descent']['t_descent_24_15'] = t_descent_24_15 # s
                            flight_path['descent']['s_descent_24_15'] = s_descent_24_15 # m
                            flight_path['descent']['h_descent_24_15_start'] = alt_end # m
                            flight_path['descent']['h_descent_24_15_end'] = alt_start # m

                            
                            # Build Descent ceiling-24000 ft
                            if aircraft_data['initial_descent_to_fl_240_MACH'] == 'no data':
                                v_descent_ceil = icet(kts_to_ms(aircraft_data['descent_to_fl_100_IAS']), (alt_end + alt_cruise)/2)[0]
                                w_descent_ceil = w_descent_24_15
                            else:
                                v_descent_ceil = aircraft_data['initial_descent_to_fl_240_MACH'] * Atmosphere((alt_end + alt_cruise)/2).speed_of_sound
                                w_descent_ceil = ftmin_to_ms(aircraft_data['initial_descent_to_fl_240_ROD'])
                            gs_descent_ceil = (v_descent_ceil**2 - w_descent_ceil**2) ** 0.5 # ground speed and thus distance covered
                            
                            t_descent_ceil = (alt_cruise - alt_end) / w_descent_ceil # s
                            s_descent_ceil = gs_descent_ceil * t_descent_ceil
                            M_descent_ceil = v_descent_ceil / Atmosphere((alt_end + alt_cruise)/2).speed_of_sound # m/s
                            flight_path['descent']['M_descent_ceil'] = M_descent_ceil # m/s
                            flight_path['descent']['t_descent_ceil'] = t_descent_ceil # s
                            flight_path['descent']['s_descent_ceil'] = s_descent_ceil # m
                            flight_path['descent']['h_descent_ceil_start'] = alt_cruise # m
                            flight_path['descent']['h_descent_ceil_end'] = alt_end # m
                    
                    
            # Build cruise phase
            # only use mach cruise above 24kft, otherwise use TAS
            if alt_cruise > ft_to_m(24000): # if cruise altitude is greater than 24000 ft, prefer mach
                if aircraft_data['cruise_MACH'] == 'no data':
                    v_cruise = kts_to_ms(aircraft_data['cruise_TAS']) # m/s, TAS
                else:
                    v_cruise = aircraft_data['cruise_MACH'] * Atmosphere(alt_cruise).speed_of_sound # m/s, TAS
            else: # if cruise altitude is less than 24000 ft, prefer TAS
                if aircraft_data['cruise_TAS'] == 'no data':
                    v_cruise = aircraft_data['cruise_MACH'] * Atmosphere(ft_to_m(24000)).speed_of_sound # m/s, TAS
                else:
                    v_cruise = kts_to_ms(aircraft_data['cruise_TAS'])
                    
            w_cruise = 0 # assumes no climbing 
            gs_cruise = (v_cruise**2 - w_cruise**2) ** 0.5 # ground speed and thus distance covered
            s_cruise = v_cruise * t_cruise # m
            t_cruise = min_cruise_duration # s, cruise time
            
            # calculate the TOL distance and time:
            s_tol = (sum(value for key, value in flight_path['climb'].items() if key.startswith('s_')) +
                     sum(value for key, value in flight_path['descent'].items() if key.startswith('s_')))
            t_tol = (sum(value for key, value in flight_path['climb'].items() if key.startswith('t_')) +
                     sum(value for key, value in flight_path['descent'].items() if key.startswith('t_')))
            s_total = s_tol + s_cruise # m
            t_total = t_tol + t_cruise # s
            # print("alt: ", alt_cruise, " error: ", s_total - gc_dist, "gc_dist: ", gc_dist)
            counter +=1
            
    flight_path['cruise'] = {'s_cruise': s_cruise,
                             't_cruise': t_cruise,
                             'M_cruise': v_cruise / Atmosphere(alt_cruise).speed_of_sound,
                             'h_cruise_start': alt_cruise,
                             'h_cruise_end': alt_cruise}
    
    ########################################################################
    # add emissions database to the flight path:
    ########################################################################

    fuel_flow_to = aircraft_data['Fuel Flow T/O (kg/sec)'] # kg/s to lb/hr
    fuel_flow_co = aircraft_data['Fuel Flow C/O (kg/sec)'] # kg/s to lb/hr
    fuel_flow_app = aircraft_data['Fuel Flow App (kg/sec)'] # kg/s to lb/hr
    fuel_flow_idle = aircraft_data['Fuel Flow Idle (kg/sec)'] # kg/s to lb/hr
    fuel_flow_cruise = 0.5 * fuel_flow_to # kg/sec, cruise fuel flow is ~1/2 of T/O fuel flow
    
    # TO
    [HC_ei_0_5, CO_ei_0_5, NOx_ei_0_5] = bffm2(aircraft_data_df,
                                              (flight_path['climb']['h_climb_0_5_start'] + flight_path['climb']['h_climb_0_5_end']) / 2,
                                              flight_path['climb']['M_climb_0_5'],
                                              fuel_flow_to * 7936.64) #<-- kg/s to lb/hr conversion factor
    # C/O
    [HC_ei_5_10, CO_ei_5_10, NOx_ei_5_10] = bffm2(aircraft_data_df,
                                              (flight_path['climb']['h_climb_5_10_start'] + flight_path['climb']['h_climb_5_10_end']) / 2,
                                              flight_path['climb']['M_climb_5_10'],
                                              fuel_flow_co * 7936.64)
    [HC_ei_10_15, CO_ei_10_15, NOx_ei_10_15] = bffm2(aircraft_data_df,
                                              (flight_path['climb']['h_climb_10_15_start'] + flight_path['climb']['h_climb_10_15_end']) / 2,
                                              flight_path['climb']['M_climb_10_15'],
                                              fuel_flow_co * 7936.64)
    [HC_ei_15_24, CO_ei_15_24, NOx_ei_15_24] = bffm2(aircraft_data_df,
                                              (flight_path['climb']['h_climb_15_24_start'] + flight_path['climb']['h_climb_15_24_end']) / 2,
                                              flight_path['climb']['M_climb_15_24'],
                                              fuel_flow_co * 7936.64)
    [HC_ei_ceil, CO_ei_ceil, NOx_ei_ceil] = bffm2(aircraft_data_df,
                                              (flight_path['climb']['h_climb_ceil_start'] + flight_path['climb']['h_climb_ceil_end']) / 2,
                                              flight_path['climb']['M_climb_ceil'],
                                              fuel_flow_co * 7936.64)
    #Cruise
    [HC_ei_cruise, CO_ei_cruise, NOx_ei_cruise] = bffm2(aircraft_data_df,
                                              (flight_path['cruise']['h_cruise_start'] + flight_path['cruise']['h_cruise_end']) / 2,
                                              flight_path['cruise']['M_cruise'],
                                              fuel_flow_cruise * 7936.64)
    # Descent
    [HC_ei_ceild, CO_ei_ceild, NOx_ei_ceild] = bffm2(aircraft_data_df,
                                              (flight_path['descent']['h_descent_ceil_start'] + flight_path['descent']['h_descent_ceil_end']) / 2,
                                              flight_path['descent']['M_descent_ceil'],
                                              fuel_flow_app * 7936.64)
    [HC_ei_24_15, CO_ei_24_15, NOx_ei_24_15] = bffm2(aircraft_data_df,
                                              (flight_path['descent']['h_descent_24_15_start'] + flight_path['descent']['h_descent_24_15_end']) / 2,
                                              flight_path['descent']['M_descent_24_15'],
                                              fuel_flow_app * 7936.64)
    [HC_ei_15_10, CO_ei_15_10, NOx_ei_15_10] = bffm2(aircraft_data_df,
                                              (flight_path['descent']['h_descent_15_10_start'] + flight_path['descent']['h_descent_15_10_end']) / 2,
                                              flight_path['descent']['M_descent_15_10'],
                                              fuel_flow_app * 7936.64)
    [HC_ei_10_5, CO_ei_10_5, NOx_ei_10_5] = bffm2(aircraft_data_df,
                                              (flight_path['descent']['h_descent_10_5_start'] + flight_path['descent']['h_descent_10_5_end']) / 2,
                                              flight_path['descent']['M_descent_10_5'],
                                              fuel_flow_app * 7936.64)
    [HC_ei_5_0, CO_ei_5_0, NOx_ei_5_0] = bffm2(aircraft_data_df,
                                              (flight_path['descent']['h_descent_5_0_start'] + flight_path['descent']['h_descent_5_0_end']) / 2,
                                              flight_path['descent']['M_descent_5_0'],
                                              fuel_flow_app * 7936.64)


    # Climb emissions: g = emission index (g/kg) * fuel flow (kg/s) * time (s)
    flight_path['climb']['HC_climb_0_5'] = HC_ei_0_5 * fuel_flow_to * flight_path['climb']['t_climb_0_5']
    flight_path['climb']['CO_climb_0_5'] = CO_ei_0_5 * fuel_flow_to * flight_path['climb']['t_climb_0_5']
    flight_path['climb']['NOx_climb_0_5'] = NOx_ei_0_5 * fuel_flow_to * flight_path['climb']['t_climb_0_5']
    
    flight_path['climb']['HC_climb_5_10'] = HC_ei_5_10 * fuel_flow_co * flight_path['climb']['t_climb_5_10']
    flight_path['climb']['CO_climb_5_10'] = CO_ei_5_10 * fuel_flow_co * flight_path['climb']['t_climb_5_10']
    flight_path['climb']['NOx_climb_5_10'] = NOx_ei_5_10 * fuel_flow_co * flight_path['climb']['t_climb_5_10']
    
    flight_path['climb']['HC_climb_10_15'] = HC_ei_10_15 * fuel_flow_co * flight_path['climb']['t_climb_10_15']
    flight_path['climb']['CO_climb_10_15'] = CO_ei_10_15 * fuel_flow_co * flight_path['climb']['t_climb_10_15']
    flight_path['climb']['NOx_climb_10_15'] = NOx_ei_10_15 * fuel_flow_co * flight_path['climb']['t_climb_10_15']

    flight_path['climb']['HC_climb_15_24'] = HC_ei_15_24 * fuel_flow_co * flight_path['climb']['t_climb_15_24']
    flight_path['climb']['CO_climb_15_24'] = CO_ei_15_24 * fuel_flow_co * flight_path['climb']['t_climb_15_24']
    flight_path['climb']['NOx_climb_15_24'] = NOx_ei_15_24 * fuel_flow_co * flight_path['climb']['t_climb_15_24']
    
    flight_path['climb']['HC_climb_ceil'] = HC_ei_ceil * fuel_flow_co * flight_path['climb']['t_climb_ceil']
    flight_path['climb']['CO_climb_ceil'] = CO_ei_ceil * fuel_flow_co * flight_path['climb']['t_climb_ceil']
    flight_path['climb']['NOx_climb_ceil'] = NOx_ei_ceil * fuel_flow_co * flight_path['climb']['t_climb_ceil']

    # Cruise emissions
    flight_path['cruise']['HC_cruise'] = HC_ei_cruise * fuel_flow_cruise * flight_path['cruise']['t_cruise']
    flight_path['cruise']['CO_cruise'] = CO_ei_cruise * fuel_flow_cruise * flight_path['cruise']['t_cruise']
    flight_path['cruise']['NOx_cruise'] = NOx_ei_cruise * fuel_flow_cruise * flight_path['cruise']['t_cruise']

    # Descent emissions
    flight_path['descent']['HC_descent_ceil'] = HC_ei_ceild * fuel_flow_app * flight_path['descent']['t_descent_ceil']
    flight_path['descent']['CO_descent_ceil'] = CO_ei_ceild * fuel_flow_app * flight_path['descent']['t_descent_ceil']
    flight_path['descent']['NOx_descent_ceil'] = NOx_ei_ceild * fuel_flow_app * flight_path['descent']['t_descent_ceil']

    flight_path['descent']['HC_descent_24_15'] = HC_ei_24_15 * fuel_flow_app * flight_path['descent']['t_descent_24_15']
    flight_path['descent']['CO_descent_24_15'] = CO_ei_24_15 * fuel_flow_app * flight_path['descent']['t_descent_24_15']
    flight_path['descent']['NOx_descent_24_15'] = NOx_ei_24_15 * fuel_flow_app * flight_path['descent']['t_descent_24_15']

    flight_path['descent']['HC_descent_15_10'] = HC_ei_15_10 * fuel_flow_app * flight_path['descent']['t_descent_15_10']
    flight_path['descent']['CO_descent_15_10'] = CO_ei_15_10 * fuel_flow_app * flight_path['descent']['t_descent_15_10']
    flight_path['descent']['NOx_descent_15_10'] = NOx_ei_15_10 * fuel_flow_app * flight_path['descent']['t_descent_15_10']

    flight_path['descent']['HC_descent_10_5'] = HC_ei_10_5 * fuel_flow_app * flight_path['descent']['t_descent_10_5']
    flight_path['descent']['CO_descent_10_5'] = CO_ei_10_5 * fuel_flow_app * flight_path['descent']['t_descent_10_5']
    flight_path['descent']['NOx_descent_10_5'] = NOx_ei_10_5 * fuel_flow_app * flight_path['descent']['t_descent_10_5']

    flight_path['descent']['HC_descent_5_0'] = HC_ei_5_0 * fuel_flow_app * flight_path['descent']['t_descent_5_0']
    flight_path['descent']['CO_descent_5_0'] = CO_ei_5_0 * fuel_flow_app * flight_path['descent']['t_descent_5_0']
    flight_path['descent']['NOx_descent_5_0'] = NOx_ei_5_0 * fuel_flow_app * flight_path['descent']['t_descent_5_0']

    # Convert everything to a basic python float
    for key in flight_path:
        for sub_key in flight_path[key]:
            flight_path[key][sub_key] = float(flight_path[key][sub_key])
    
    # return the results
    return flight_path