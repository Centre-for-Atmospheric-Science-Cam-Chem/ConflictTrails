from ambiance import Atmosphere
def icet(v_i, calibration_factor, altitude_m):
    """
    Calculate the equivalent airspeed (EAS) from indicated airspeed (IAS) using a calibration factor and altitude.

    :param indicated: Indicated airspeed in knots
    :param calibration_factor: Calibration factor for the airspeed indicator
    :param altitude_m: Altitude in meters
    :return: True airspeed (TAS) in m/s and Mach number
    """
    # Constants
    k = 1.4 # Ratio of specific heats for air
    
    P_0 = Atmosphere(0).pressure 
    P   = Atmosphere(altitude_m).pressure
    a_0 = Atmosphere(0).speed_of_sound
    a   = Atmosphere(altitude_m).speed_of_sound
    r_0 = Atmosphere(0).density
    r   = Atmosphere(altitude_m).density
    
    # Indicated -> Calibrated
    v_c = v_i + calibration_factor
    
    # Calibrated -> Equivalent   
    q = ((((k-1)/2 * (v_c/a_0)**2) + 1)**(k/(k-1)) - 1) * P_0
    M = (2/(k-1) *((q/P + 1)**(2/7) - 1))**0.5
    v_e = a_0*M*(P/P_0)**0.5
    
    # Equivalent -> True
    v_t = v_e*(r_0/r)**0.5
    
    return v_t, M