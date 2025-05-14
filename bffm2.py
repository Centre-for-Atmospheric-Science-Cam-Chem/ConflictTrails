from ambiance import Atmosphere
import numpy as np
from pa_to_psi import pa_to_psi
def bffm2(aircraft_data, altitude, mach, W_f):
    '''
    BFFM2 - Boeing Fuel Flow Model 2 is the second version of the Boeing Fuel Flow Model, which is a fuel flow model used to estimate the fuel consumption of commercial aircraft during flight. It is based on a combination of empirical data and theoretical models, and it takes into account various factors such as aircraft weight, altitude, speed, and engine performance.
    The BFFM2 model is used by airlines and aviation authorities to estimate fuel consumption and emissions for different flight profiles, and it is often used in conjunction with other models to provide a comprehensive analysis of aircraft performance.
    The BFFM2 model is a widely used tool in the aviation industry for fuel consumption and emissions estimation, and it is considered to be one of the most accurate models available for this purpose.
    The BFFM2 model is a proprietary model developed by Boeing, and it is based on extensive data collected from commercial aircraft operations. The model uses a combination of empirical data and theoretical models to estimate fuel consumption and emissions for different flight profiles, and it takes into account various factors such as aircraft weight, altitude, speed, and engine performance.
    '''
    # get the necessary data from the aircraft_data dictionary
    fuel_flow_icao = np.array([aircraft_data.iloc[0]['Fuel Flow T/O (kg/sec)'],
                               aircraft_data.iloc[0]['Fuel Flow C/O (kg/sec)'],
                               aircraft_data.iloc[0]['Fuel Flow App (kg/sec)'],
                               aircraft_data.iloc[0]['Fuel Flow Idle (kg/sec)']])
    print(fuel_flow_icao)                       

    HC_ei_vec = np.array([aircraft_data.iloc[0]['HC EI T/O (g/kg)'],
                          aircraft_data.iloc[0]['HC EI C/O (g/kg)'],
                          aircraft_data.iloc[0]['HC EI App (g/kg)'],
                          aircraft_data.iloc[0]['HC EI Idle (g/kg)']])
    print(HC_ei_vec)

    CO_ei_vec = np.array([aircraft_data.iloc[0]['CO EI T/O (g/kg)'],
                          aircraft_data.iloc[0]['CO EI C/O (g/kg)'],
                          aircraft_data.iloc[0]['CO EI App (g/kg)'],
                          aircraft_data.iloc[0]['CO EI Idle (g/kg)']])
    print(CO_ei_vec)

    NOx_ei_vec = np.array([aircraft_data.iloc[0]['NOx EI T/O (g/kg)'],
                           aircraft_data.iloc[0]['NOx EI C/O (g/kg)'],
                           aircraft_data.iloc[0]['NOx EI App (g/kg)'],
                           aircraft_data.iloc[0]['NOx EI Idle (g/kg)']])
    print(NOx_ei_vec)

    atmosphere = Atmosphere(altitude)
    P_amb = pa_to_psi(atmosphere.pressure)
    T_amb = Atmosphere.T2t(atmosphere.temperature)

    # 1. Calculate the ambient pressure and temperature at the given altitude:

    # Convert from kg/s to lbs/hr:
    fuel_flow_icao = fuel_flow_icao * 7936.64
    # 1. Installation Correction Factor (ICF) for the engine
    #                   T/O   C/O    App    Idle                
    icf_vec = np.array([1.01, 1.013, 1.020, 1.100])
    fuel_flow_icao = np.array(fuel_flow_icao * icf_vec)

    # 2. Use linear log-log fit on the emission index vs fuel flow rate
    def piecewise_fit(x, EI_vec, fuel_flow_icao_local):
        x = np.log(x)
        EI_vec = np.log(EI_vec)
        fuel_flow_icao_local = np.log(fuel_flow_icao_local)
        # Line between Idle and Approach:
        m_3_2 = (EI_vec[3] - EI_vec[2]) / (fuel_flow_icao_local[3] - fuel_flow_icao_local[2])
        b_3_2 = EI_vec[3] - m_3_2 * fuel_flow_icao_local[3]

        # Line at the average of T/0 and C/O:
        m_0_1 = 0
        b_0_1 = (EI_vec[0] + EI_vec[1])/2
        
        # find line between intersection and constant
        intersect_x = (b_0_1 - b_3_2) / m_3_2 if m_3_2 != 0 else float('inf')
        # If intersection is within [App, C/O]
        if fuel_flow_icao_local[2] <= intersect_x <= fuel_flow_icao_local[1]:
            if x >= intersect_x:
                return np.exp(b_0_1)
            else:
                return np.exp(m_3_2 * x + b_3_2)
        else: # if the intersection is outside of app, CO
            # find line between approach and average of climbout and takeoff:
            m_2_1 = (EI_vec[2] - b_0_1) / (fuel_flow_icao_local[2] - fuel_flow_icao_local[1])
            b_2_1 = EI_vec[2] - m_2_1 * fuel_flow_icao_local[2]
            if x >= fuel_flow_icao_local[1]: # if x is greater than climbout
                return np.exp(b_0_1)
            elif x <= fuel_flow_icao_local[2]: # if x is less than approach
                return np.exp(m_3_2 * x + b_3_2)
            else: # if x is between approach and climbout
                return np.exp(m_2_1 * x + b_2_1)
    # For NOx: Fit a piecewise continuous line through all four points
    # Create a piecewise linear point-to-point interpolator in log-log space
    EI_HC_ret = piecewise_fit(W_f, HC_ei_vec, fuel_flow_icao)
    EI_CO_ret = piecewise_fit(W_f, CO_ei_vec, fuel_flow_icao)
    EI_NOx_ret = np.interp(W_f, np.flip(fuel_flow_icao), np.flip(NOx_ei_vec))
    # Calculate the interpolated emission indices for the given fuel flow rate:
    ###
    # log_fuel_flow_evaluate = np.log10(fuel_flow_evaluate)
    ###
   
    
    # 2a. calculate the temp and pressure correction factors:
    
    
    del_amb = P_amb/14.696
    theta_amb = (T_amb + 273.15)/288.15
    # 2b. fuel flow values are further modified by the ambient values:
    W_ff = (W_f/del_amb) * (theta_amb**3.8) * np.exp(0.2*mach**2)
    # 2c. calculate the humidity correction factor, using modified EUCONTROL correction (0.37802)
    phi = 0.6 # relative humidity, assumed to be 60% for this calculation
    beta = 7.90298*(1-373.16/(T_amb + 273.16)) + 3.00571 + (5.02808) * np.log(373.16/(T_amb + 273.16)) + 1.3816*10**(-7) * (1-10**(11.344*(1-(T_amb + 273.16)/373.16))) + 8.1328 * 10**(-3) * (10**(3.49149*(1-373.16/(T_amb+273.16)))-1)
    P_v = (0.014504)*10**beta
    omega = (0.62198*phi*P_v)/(P_amb-0.37802*phi*P_v)
    H_corr_factor = -19*(omega-0.00634)
    
    # 3. Compute the EI:
    rEI_HC = piecewise_fit(W_ff, HC_ei_vec, fuel_flow_icao)
    rEI_CO = piecewise_fit(W_ff, CO_ei_vec, fuel_flow_icao)
    rEI_NOx = np.interp(W_ff, np.flip(fuel_flow_icao), np.flip(NOx_ei_vec))

    EI_HC_corr = rEI_HC * theta_amb**3.3 / (del_amb**1.02)
    EI_CO_corr = rEI_CO * theta_amb**3.3 / (del_amb**1.02)
    EI_NOx_corr = rEI_NOx * (del_amb**1.02 / theta_amb**3.3)**0.5 * np.exp(H_corr_factor)   
    
    # 4. Calculate the final EI:
    return np.array([EI_HC_corr, EI_CO_corr, EI_NOx_corr])