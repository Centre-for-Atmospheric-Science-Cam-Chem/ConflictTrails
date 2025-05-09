from ambiance import Atmosphere
import numpy as np
def bffm2(aircraft_data, altitude):
    '''
    BFFM2 - Boeing Fuel Flow Model 2 is the second version of the Boeing Fuel Flow Model, which is a fuel flow model used to estimate the fuel consumption of commercial aircraft during flight. It is based on a combination of empirical data and theoretical models, and it takes into account various factors such as aircraft weight, altitude, speed, and engine performance.
    The BFFM2 model is used by airlines and aviation authorities to estimate fuel consumption and emissions for different flight profiles, and it is often used in conjunction with other models to provide a comprehensive analysis of aircraft performance.
    The BFFM2 model is a widely used tool in the aviation industry for fuel consumption and emissions estimation, and it is considered to be one of the most accurate models available for this purpose.
    The BFFM2 model is a proprietary model developed by Boeing, and it is based on extensive data collected from commercial aircraft operations. The model uses a combination of empirical data and theoretical models to estimate fuel consumption and emissions for different flight profiles, and it takes into account various factors such as aircraft weight, altitude, speed, and engine performance.
    '''
    
    # convert to vector
    fuel_flow_vec = np.array([aircraft_data['Fuel Flow T/O (kg/sec)'], aircraft_data['Fuel Flow C/O (kg/sec)'], aircraft_data['Fuel Flow App (kg/sec)'], aircraft_data['Fuel Flow Idle (kg/sec)']])
    HC_ei_vec = np.array([aircraft_data['HC EI T/O (g/kg)'], aircraft_data['HC EI C/O (g/kg)'], aircraft_data['HC EI App (g/kg)'], aircraft_data['HC EI Idle (g/kg)']])
    CO_ei_vec = np.array([aircraft_data['CO EI T/O (g/kg)'], aircraft_data['CO EI C/O (g/kg)'], aircraft_data['CO EI App (g/kg)'], aircraft_data['CO EI Idle (g/kg)']])
    NOx_ei_vec = np.array([aircraft_data['NOx EI T/O (g/kg)'], aircraft_data['NOx EI C/O (g/kg)'], aircraft_data['NOx EI App (g/kg)'], aircraft_data['NOx EI Idle (g/kg)']])
    # Convert from kg/s to lbs/hr:
    Fuel_flow_vec = Fuel_flow_vec * 7936.64
    # 1. Installation Correction Factor (ICF) for the engine
    icf_vec = np.array([1.01, 1.013, 1.020, 1.100])
    Fuel_flow_vec = Fuel_flow_vec * icf_vec
    # 2. Use linear log-log fit on the emission index vs fuel flow rate
    log_fuel_flow_vec = np.log10(Fuel_flow_vec)
    log_HC_ei_vec = np.log10(HC_ei_vec)
    log_CO_ei_vec = np.log10(CO_ei_vec)
    log_NOx_ei_vec = np.log10(NOx_ei_vec)
    # Helper to compute average of T/O and C/O in log-log space
    log_HC_ei_avg_to_co = np.mean([log_HC_ei_vec[0], log_HC_ei_vec[1]])
    log_CO_ei_avg_to_co = np.mean([np.log10(CO_ei_vec[0]), np.log10(CO_ei_vec[1])])
    log_fuel_flow_co = log_fuel_flow_vec[1]
    log_fuel_flow_app = log_fuel_flow_vec[2]
    log_fuel_flow_idle = log_fuel_flow_vec[3]

    # For HC
    def log_HC_ei_piecewise(log_fuel_flow):
        # Line between Idle and App
        m1 = (log_HC_ei_vec[2] - log_HC_ei_vec[3]) / (log_fuel_flow_vec[2] - log_fuel_flow_vec[3])
        b1 = log_HC_ei_vec[3] - m1 * log_fuel_flow_vec[3]
        # Line between App and avg(T/O, C/O)
        m2 = (log_HC_ei_avg_to_co - log_HC_ei_vec[2]) / (log_fuel_flow_co - log_fuel_flow_vec[2])
        b2 = log_HC_ei_vec[2] - m2 * log_fuel_flow_vec[2]

        # Find intersection of first line and constant
        intersect_x = (log_HC_ei_avg_to_co - b1) / m1 if m1 != 0 else float('inf')
        # If intersection is within [App, C/O], use piecewise; else, use two segments
        if log_fuel_flow_app <= intersect_x <= log_fuel_flow_co:
            if log_fuel_flow <= intersect_x:
                return m1 * log_fuel_flow + b1
            else:
                return log_HC_ei_avg_to_co
        else:
            if log_fuel_flow <= log_fuel_flow_app:
                return m1 * log_fuel_flow + b1
            elif log_fuel_flow_app < log_fuel_flow <= log_fuel_flow_co:
                return m2 * log_fuel_flow + b2
            else:
                return log_HC_ei_avg_to_co

    # For CO
    log_CO_ei_app = np.log10(CO_ei_vec[2])
    log_CO_ei_idle = np.log10(CO_ei_vec[3])
    def log_CO_ei_piecewise(log_fuel_flow):
        # Line between Idle and App
        m1 = (log_CO_ei_app - log_CO_ei_idle) / (log_fuel_flow_app - log_fuel_flow_idle)
        b1 = log_CO_ei_idle - m1 * log_fuel_flow_idle
        # Line between App and avg(T/O, C/O)
        m2 = (log_CO_ei_avg_to_co - log_CO_ei_app) / (log_fuel_flow_co - log_fuel_flow_app)
        b2 = log_CO_ei_app - m2 * log_fuel_flow_app
        # Find intersection of first line and constant
        intersect_x = (log_CO_ei_avg_to_co - b1) / m1 if m1 != 0 else float('inf')
        # If intersection is within [App, C/O], use piecewise; else, use two segments
        if log_fuel_flow_app <= intersect_x <= log_fuel_flow_co:
            if log_fuel_flow <= intersect_x:
                return m1 * log_fuel_flow + b1
            else:
                return log_CO_ei_avg_to_co
        else:
            if log_fuel_flow <= log_fuel_flow_app:
                return m1 * log_fuel_flow + b1
            elif log_fuel_flow_app < log_fuel_flow <= log_fuel_flow_co:
                return m2 * log_fuel_flow + b2
            else:
                return log_CO_ei_avg_to_co
    
    # For NOx: Fit a piecewise continuous line through all four points

    # Create a piecewise linear point-to-point interpolator in log-log space
    def log_NOx_ei_piecewise(log_fuel_flow):
        return np.interp(log_fuel_flow, log_fuel_flow_vec, log_NOx_ei_vec)
    
    import matplotlib.pyplot as plt

    # Prepare log-log data for plotting
    x_vals = np.linspace(log_fuel_flow_vec[3], log_fuel_flow_vec[1], 200)
    hc_fit = [log_HC_ei_piecewise(x) for x in x_vals]
    co_fit = [log_CO_ei_piecewise(x) for x in x_vals]

    plt.figure(figsize=(8, 6))
    # Plot actual points
    plt.scatter(log_fuel_flow_vec, log_HC_ei_vec, color='blue', label='HC EI data')
    plt.scatter(log_fuel_flow_vec, np.log10(CO_ei_vec), color='red', label='CO EI data')
    # Plot fits
    plt.plot(x_vals, hc_fit, color='blue', linestyle='--', label='HC EI fit')
    plt.plot(x_vals, co_fit, color='red', linestyle='--', label='CO EI fit')

    plt.xlabel('log10(Fuel Flow) [log10(lbs/hr)]')
    plt.ylabel('log10(Emission Index) [log10(g/kg)]')
    plt.title('Log-Log Fits for HC and CO Emission Indices')
    plt.legend()
    plt.grid(True, which='both', ls=':')
    plt.show()
    # Step 1. 

