def pa_to_psi(pa):
    """
    Convert a value in pascals (Pa) to pounds per square inch (psi).
    
    Parameters:
    pa (float): Pressure in pascals.
    
    Returns:
    float: Pressure in psi.
    """
    return pa / 6894.76
    # 1 psi = 6894.76 Pa