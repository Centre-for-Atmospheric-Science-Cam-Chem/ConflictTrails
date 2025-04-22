def ftmin_to_ms(ftmin):
    """
    Convert speed from feet per minute to meters per second.
    Args:
        ftmin (float): Speed in feet per minute.
    Returns:
        float: Speed in meters per second.
    """
    return ftmin * 0.00508
    # 1 foot = 0.3048 meters
    # 1 minute = 60 seconds
    # Therefore, 1 ft/min = 0.3048 m / 60 s = 0.00508 m/s
    # 1 ft/min = 0.00508 m/s