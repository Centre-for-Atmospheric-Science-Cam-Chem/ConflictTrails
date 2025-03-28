from download_day import download_day
# User Inputs:
start_time_str       = '2022-02-01T00:00:00Z'
stop_time_str        = '2022-03-01T01:00:00Z'
# plane_callsign       = "EZY158T"
query_limit          = 1e3
send_notification    = True
make_plot            = True


# Downlad data from OpenSky history database
download_day(start_time_str, stop_time_str, query_limit, send_notification, make_plot)