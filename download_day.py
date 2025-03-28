# Import all necessary libraries
import datetime
import time
from pyopensky.trino import Trino
import pandas as pd
import os
from telegram_notifier import send_telegram_notification
import matplotlib
# forces matplotlib to use non-gui backends, freeing up resources and preventing errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def download_day(start_time_str: str,
                 stop_time_str: str,
                 query_limit: int,
                 telegram_updates: bool = False,
                 make_plot: bool = False):
    """ 
    Downloads data from the OpenSky database for a given day and saves it to a pickle file.
    
    Parameters:
    :param start_time_str (str): The start time in ISO format (YYYY-MM-DDTHH:MM:SSZ).
    :stop_time_str (str): The end time in ISO format (YYYY-MM-DDTHH:MM:SSZ).
    :query_limit (int): The maximum number of records to download. recommend less than 3e4
    :make_plot (bool): If True, generates a plot of the data.
    
    may be issues when server is busy, so generally recommended to run this code at night
    or keep queries low
    """
    # Loop through the dates between start and end date
    time_range = pd.date_range(start_time_str, stop_time_str, freq='D')
    for current_day in time_range:
        loop_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Convert the start and stop time to POSIX time
        start_time_posix    = int(current_day.timestamp())
        end_time_posix      = int(datetime.datetime.fromisoformat(stop_time_str).timestamp())

        # Define the SQL query to get the data from the OpenSky database
        sql_query  =    """SELECT icao24, estdepartureairport, estarrivalairport, callsign, track FROM flights_data4 
                            WHERE day = """ + str(start_time_posix) + """
                            LIMIT """ + str(round(query_limit)) 
                            
        # Pass the query to the database and get the result
        db = Trino()
        print(sql_query)
        result_df = db.query(sql_query, cached=False)

        # Save the result to a pickle file
        timestamp = current_day.strftime("%Y-%m-%d")
        output_dir = os.path.expanduser("~/Data")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"result_df_{timestamp}.pkl")
        result_df.to_pickle(filename)
        print("Saved result_df to", filename)
        
        # Plot the data on a map to be sent to the user via text message if true
        if make_plot:
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            ax.set_facecolor("black")
            ax.add_feature(cfeature.OCEAN, facecolor="black")
            ax.add_feature(cfeature.LAND, facecolor="dimgray")
            ax.add_feature(cfeature.COASTLINE, edgecolor="white", linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=0.5)
            ax.title.set_text(timestamp)

            for _, row in result_df.iterrows():
                track = row["track"]
                if not track:
                    continue
                lats = [pt.latitude for pt in track]
                lons = [pt.longitude for pt in track]
                ax.plot(lons, lats, color="red", linewidth=1, alpha=0.2, transform=ccrs.Geodetic())
            
            # save the plot to an image
            num_flights = len(result_df)
            image_file = os.path.join(output_dir, f"map_{timestamp}_{num_flights}.png")
            plt.savefig(image_file, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            if telegram_updates:
                loop_stop_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Send the plot to the user via text message
                send_telegram_notification("Download of " + str(round(query_limit)) + " flights from " + str(current_day) + ".\nStart time: " + loop_start_time + "\nEnd time: " + loop_stop_time,  image_path=image_file)
            
            
            
        # if user wants telegram updates but not a plot
        elif not make_plot and telegram_updates:
            loop_stop_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_telegram_notification("Download of " + str(round(query_limit)) + " flights from " + str(current_day) + ".\nStart time: " + loop_start_time + "\nEnd time: " + loop_stop_time)
        # if user wants a plot but not telegram updates
        elif make_plot and not telegram_updates:
            loop_stop_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        