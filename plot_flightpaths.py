import matplotlib.pyplot as plt

def plot_flightpaths(flightpath_list):
    """
    Plot the flight path of an aircraft.
    Parameters:
    flightpath (dict): A dictionary containing the flight path data.
    The dictionary should contain keys representing different segments of the flight,
    and each segment should contain keys for distance and altitude.
    """
    
    plt.figure(figsize=(10, 6))
    
    for flightpath in flightpath_list:
        distances_list = [0]
        altitudes_list = [0]

        # Climb phase:
        cum_distance = distances_list[-1]
        for key in flightpath:
            for key2 in flightpath[key]:
                if key2.startswith('s_'):
                    cum_distance += flightpath[key][key2]
                    distances_list.append(cum_distance)
                    altitudes_list.append(flightpath[key][f'{key2.replace('s_', 'h_')}_end'])
        plt.plot(distances_list, altitudes_list, marker = 'o', linestyle='-', label=f"{cum_distance:.0f} km")

    # Plot the altitude profile vs. cumulative flight distance.
    plt.xlabel('Cumulative Flight Distance (meters)')
    plt.ylabel("Altitude (meters)")
    plt.title("Flightpaths for varying flight distances")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    