from pyopensky.trino import Trino as BaseTrino

class CustomTrino(BaseTrino):
    def history(
        self,
        start,
        stop=None,
        *args,
        callsign=None,
        icao24=None,
        serials=None,
        bounds=None,
        departure_airport=None,
        arrival_airport=None,
        airport=None,
        time_buffer=None,
        cached=True,
        compress=False,
        limit=None,
        selected_columns=(),
        **kwargs,
    ):
        # Make your changes here
        print("Custom history method called")
        # Optionally call the base implementation if needed:
        return super().history(
            start,
            stop=stop,
            *args,
            callsign=callsign,
            icao24=icao24,
            serials=serials,
            bounds=bounds,
            departure_airport=departure_airport,
            arrival_airport=arrival_airport,
            airport=airport,
            time_buffer=time_buffer,
            cached=cached,
            compress=compress,
            limit=limit,
            selected_columns=selected_columns,
            **kwargs,
        )