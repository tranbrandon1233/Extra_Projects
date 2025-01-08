from ambiance import Atmosphere

MAX_ALTITUDE = 81020
MIN_ALTITUDE = 0
ZERO_C_IN_K = -273.15

class AdvancedAtmosphericConditions:
    def __init__(self, altitude_m, ground_temp_C, ground_altitude_m):
        self.altitude_m = altitude_m
        self.ground_temp_C = ground_temp_C
        self.ground_altitude_m = ground_altitude_m

    def validate_input(self):
        if not isinstance(self.altitude_m, (int, float)) or not isinstance(self.ground_temp_C, (int, float)) or not isinstance(self.ground_altitude_m, (int, float)):
            raise TypeError("Inputs must be numeric (int or float).")
        
        if self.altitude_m < MIN_ALTITUDE or self.altitude_m > MAX_ALTITUDE:
            raise ValueError(f"Altitude must be between {MIN_ALTITUDE} and {MAX_ALTITUDE}.")
        
        if self.ground_altitude_m < MIN_ALTITUDE or self.ground_altitude_m > MAX_ALTITUDE:
            raise ValueError(f"Ground altitude must be between {MIN_ALTITUDE} and {MAX_ALTITUDE}.")
        
        if self.ground_temp_C < ZERO_C_IN_K:
            raise ValueError(f"Ground temperature must be above {ZERO_C_IN_K}°C.")

    def get_atmospheric_data(self):
        atmosphere = Atmosphere(self.altitude_m)
        return atmosphere

    def calculate_temperature_offset(self):
        ground_atmosphere = Atmosphere(self.ground_altitude_m)
        ground_temperature = ground_atmosphere.temperature_in_celsius[0]
        temperature_offset = self.ground_temp_C - ground_temperature
        return temperature_offset

    def calculate_temperature(self, temperature_offset):
        atmosphere = self.get_atmospheric_data()
        adjusted_temperature = atmosphere.temperature_in_celsius[0] + temperature_offset
        return adjusted_temperature

    def calculate_pressure(self):
        atmosphere = self.get_atmospheric_data()
        return atmosphere.pressure[0]

    def calculate_uv_index(self):
        uv_index = 5 + (self.altitude_m * 0.02)  # Simplified UV index formula based on altitude
        return round(uv_index, 2)

    # Dew Point Calculation
    def calculate_dew_point(temperature_c, humidity_percent):
        """
        Calculates the dew point using the corrected Magnus formula.
        """
        a, b = 17.625, 243.04
        alpha = math.log(humidity_percent / 100) + (a * temperature_c) / (b + temperature_c)
        dew_point = (b * alpha) / (a - alpha)
        return round(dew_point, 2)

    def generate_atmospheric_conditions(self, humidity_percent=50):
        
        self.validate_input()
        
        temperature_offset = self.calculate_temperature_offset()
        temperature_C = self.calculate_temperature(temperature_offset)
        pressure_Pa = self.calculate_pressure()
        uv_index = self.calculate_uv_index()
        dew_point = self.estimate_dew_point(temperature_C, humidity_percent)

        return {
            "temperature (C)": round(temperature_C, 2),
            "temperature (F)": round(temperature_C * 9/5 + 32, 2),
            "pressure (Pa)": round(pressure_Pa, 2),
            "pressure (mbar)": round(pressure_Pa / 100, 2),
            "UV Index": uv_index,
            "dew point (C)": dew_point
        }

    def generate_atmospheric_profile(self, start_altitude, end_altitude, step, humidity_percent=50):
        
        profile = []
        for altitude in range(start_altitude, end_altitude + 1, step):
            self.altitude_m = altitude  # Update altitude for each step
            conditions = self.generate_atmospheric_conditions(humidity_percent)
            profile.append({"altitude (m)": altitude, **conditions})
        return profile