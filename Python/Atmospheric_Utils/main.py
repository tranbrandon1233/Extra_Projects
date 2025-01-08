import math
from ambiance import Atmosphere

# Constants
MAX_ALTITUDE = 81020
MIN_ALTITUDE = 0
ZERO_C_IN_K = -273.15  # Absolute zero in Celsius

# Input Validation
def validate_inputs(altitude_query_m, ground_altitude_m, measured_ground_temp_C):
    """
    Validates input parameters for altitude and temperature.
    """
    if not all(isinstance(arg, (int, float)) for arg in [altitude_query_m, ground_altitude_m, measured_ground_temp_C]):
        raise TypeError("All inputs must be numbers (int or float)")
    
    if not (MIN_ALTITUDE <= altitude_query_m <= MAX_ALTITUDE):
        raise ValueError(f"Altitude query must be {MIN_ALTITUDE} <= altitude <= {MAX_ALTITUDE}.")
    
    if not (MIN_ALTITUDE <= ground_altitude_m <= MAX_ALTITUDE):
        raise ValueError(f"Ground altitude must be {MIN_ALTITUDE} <= altitude <= {MAX_ALTITUDE}.")
    
    if measured_ground_temp_C < ZERO_C_IN_K:
        raise ValueError(f"Temperature must be >= {ZERO_C_IN_K}°C.")

# Calculate Ground-Level Atmospheric Conditions
def calculate_ground_conditions(ground_altitude_m):
    """
    Calculates atmospheric conditions at ground level.
    """
    ground_atmosphere = Atmosphere(ground_altitude_m)
    return ground_atmosphere.temperature_in_celsius[0]

# Calculate Atmospheric Conditions at Queried Altitude
def calculate_query_conditions(altitude_query_m, temp_offset_C):
    """
    Calculates atmospheric conditions at a given altitude, adjusted for temperature offset.
    """
    altitude_atmosphere = Atmosphere(altitude_query_m)
    adjusted_temperature = altitude_atmosphere.temperature_in_celsius[0] + temp_offset_C
    return {
        "temperature (C)": round(adjusted_temperature),
        "temperature (F)": round(adjusted_temperature * 9 / 5 + 32),
        "pressure (Pa)": round(altitude_atmosphere.pressure[0]),
        "pressure (mbar)": round(altitude_atmosphere.pressure[0] / 100),
        "density (kg/m³)": round(altitude_atmosphere.density[0], 3),
        "speed of sound (m/s)": round(altitude_atmosphere.speed_of_sound[0], 2),
    }

# Dew Point Calculation
def calculate_dew_point(temperature_c, humidity_percent):
    """
    Calculates the dew point using the corrected Magnus formula.
    """
    a, b = 17.625, 243.04
    alpha = math.log(humidity_percent / 100) + (a * temperature_c) / (b + temperature_c)
    dew_point = (b * alpha) / (a - alpha)
    return round(dew_point, 2)

# Pressure Difference Calculation
def calculate_pressure_difference(ground_altitude_m, altitude_query_m, temp_offset_C=0):
    """
    Calculates pressure difference between ground and a specified altitude, 
    considering temperature adjustments.
    """
    ground_atmosphere = Atmosphere(ground_altitude_m)
    altitude_atmosphere = Atmosphere(altitude_query_m)

    ground_pressure_corrected = ground_atmosphere.pressure[0]
    altitude_pressure_corrected = altitude_atmosphere.pressure[0] + temp_offset_C * 10  # Example correction factor

    return round(altitude_pressure_corrected - ground_pressure_corrected, 2)

# New Feature: Relative Humidity Calculation at Altitude
def calculate_relative_humidity(temperature_c):
    """
    Calculates the relative humidity based on temperature and dew point. (Assuming y-intercept is 0)
    """
    L = 2.501 * 10 ** 6
    Rv = 461
    T0 = 273.15
    e0 = 0.611
    Hvap = 40.67
    R = 8.314
    saturation_vapor_pressure = e0 *  math.exp((L/Rv) * ((1/T0) - (1/(temperature_c))))
    actual_vapor_pressure = math.exp(-Hvap/(R*temperature_c))
    relative_humidity = (actual_vapor_pressure / saturation_vapor_pressure) * 100
    return round(relative_humidity, 2)

# Example Usage:
if __name__ == "__main__":
    # Example Inputs
    altitude_query_m = 5000
    ground_altitude_m = 0
    measured_ground_temp_C = 15
    dew_point_c = calculate_dew_point(measured_ground_temp_C, 60)  # Assume 60% ground humidity

    # Validate Inputs
    validate_inputs(altitude_query_m, ground_altitude_m, measured_ground_temp_C)

    # Calculate Conditions
    query_conditions = calculate_query_conditions(altitude_query_m, 0)
    relative_humidity = calculate_relative_humidity(query_conditions["temperature (C)"], dew_point_c)

    print(f"Query Conditions: {query_conditions}")
    print(f"Relative Humidity at Altitude: {relative_humidity}%")
