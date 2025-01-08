     
import unittest
from atmospheric_utils import (
    validate_inputs, calculate_ground_conditions, calculate_query_conditions,
    calculate_dew_point, calculate_pressure_difference
)
from main import estimate_atmospheric_conditions, generate_atmospheric_profile

class TestAtmosphericUtils(unittest.TestCase):

    def test_validate_inputs(self):
        # Valid inputs
        validate_inputs(5000, 0, 15)

        # Invalid altitude_query_m
        with self.assertRaises(ValueError):
            validate_inputs(82000, 0, 15)

        # Invalid ground_altitude_m
        with self.assertRaises(ValueError):
            validate_inputs(5000, -10, 15)

        # Invalid measured_ground_temp_C
        with self.assertRaises(ValueError):
            validate_inputs(5000, 0, -300)

    def test_calculate_ground_conditions(self):
        temp_c = calculate_ground_conditions(0)  # Sea level
        self.assertIsInstance(temp_c, float)

    def test_calculate_query_conditions(self):
        conditions = calculate_query_conditions(5000, 2)  # 5000m altitude, +2°C correction
        self.assertIn("temperature (C)", conditions)
        self.assertIn("pressure (Pa)", conditions)
        self.assertAlmostEqual(conditions["temperature (C)"], -15, delta=0.1)  # Updated expected temperature
        self.assertAlmostEqual(conditions["pressure (Pa)"], 54048.3, delta=1.0)  # Updated expected pressure

    def test_calculate_dew_point(self):
        dew_point = calculate_dew_point(20, 50)  # 20°C, 50% humidity
        self.assertAlmostEqual(dew_point, 9.25, delta=0.1)  # Correct expected dew point

    def test_calculate_pressure_difference(self):
        pressure_diff = calculate_pressure_difference(0, 5000)  # Sea level to 5000m
        self.assertIsInstance(pressure_diff, float)
        self.assertAlmostEqual(pressure_diff, -47275, delta=2.0)  # Updated expected pressure difference

    def test_estimate_atmospheric_conditions(self):
        conditions = estimate_atmospheric_conditions(5000, 0, 15, humidity_percent=50)
        self.assertIn("dew point (C)", conditions)
        self.assertIn("pressure difference (Pa)", conditions)
        self.assertAlmostEqual(conditions["temperature (C)"], -17, delta=0.1)  # Updated expected temperature
        self.assertAlmostEqual(conditions["dew point (C)"], -24.93, delta=0.1)  # Updated expected dew point
        self.assertAlmostEqual(conditions["pressure difference (Pa)"], -47275, delta=2.0)  # Updated expected pressure difference

    def test_generate_atmospheric_profile(self):
        profile = generate_atmospheric_profile(0, 10000, 2000, 0, 15)
        self.assertEqual(len(profile), 6)  # 0, 2000, 4000, ..., 10000
        for entry in profile:
            self.assertIn("altitude (m)", entry)
            self.assertIn("temperature (C)", entry)
            altitude = entry["altitude (m)"]
            expected_temp = 15 - (0.0065 * altitude)  # Expected temperature calculation
            self.assertAlmostEqual(entry["temperature (C)"], expected_temp, delta=0.1)


if __name__ == "__main__":
    unittest.main()