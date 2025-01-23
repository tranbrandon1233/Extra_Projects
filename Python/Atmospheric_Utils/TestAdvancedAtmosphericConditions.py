import unittest
from atmosphere_module import AdvancedAtmosphericConditions  # Adjust import to match your module's name

class TestAdvancedAtmosphericConditions(unittest.TestCase):

    # Input Validation
    def test_validate_input_valid(self):
        """Test valid input values for altitude, ground temperature, and ground altitude."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=25, ground_altitude_m=0)
        atmospheric_conditions.validate_input()  # Should not raise an exception

    def test_validate_input_invalid_type(self):
        """Test invalid input types, such as strings."""
        with self.assertRaises(TypeError):
            atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m="2000", ground_temp_C=25, ground_altitude_m=0)
            atmospheric_conditions.validate_input()

    def test_validate_input_invalid_altitude_range(self):
        """Test invalid altitude range beyond the limits."""
        with self.assertRaises(ValueError):
            atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=90000, ground_temp_C=25, ground_altitude_m=0)
            atmospheric_conditions.validate_input()

    def test_validate_input_invalid_temperature(self):
        """Test ground temperature below absolute zero."""
        with self.assertRaises(ValueError):
            atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=-300, ground_altitude_m=0)
            atmospheric_conditions.validate_input()

    # Temperature Offset Calculation
    def test_calculate_temperature_offset(self):
        """Verify correct calculation of temperature offset."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=25, ground_altitude_m=0)
        offset = atmospheric_conditions.calculate_temperature_offset()
        self.assertIsInstance(offset, float)
        self.assertAlmostEqual(offset, 10.0, delta=0.1)  # Adjust expected value based on actual behavior

    # Advanced Calculations
    def test_calculate_temperature(self):
        """Verify temperature adjustment based on offset."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=25, ground_altitude_m=0)
        temperature_offset = atmospheric_conditions.calculate_temperature_offset()
        temperature = atmospheric_conditions.calculate_temperature(temperature_offset)
        self.assertIsInstance(temperature, float)
        self.assertAlmostEqual(temperature, 12.004, delta=0.1)  # Adjust expected value

    def test_calculate_pressure(self):
        """Ensure accurate pressure retrieval."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=25, ground_altitude_m=0)
        pressure = atmospheric_conditions.calculate_pressure()
        self.assertIsInstance(pressure, float)
        self.assertGreater(pressure, 0)

    def test_calculate_uv_index(self):
        """Test UV index calculation for valid altitudes."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=25, ground_altitude_m=0)
        uv_index = atmospheric_conditions.calculate_uv_index()
        self.assertIsInstance(uv_index, float)
        self.assertGreater(uv_index, 5.0)

    # Dew Point Calculation
    def test_calculate_dew_point(self):
        """Validate dew point calculation for varying humidity and temperature."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=25, ground_altitude_m=0)
        dew_point = atmospheric_conditions.calculate_dew_point(temperature_C=25, humidity_percent=50)
        self.assertIsInstance(dew_point, float)
        self.assertAlmostEqual(dew_point, 9.68, delta=0.5)  # Adjust expected value 

    # Output Consistency
    def test_generate_atmospheric_conditions(self):
        """Ensure the output structure and content of atmospheric conditions."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=2000, ground_temp_C=25, ground_altitude_m=0)
        conditions = atmospheric_conditions.generate_atmospheric_conditions(humidity_percent=50)
        self.assertIsInstance(conditions, dict)
        self.assertIn("temperature (C)", conditions)
        self.assertIn("pressure (Pa)", conditions)
        self.assertIn("UV Index", conditions)
        self.assertIn("dew point (C)", conditions)

    def test_generate_atmospheric_profile(self):
        """Validate atmospheric profile generation for a range of altitudes."""
        atmospheric_conditions = AdvancedAtmosphericConditions(altitude_m=1000, ground_temp_C=20, ground_altitude_m=0)
        profile = atmospheric_conditions.generate_atmospheric_profile(start_altitude=1000, end_altitude=3000, step=1000, humidity_percent=50)
        self.assertIsInstance(profile, list)
        self.assertEqual(len(profile), 3)
        self.assertIn("altitude (m)", profile[0])
        self.assertIn("temperature (C)", profile[0])

# Run the tests
if __name__ == "__main__":
    unittest.main()

