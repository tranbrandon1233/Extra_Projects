import unittest
from EnergyTracker import EnergyTracker


class TestEnergyTracker(unittest.TestCase):
    def setUp(self):
        """Set up an EnergyTracker instance and populate it with sample data."""
        self.tracker = EnergyTracker()

        # Log sample appliances with valid data
        self.tracker.log_appliance("Refrigerator", "Kitchen", 1.5, 24)
        self.tracker.log_appliance("Washing Machine", "Laundry", 2.5, 2)
        self.tracker.log_appliance("TV", "Entertainment", 0.8, 5)

        # Add usage logs
        self.tracker.add_usage_log("Refrigerator", 10)
        self.tracker.add_usage_log("Washing Machine", 5)
        self.tracker.add_usage_log("TV", 15)

    def test_log_appliance(self):
        """Test adding a new appliance."""
        self.tracker.log_appliance("Microwave", "Kitchen", 1.2, 2)
        self.assertEqual(len(self.tracker.appliances), 4)

    def test_add_usage_log(self):
        """Test logging energy usage for an appliance."""
        self.tracker.add_usage_log("Microwave", 12)
        logs = [log for log in self.tracker.usage_log if log['name'] == "Microwave"]
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['energy'], 12)

    def test_generate_weekly_report(self):
        """Test generating a weekly energy usage report."""
        report = self.tracker.generate_weekly_report()
        self.assertEqual(report["Refrigerator"], 10)
        self.assertEqual(report["Washing Machine"], 5)
        self.assertEqual(report["TV"], 15)

    def test_generate_monthly_summary(self):
        """Test generating a monthly energy usage summary."""
        summary = self.tracker.generate_monthly_summary()
        self.assertEqual(summary["Refrigerator"], 10)
        self.assertEqual(summary["Washing Machine"], 5)
        self.assertEqual(summary["TV"], 15)

    def test_calculate_energy_intensity(self):
        """Test calculating energy intensity for an appliance."""
        intensity = self.tracker.calculate_energy_intensity("Refrigerator")
        self.assertAlmostEqual(intensity, 1.5 / 24)

    def test_calculate_energy_intensity_division_by_zero(self):
        """Test handling division by zero in energy intensity calculation."""
        with self.assertRaises(ValueError):
            self.tracker.log_appliance("ZeroHoursAppliance", "Misc", 1.5, 0)

    def test_get_most_energy_intensive_appliance(self):
        """Test identifying the most energy-intensive appliance."""
        most_intensive = self.tracker.get_most_energy_intensive_appliance()
        self.assertEqual(most_intensive.name, "Washing Machine")

    def test_energy_consumption_trend(self):
        """Test generating energy consumption trends."""
        trend = self.tracker.energy_consumption_trend(days=7)
        self.assertEqual(len(trend), 7)  # Ensure trend covers 7 days
        total_energy = sum(trend.values())
        self.assertEqual(total_energy, 30)  # 10 + 5 + 15 from usage logs

    def test_generate_recommendations(self):
        """Test generating energy reduction recommendations."""
        # Adjust usage logs to exceed thresholds
        self.tracker.add_usage_log("Refrigerator", 60)  # Exceeds kitchen threshold
        self.tracker.add_usage_log("Washing Machine", 35)  # Exceeds laundry threshold
        self.tracker.add_usage_log("TV", 25)  # Exceeds entertainment threshold

        # Generate recommendations
        recommendations = self.tracker.generate_recommendations()

        # Check for specific recommendations
        self.assertIn("Consider reducing appliance usage in the kitchen.", recommendations)
        self.assertIn("Opt for energy-efficient laundry cycles.", recommendations)
        self.assertIn("Minimize usage of high-energy entertainment devices.", recommendations)

    def test_invalid_energy_usage(self):
        """Test handling invalid energy usage input."""
        initial_log_count = len(self.tracker.usage_log)
        self.tracker.add_usage_log("TV", -10)  # Invalid input
        self.assertEqual(len(self.tracker.usage_log), initial_log_count)  # Log count should not change

    def test_invalid_appliance_logging(self):
        """Test handling invalid appliance logging input."""
        with self.assertRaises(ValueError):
            self.tracker.log_appliance("Broken Appliance", "Misc", -1, 10)
        with self.assertRaises(ValueError):
            self.tracker.log_appliance("Broken Appliance", "Misc", 1.5, 0)

    def test_large_dataset_performance(self):
        """Test performance with a large dataset."""
        # Add a large number of appliances and usage logs
        for i in range(10000):
            appliance_name = f"Appliance_{i}"
            self.tracker.log_appliance(appliance_name, "General", 1.5, 10)
            self.tracker.add_usage_log(appliance_name, 5)

        # Verify that all appliances were added
        self.assertEqual(len(self.tracker.appliances), 10003)  # Including initial 3 appliances

        # Check the performance of weekly and monthly summaries
        weekly_report = self.tracker.generate_weekly_report()
        monthly_summary = self.tracker.generate_monthly_summary()

        self.assertEqual(len(weekly_report), 10003)
        self.assertEqual(len(monthly_summary), 10003)

if __name__ == "__main__":
    unittest.main()