import logging
from datetime import datetime, timedelta

# Configure logging for energy usage
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("usage_log.log"),  # Log energy usage to a file
        logging.StreamHandler()  # Display logs in the console
    ]
)

class Appliance:
    def __init__(self, name, category, daily_energy_kwh, operational_hours):
        self.name = name
        self.category = category
        self.daily_energy_kwh = daily_energy_kwh
        self.operational_hours = operational_hours
        self.maintenance_log = []  # To store maintenance dates

    def log_maintenance(self, date):
        """Logs maintenance date for an appliance."""
        self.maintenance_log.append(date)
        logging.info(f"Logged maintenance for {self.name} on {date}.")

    def get_maintenance_reminder(self):
        """Provides a reminder if maintenance is due based on the last log."""
        if self.maintenance_log:
            last_maintenance = self.maintenance_log[-1]
            next_due = last_maintenance + timedelta(days=180)  # Reminder every 6 months
            if datetime.now() > next_due:
                logging.warning(f"Maintenance due for {self.name}. Last maintenance: {last_maintenance}.")
                return f"Maintenance due for {self.name}."
        return f"No maintenance due for {self.name}."


class EnergyTracker:
    def __init__(self):
        self.appliances = []

    def log_appliance(self, name, category, daily_energy_kwh, operational_hours):
        if daily_energy_kwh < 0 or operational_hours <= 0:
            logging.error("Failed to log appliance: Invalid energy consumption or operational hours.")
            raise ValueError("Energy consumption and operational hours must be positive.")
        appliance = Appliance(name, category, daily_energy_kwh, operational_hours)
        self.appliances.append(appliance)
        logging.info(f"Logged new appliance: {name} ({category}).")

    def add_usage_log(self, name, energy):
        """Logs energy usage for an appliance."""
        if energy < 0:
            logging.error(f"Invalid usage log for appliance '{name}': Energy usage cannot be negative.")
            return
        logging.info(f"Usage log - Appliance: {name}, Energy: {energy} kWh, Timestamp: {datetime.now()}")

    def _parse_logs(self):
        """Parses the usage log file and returns structured log data."""
        logs = []
        with open("usage_log.log", "r") as file:
            for line in file:
                if "Usage log - Appliance" in line:
                    parts = line.strip().split(", ")
                    name = parts[0].split(": ")[1]
                    energy = float(parts[1].split(": ")[1].split(" ")[0])
                    timestamp = datetime.strptime(parts[2].split(": ")[1], "%Y-%m-%d %H:%M:%S.%f")
                    logs.append({"name": name, "energy": energy, "date": timestamp})
        return logs

    def generate_daily_usage_summary(self):
        """Generate a daily energy usage summary."""
        logs = self._parse_logs()
        daily_summary = {}
        for log in logs:
            day = log['date'].strftime('%Y-%m-%d')
            daily_summary[day] = daily_summary.get(day, 0) + log['energy']
        logging.info("Generated daily usage summary.")
        return daily_summary

    def generate_weekly_report(self):
        """Generate a weekly energy usage report."""
        logs = self._parse_logs()
        weekly_data = {}
        one_week_ago = datetime.now() - timedelta(days=7)
        for log in logs:
            if log['date'] >= one_week_ago:
                appliance_name = log['name']
                weekly_data[appliance_name] = weekly_data.get(appliance_name, 0) + log['energy']
        logging.info("Generated weekly energy usage report.")
        return weekly_data

    def generate_monthly_summary(self):
        """Generate a monthly energy usage summary."""
        logs = self._parse_logs()
        monthly_summary = {}
        one_month_ago = datetime.now() - timedelta(days=30)
        for log in logs:
            if log['date'] >= one_month_ago:
                appliance_name = log['name']
                monthly_summary[appliance_name] = monthly_summary.get(appliance_name, 0) + log['energy']
        logging.info("Generated monthly energy usage summary.")
        return monthly_summary


# Example Usage
tracker = EnergyTracker()

# Log appliances and usage
tracker.log_appliance("Refrigerator", "Kitchen", 1.5, 24)
tracker.log_appliance("Washing Machine", "Laundry", 2.5, 2)

tracker.add_usage_log("Refrigerator", 10)
tracker.add_usage_log("Washing Machine", 5)

# Generate summaries
print("Daily Usage Summary:", tracker.generate_daily_usage_summary())
print("Weekly Report:", tracker.generate_weekly_report())
print("Monthly Summary:", tracker.generate_monthly_summary())