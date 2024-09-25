import datetime
import requests
import holidays

class Aircraft:
    def __init__(self, aircraft_id, maintenance_schedule):
        self.aircraft_id = aircraft_id
        self.maintenance_schedule = maintenance_schedule

    def is_under_maintenance(self, date):
        return date in self.maintenance_schedule


class Flight:
    def __init__(self, flight_id, aircraft, departure_time, arrival_time):
        self.flight_id = flight_id
        self.aircraft = aircraft
        self.departure_time = departure_time
        self.arrival_time = arrival_time

    def __str__(self):
        return f"Flight {self.flight_id}: {self.departure_time} - {self.arrival_time}"


class Airport:
    def __init__(self):
        self.flights = []
        self.aircrafts = []
        self.passenger_demand = {}
        self.weather_conditions = {}
        self.air_traffic_control_restrictions = {}
        self.holidays = holidays.US()

    def add_aircraft(self, aircraft):
        self.aircrafts.append(aircraft)

    def add_flight(self, flight):
        self.flights.append(flight)

    def schedule_flight(self, flight_id, aircraft_id, departure_time, arrival_time, departure_location):
        aircraft = next((a for a in self.aircrafts if a.aircraft_id == aircraft_id), None)
        if aircraft and not aircraft.is_under_maintenance(departure_time.date()):
            # Check if it's a holiday
            if departure_time.date() not in self.holidays:
                # Check passenger demand
                if self.passenger_demand.get(departure_time.date(), 0) > 500:
                    # Check air traffic control restrictions
                    if departure_time not in self.air_traffic_control_restrictions:
                        # Check weather conditions
                        weather_api_key = "638d533795b24fc787a221503232605"
                        weather_url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={departure_location}"
                        response = requests.get(weather_url)
                        weather_data = response.json()
                        if weather_data["current"]["condition"]["text"] == "Sunny":
                            flight = Flight(flight_id, aircraft, departure_time, arrival_time)
                            self.add_flight(flight)
                            print(f"Flight {flight_id} scheduled successfully for {departure_time}.")
                        else:
                            print(f"Unable to schedule flight {flight_id} for {departure_time}. Bad weather conditions.")
                            # Reschedule flight
                            rescheduled_departure_time = self.reschedule_flight(departure_time)
                            if rescheduled_departure_time:
                                self.schedule_flight(flight_id, aircraft_id, rescheduled_departure_time, arrival_time, departure_location)
                    else:
                        print(f"Unable to schedule flight {flight_id} for {departure_time}. Air traffic control restrictions.")
                        # Reschedule flight
                        rescheduled_departure_time = self.reschedule_flight(departure_time)
                        if rescheduled_departure_time:
                            self.schedule_flight(flight_id, aircraft_id, rescheduled_departure_time, arrival_time, departure_location)
                else:
                    print(f"Unable to schedule flight {flight_id} for {departure_time}. Low passenger demand.")
                    # Reschedule flight
                    rescheduled_departure_time = self.reschedule_flight(departure_time)
                    if rescheduled_departure_time:
                        self.schedule_flight(flight_id, aircraft_id, rescheduled_departure_time, arrival_time, departure_location)
            else:
                print(f"Unable to schedule flight {flight_id} for {departure_time}. It's a holiday.")
                # Reschedule flight
                rescheduled_departure_time = self.reschedule_flight(departure_time)
                if rescheduled_departure_time:
                    self.schedule_flight(flight_id, aircraft_id, rescheduled_departure_time, arrival_time, departure_location)
        else:
            print(f"Unable to schedule flight {flight_id} for {departure_time}. Aircraft {aircraft_id} is under maintenance.")
            # Reschedule flight
            rescheduled_departure_time = self.reschedule_flight(departure_time)
            if rescheduled_departure_time:
                self.schedule_flight(flight_id, aircraft_id, rescheduled_departure_time, arrival_time, departure_location)

    def reschedule_flight(self, departure_time):
        # Find a day with higher demand
        for date, demand in self.passenger_demand.items():
            if demand > 500 and date not in self.holidays:
                return datetime.datetime.combine(date, departure_time.time())
        # If no day with higher demand is found, try to reschedule for any other day
        for date in self.passenger_demand:
            if date not in self.holidays:
                return datetime.datetime.combine(date, departure_time.time())
        return None

    def analyze_past_flight_data(self):
        # Simulate past flight data analysis
        past_flight_data = [
            {"flight_id": "F001", "departure_time": datetime.datetime(2022, 1, 1, 10, 0), "arrival_time": datetime.datetime(2022, 1, 1, 11, 0), "delayed": True},
            {"flight_id": "F002", "departure_time": datetime.datetime(2022, 1, 2, 10, 0), "arrival_time": datetime.datetime(2022, 1, 2, 11, 0), "delayed": False},
            {"flight_id": "F003", "departure_time": datetime.datetime(2022, 1, 3, 10, 0), "arrival_time": datetime.datetime(2022, 1, 3, 11, 0), "delayed": True},
        ]

        delayed_flights = [f for f in past_flight_data if f["delayed"]]
        print(f"Delayed flights: {len(delayed_flights)}")

        # Warn about potential delays
        potential_delay_dates = [f["departure_time"].date() for f in delayed_flights]
        print(f"Potential delay dates: {potential_delay_dates}")


# Example usage
airport = Airport()

# Create aircrafts
aircraft1 = Aircraft("A001", [datetime.date(2024, 6, 15), datetime.date(2024, 6, 22)])
aircraft2 = Aircraft("A002", [datetime.date(2024, 6, 18), datetime.date(2024, 6, 25)])

# Add aircrafts to airport
airport.add_aircraft(aircraft1)
airport.add_aircraft(aircraft2)

# Update passenger demand
airport.passenger_demand[datetime.date(2024, 6, 16)] = 1000
airport.passenger_demand[datetime.date(2024, 6, 17)] = 800
airport.passenger_demand[datetime.date(2024, 6, 20)] = 1200

# Update air traffic control restrictions
airport.air_traffic_control_restrictions[datetime.datetime(2024, 6, 16, 10, 0)] = "Restriction 1"
airport.air_traffic_control_restrictions[datetime.datetime(2024, 6, 17, 11, 0)] = "Restriction 2"

# Schedule flights
airport.schedule_flight("F001", "A001", datetime.datetime(2024, 6, 16, 10, 0), datetime.datetime(2024, 6, 16, 11, 0), "New York")
airport.schedule_flight("F002", "A002", datetime.datetime(2024, 6, 17, 10, 0), datetime.datetime(2024, 6, 17, 11, 0), "Los Angeles")
airport.schedule_flight("F003", "A001", datetime.datetime(2024, 12, 25, 10, 0), datetime.datetime(2024, 12, 25, 11, 0), "New York")  # Christmas

# Analyze past flight data
airport.analyze_past_flight_data()