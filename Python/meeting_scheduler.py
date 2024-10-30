from datetime import datetime, timedelta
import random
import itertools

class MeetingScheduler:
    def __init__(self, num_employees=20, num_rooms=2, start_hour=9, end_hour=17, meeting_duration=60):
        self.num_employees = num_employees  # Number of employees
        self.num_rooms = num_rooms  # Number of available meeting rooms
        self.start_hour = start_hour  # Start hour for meetings
        self.end_hour = end_hour  # End hour for meetings
        self.meeting_duration = meeting_duration  # Duration of each meeting in minutes
        self.employees = list(range(1, num_employees + 1))  # List of employee IDs

    def generate_all_pairs(self):
        """Generate all possible pairs of employees"""
        return list(itertools.combinations(self.employees, 2))

    def get_available_slots(self, start_date):
        """Generate all possible meeting slots for a month"""
        slots = []
        current_date = start_date

        # Continue until we're in a different month
        while current_date.month == start_date.month:
            if current_date.weekday() < 5:  # Monday to Friday
                current_time = current_date.replace(hour=self.start_hour, minute=0)
                end_time = current_date.replace(hour=self.end_hour, minute=0)

                # Generate time slots within the working hours
                while current_time < end_time:
                    slots.append(current_time)
                    current_time += timedelta(minutes=self.meeting_duration)
            
            # Move to the next day
            current_date += timedelta(days=1) 
        
        return slots

    def schedule_meetings(self, start_date):
        """Schedule all meetings for the month"""
        # Get all pairs that need to meet
        pairs = self.generate_all_pairs()
        
        # Get all available time slots
        available_slots = self.get_available_slots(start_date)

        # Calculate if we have enough slots
        total_meetings = len(pairs)
        total_available_slots = len(available_slots) * self.num_rooms
        
        # Check if there are enough slots for all meetings
        if total_available_slots < total_meetings:
            raise ValueError(f"Not enough slots available. Need {total_meetings} slots but only have {total_available_slots}")

        # Shuffle pairs and slots for random distribution
        random.shuffle(pairs)

        schedule = []
        current_slot_idx = 0
        rooms_used = 0
        
        # Loop for each pair of employees
        for pair in pairs:
            # Loop through available slots to find a free slot
            while current_slot_idx < len(available_slots):
                # If we have available rooms, schedule the meeting
                if rooms_used < self.num_rooms:
                    # Add the meeting to the schedule
                    schedule.append({
                        'employees': pair,
                        'time': available_slots[current_slot_idx],
                        'room': rooms_used + 1
                    })
                    # Move to the next pair
                    rooms_used += 1
                    break
                # Move to the next slot
                else:
                    # Move to the next slot and reset the room count
                    current_slot_idx += 1
                    rooms_used = 0
        
        return schedule

    def print_schedule(self, schedule):
        """Print the schedule in a readable format"""
        schedule.sort(key=lambda x: (x['time'], x['room']))

        current_date = None
        # Loop through each meeting and print the details
        for meeting in schedule:
            # Print the date if it's a new day
            meeting_date = meeting['time'].date()
            # Print the date if it's a new day
            if current_date != meeting_date:
                current_date = meeting_date
                print(f"\n=== {current_date.strftime('%A, %B %d, %Y')} ===")
            
            # Print the meeting details
            print(f"Time: {meeting['time'].strftime('%H:%M')} | "
                  f"Room {meeting['room']} | "
                  f"Employees {meeting['employees'][0]} and {meeting['employees'][1]}")

# Example usage
if __name__ == "__main__":
    scheduler = MeetingScheduler()
    start_date = datetime(2024, 11, 1)  # Schedule for November 2024

    try:
        # Generate and print the schedule
        schedule = scheduler.schedule_meetings(start_date)
        scheduler.print_schedule(schedule)
    except ValueError as e:
        print(f"Error: {e}")