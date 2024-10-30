def calculate_points(race_time, penalties):
    """Calculates the points based on race time and penalties."""
    hours, minutes, seconds, milliseconds = map(int, race_time.split(':'))
    total_seconds = (hours * 3600 + minutes * 60 + seconds) + milliseconds/1000
    penalty_deduction = penalties[0] * 3000 + penalties[1] * 15000 + penalties[2] * 25000
    total_seconds += penalty_deduction
    points = 600000 - total_seconds  
    return points

def sort_racers(racers_data):
    """Sorts racers based on their final point score."""
    results = []
    for racer in racers_data:
        name, time, p1, p2, p3 = racer
        penalties = (int(p1), int(p2), int(p3))
        points = calculate_points(time, penalties)
        results.append((name, time, points))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

# Example usage
racers_data = [
    ("racer1", "01:30:45:123", "1", "0", "0"),
    ("racer2", "01:28:30:500", "0", "1", "0"),
    ("racer3", "01:35:00:000", "0", "0", "0"),
    ("racer4", "01:25:15:800", "2", "0", "1"),
    ("racer5", "01:32:20:250", "0", "0", "0")
]

sorted_results = sort_racers(racers_data)

for name, time, points in sorted_results:
    display_points = max(0, points)  # Display 0 if points are negative
    print(f"{name}, {time}, {display_points}")