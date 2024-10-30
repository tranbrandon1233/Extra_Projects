def find_paths_with_travel_time(graph, start, end, intermediate=None, time_limit=120, max_stops=5, current_path=[], current_time=0, all_paths=[]):
    current_path = current_path + [start]
    if start == end:
        if intermediate and intermediate not in current_path:
            return
        if len(current_path) - 1 <= max_stops and current_time <= time_limit:
            all_paths.append((current_path, current_time))
        return
    if len(current_path) - 1 > max_stops or current_time > time_limit:
        return

    for neighbor, time in graph.get(start, []):
        if neighbor not in current_path:
            find_paths_with_travel_time(graph, neighbor, end, intermediate, time_limit, max_stops, current_path, current_time + time, all_paths)

def generate_detailed_report(graph, start, end, intermediate=None, time_limit=120, max_stops=5):
    all_paths = []
    find_paths_with_travel_time(graph, start, end, intermediate, time_limit, max_stops, [], 0, all_paths)

    if not all_paths:
        return "No valid path found."

    report = f"Total number of valid paths: {len(all_paths)}\n"
    shortest_path = min(all_paths, key=lambda x: x[1])

    for path, time in all_paths:
        report += f"Path: {' -> '.join(path)}, Total Time: {time} minutes\n"

    report += f"\nShortest Path: {' -> '.join(shortest_path[0])}, Total Time: {shortest_path[1]} minutes"
    return report

# Example usage
graph = {
    'A': [('B', 30), ('C', 45)],
    'B': [('C', 20), ('D', 60)],
    'C': [('D', 40)],
    'D': [('E', 25)],
    'E': []
}

print(generate_detailed_report(graph, 'A', 'E', intermediate='D'))