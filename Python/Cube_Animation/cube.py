import time
import math

# Define the vertices of a cube
cube_vertices = [
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
]

# Define the edges connecting the vertices
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

def rotate_point(x, y, z, angle_x, angle_y, angle_z):
    """Rotates a point in 3D space."""
    # X-axis rotation
    rad = angle_x * math.pi / 180
    y, z = y * math.cos(rad) - z * math.sin(rad), y * math.sin(rad) + z * math.cos(rad)

    # Y-axis rotation
    rad = angle_y * math.pi / 180
    x, z = x * math.cos(rad) + z * math.sin(rad), -x * math.sin(rad) + z * math.cos(rad)

    # Z-axis rotation
    rad = angle_z * math.pi / 180
    x, y = x * math.cos(rad) - y * math.sin(rad), x * math.sin(rad) + y * math.cos(rad)

    return x, y, z

def project_point(x, y, z, scale=10):
    """Projects a 3D point onto a 2D plane."""
    return int(x * scale), int(y * scale)

def clear_console():
    """Clears the console."""
    print("\033[H\033[J", end="")

def print_frame(vertices):
    """Prints a frame of the rotating cube."""
    max_x = max(v[0] for v in vertices)
    min_x = min(v[0] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_y = min(v[1] for v in vertices)

    for y in range(int(min_y), int(max_y) + 1):
        for x in range(int(min_x), int(max_x) + 1):
            if any((round(v[0]) == x and round(v[1]) == y) for v in vertices):
                print("#", end="")
            else:
                print(" ", end="")
        print()

# Animation loop
angle_x = 0
angle_y = 0
angle_z = 0
while True:
    # Rotate the cube
    rotated_vertices = [
        rotate_point(x, y, z, angle_x, angle_y, angle_z) for x, y, z in cube_vertices
    ]

    # Project the rotated vertices onto 2D
    projected_vertices = [
        project_point(x, y, z) for x, y, z in rotated_vertices
    ]

    # Clear the console and print the frame
    clear_console()
    print_frame(projected_vertices)

    # Update rotation angles
    angle_x += 1
    angle_y += 2
    angle_z += 3

    # Pause for a short time to control animation speed
    time.sleep(0.05)