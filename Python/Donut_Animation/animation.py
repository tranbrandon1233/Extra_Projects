import time
import math

# Define donut parameters
R1 = 1  # Radius of the donut
R2 = 2  # Radius of the torus
K2 = 5  # Distance from the center to the donut hole

# Define screen dimensions
WIDTH = 80
HEIGHT = 40

# Define rotation variables
A = 1
B = 1

# Define characters for shading
CHARS = ".,-~:;=!*#$@"

def render_frame(A, B):
    """Renders a single frame of the spinning donut."""

    # Precompute sines and cosines
    cosA = math.cos(A)
    sinA = math.sin(A)
    cosB = math.cos(B)
    sinB = math.sin(B)

    # Initialize output buffer
    output = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
    zbuffer = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]

    # Iterate over torus surface
    for theta in range(0, 628, 12):
        theta /= 100
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        for phi in range(0, 628, 6):
            phi /= 100
            cosphi = math.cos(phi)
            sinphi = math.sin(phi)

            # Calculate 3D coordinates
            circlex = R2 + R1 * costheta
            circley = R1 * sintheta

            x = circlex * (cosB * cosphi + sinA * sinB * sinphi) - circley * cosA * sinB
            y = circlex * (sinB * cosphi - sinA * cosB * sinphi) + circley * cosA * cosB
            z = K2 + cosA * circlex * sinphi + circley * sinA
            ooz = 1 / z  # One over z for depth

            # Project to 2D
            xp = int(WIDTH / 2 + K2 * ooz * x)
            yp = int(HEIGHT / 2 + K2 * ooz * y)

            # Calculate luminance
            L = cosphi * costheta * sinB - cosA * costheta * sinphi - \
                sinA * sintheta + cosB * (cosA * sintheta - costheta * sinA * sinphi)
            if L > 0:
                # Calculate luminance index and clip
                L_index = int(L * 8)
                L_index = min(L_index, len(CHARS) - 1)

                # Update output buffer if closer
                if ooz > zbuffer[yp][xp]:
                    zbuffer[yp][xp] = ooz
                    output[yp][xp] = CHARS[L_index]

    # Print the frame
    print('\x1b[H')  # Move cursor to top-left
    for row in output:
        print(''.join(row))

# Main loop
while True:
    render_frame(A, B)
    A += 0.07
    B += 0.03
    time.sleep(0.015)