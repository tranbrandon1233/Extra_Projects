import pygame
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

# Initialize pygame and settings
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Simulation")

# OpenGL settings
glClearColor(0.5, 0.8, 1.0, 1.0)  # Sky color
glEnable(GL_DEPTH_TEST)

# Camera setup
camera_offset = np.array([0.0, 2.5, -6.0])

# Player and object settings
player_pos = np.array([0.0, 1.0, 0.0])
player_radius = 0.5
player_speed = 0.1
cubes = [{"pos": np.array([random.uniform(-5, 5), 0.5, random.uniform(-5, 5)]), "dir": np.random.randn(3) * 0.05} for _ in range(5)]

# Game variables
score = 0
arrows = []

# Load the font for score display
font = pygame.font.Font(None, 36)

def draw_sphere(position, radius, color):
    glPushMatrix()
    glColor3f(*color)
    glTranslatef(*position)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 32, 32)
    gluDeleteQuadric(quad)
    glPopMatrix()

def draw_cube(position, size, color):
    glPushMatrix()
    glColor3f(*color)
    glTranslatef(*position)
    glutWireCube(size)
    glPopMatrix()

def draw_ground():
    glBegin(GL_QUADS)
    glColor3f(0.1, 0.6, 0.1)  # Green ground
    glVertex3f(-20, 0, -20)
    glVertex3f(20, 0, -20)
    glVertex3f(20, 0, 20)
    glVertex3f(-20, 0, 20)
    glEnd()

def handle_movement(keys):
    global player_pos
    if keys[pygame.K_LEFT]: player_pos[0] -= player_speed
    if keys[pygame.K_RIGHT]: player_pos[0] += player_speed
    if keys[pygame.K_UP]: player_pos[2] += player_speed
    if keys[pygame.K_DOWN]: player_pos[2] -= player_speed

def fire_arrow():
    arrows.append({"pos": np.copy(player_pos), "dir": np.array([0, 0, -0.2])})

def update_arrows():
    global score
    for arrow in arrows[:]:
        arrow["pos"] += arrow["dir"]
        for cube in cubes[:]:
            if np.linalg.norm(arrow["pos"] - cube["pos"]) < 0.5:
                cubes.remove(cube)
                arrows.remove(arrow)
                score += 1
                break

def update_cubes():
    for cube in cubes:
        cube["pos"] += cube["dir"]
        if abs(cube["pos"][0]) > 5 or abs(cube["pos"][2]) > 5:
            cube["dir"] = -cube["dir"]

def display_score():
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    score_surface = pygame.image.tostring(score_text, "RGBA", True)
    glWindowPos2f(width - 100, height - 50)
    glDrawPixels(score_text.get_width(), score_text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, score_surface)

# Game loop
running = True
clock = pygame.time.Clock()
while running:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Update camera position to follow player
    camera_pos = player_pos + camera_offset
    gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2],
              player_pos[0], player_pos[1], player_pos[2], 0, 1, 0)

    # Draw ground, player, cubes, and arrows
    draw_ground()
    draw_sphere(player_pos, player_radius, (0.0, 0.0, 1.0))  # Blue player sphere
    for cube in cubes:
        draw_cube(cube["pos"], 1.0, (1.0, 0.0, 0.0))  # Red cubes
    for arrow in arrows:
        draw_sphere(arrow["pos"], 0.1, (0.8, 0.8, 0.0))  # Yellow arrows

    # Input handling
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                fire_arrow()

    keys = pygame.key.get_pressed()
    handle_movement(keys)

    # Update game objects
    update_arrows()
    update_cubes()

    display_score()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
