import pygame
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FILE_SIZE = 30
NUM_FILES = 100

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption("Windows Desktop Simulation")

clock = pygame.time.Clock()

def create_random_file_position():
    return (random.randint(0, SCREEN_WIDTH - FILE_SIZE), random.randint(0, SCREEN_HEIGHT - FILE_SIZE))

def point_inside_rect(x, y, rect):
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]

file_positions = [create_random_file_position() for _ in range(NUM_FILES)]

running = True
dragging = False
drag_start_pos = None
selection_box = pygame.Rect(0, 0, 0, 0)
selected_files = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
                drag_start_pos = pygame.mouse.get_pos()
                selection_box.topleft = drag_start_pos
                selection_box.size = (0, 0)
                selected_files.clear()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: 
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                selection_box.width = mouse_x - drag_start_pos[0]
                selection_box.height = mouse_y - drag_start_pos[1]

                # Correct the selection box position and size
                if selection_box.width < 0:
                    selection_box.x = mouse_x
                    selection_box.width = abs(selection_box.width)
                if selection_box.height < 0:
                    selection_box.y = mouse_y
                    selection_box.height = abs(selection_box.height)

                selected_files = [file for file in file_positions if point_inside_rect(file[0], file[1], selection_box)]

    screen.fill(WHITE)
    
    for file_pos in file_positions:
        pygame.draw.rect(screen, BLUE, (file_pos[0], file_pos[1], FILE_SIZE, FILE_SIZE))
    
    if dragging:
        pygame.draw.rect(screen, GREEN, selection_box, 1)
    
    for file_pos in selected_files:
        pygame.draw.rect(screen, RED, (file_pos[0], file_pos[1], FILE_SIZE, FILE_SIZE), 2)

    pygame.display.flip()

    clock.tick(144)

# Quit Pygame
pygame.quit()