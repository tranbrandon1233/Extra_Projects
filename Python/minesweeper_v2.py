import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 60
PADDING = 10  # Padding between cells
GRID_SIZE = 5
WINDOW_WIDTH = GRID_SIZE * (CELL_SIZE + PADDING) - PADDING  # Adjust for padding
WINDOW_HEIGHT = GRID_SIZE * (CELL_SIZE + PADDING) - PADDING
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Create the game window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Minefield Game")

# Initialize game state
grid = [0] * 25  # Changed to 25 to include the last cell
num_bombs = 5
bomb_indices = random.sample([i for i in range(25) if i != 12], num_bombs)  # Exclude the middle cell (index 12)
for index in bomb_indices:
    grid[index] = -1

score = 0
font = pygame.font.Font(None, 36)

def draw_grid():
    for i in range(25):  # Now iterate from 0 to 24
        if i == 12:  # Skip the middle cell
            continue
        row = i // GRID_SIZE  # Adjust row
        col = i % GRID_SIZE  # Adjust column
        x = col * (CELL_SIZE + PADDING)  # Adjust x position for padding
        y = row * (CELL_SIZE + PADDING)  # Adjust y position for padding
        pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)


def draw_grid():
    for i in range(25):  # Now iterate from 0 to 24
        if i == 12:  # Skip the middle cell
            continue
        row = i // GRID_SIZE  # Adjust row
        col = i % GRID_SIZE  # Adjust column
        x = col * (CELL_SIZE + PADDING)  # Adjust x position for padding
        y = row * (CELL_SIZE + PADDING)  # Adjust y position for padding
        pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)

def reset_game():
    global grid, score, bomb_indices
    grid = [0] * 25  # Reset grid
    num_bombs = 5  # Number of bombs
    bomb_indices = random.sample([i for i in range(25) if i != 12], num_bombs)  # New bomb placement
    for index in bomb_indices:
        grid[index] = -1
    score = 0  # Reset score

def handle_click(x, y):
    global score
    col = x // (CELL_SIZE + PADDING)  # Adjust for padding
    row = y // (CELL_SIZE + PADDING)  # Adjust for padding
    index = row * GRID_SIZE + col
    
    if index >= 12:  # Adjust for the missing cell
        index += 1

    if grid[index] == -1:
        print("Boom! You clicked on a bomb cell.")  # Print message for bomb cell
        reset_game()  # Reset game if a bomb is clicked
    else:
        score += calculate_score(index)
        print(f"You clicked on an empty cell. Current score: {score}")  # Print message for empty cell

def calculate_score(index):
    score = 0
    row = index // GRID_SIZE
    col = index % GRID_SIZE

    for r in range(max(0, row - 1), min(GRID_SIZE, row + 2)):
        for c in range(max(0, col - 1), min(GRID_SIZE, col + 2)):
            adj_index = r * GRID_SIZE + c
            if adj_index == 12:  # Skip the middle cell
                continue
            if adj_index != index and grid[adj_index] == -1:
                distance = ((row - r) ** 2 + (col - c) ** 2) ** 0.5
                score += 1.4 ** distance

    return int(score)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            handle_click(x, y)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Press 'r' to restart
                reset_game()

    screen.fill(WHITE)
    draw_grid()

    # Display score
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (10, WINDOW_HEIGHT - 40))

    pygame.display.flip()

pygame.quit()
