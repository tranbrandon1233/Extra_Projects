import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions and grid sizes
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
COLUMNS = SCREEN_WIDTH // BLOCK_SIZE
ROWS = SCREEN_HEIGHT // BLOCK_SIZE
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
PURPLE = (160, 32, 240)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# Tetrimino shapes and their rotations
TETRIMINOS = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]]  # L
]

# Tetrimino colors
TETRIMINO_COLORS = [CYAN, YELLOW, PURPLE, GREEN, RED, BLUE, ORANGE]

# Tetrimino class
class Tetrimino:
    def __init__(self):
        self.shape = random.choice(TETRIMINOS)
        self.color = random.choice(TETRIMINO_COLORS)
        self.x = COLUMNS // 2 - len(self.shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

# Tetris game class
class Tetris:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.board = [[BLACK for _ in range(COLUMNS)] for _ in range(ROWS)]
        self.score = 0
        self.current_tetrimino = Tetrimino()
        self.game_over = False
        
        # Add fall speed and timer
        self.fall_speed = 25  # Speed in milliseconds (lower = faster)
        self.fall_time = 0  # Timer to track when to move Tetrimino down

    def draw_grid(self):
        for y in range(ROWS):
            for x in range(COLUMNS):
                pygame.draw.rect(self.screen, WHITE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

    def draw_board(self):
        for y in range(ROWS):
            for x in range(COLUMNS):
                color = self.board[y][x]
                if color != BLACK:
                    pygame.draw.rect(self.screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def spawn_new_tetrimino(self, tetrimino):
        for y, row in enumerate(tetrimino.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, tetrimino.color, 
                                     ((tetrimino.x + x) * BLOCK_SIZE, (tetrimino.y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def is_valid_position(self, tetrimino, adj_x=0, adj_y=0):
        for y, row in enumerate(tetrimino.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = tetrimino.x + x + adj_x
                    new_y = tetrimino.y + y + adj_y
                    if new_x < 0 or new_x >= COLUMNS or new_y >= ROWS:
                        return False
                    if new_y >= 0 and self.board[new_y][new_x] != BLACK:
                        return False
        return True

    def place_tetrimino(self, tetrimino):
        for y, row in enumerate(tetrimino.shape):
            for x, cell in enumerate(row):
                if cell:
                    self.board[tetrimino.y + y][tetrimino.x + x] = tetrimino.color
        self.clear_lines()

    def clear_lines(self):
        lines_cleared = 0
        for y in range(ROWS):
            if all(self.board[y][x] != BLACK for x in range(COLUMNS)):
                del self.board[y]
                self.board.insert(0, [BLACK for _ in range(COLUMNS)])
                lines_cleared += 1
        self.score += lines_cleared * 100

    def run(self):
        while not self.game_over:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.is_valid_position(self.current_tetrimino, adj_x=-1):
                    self.current_tetrimino.x -= 1
                if event.key == pygame.K_RIGHT and self.is_valid_position(self.current_tetrimino, adj_x=1):
                    self.current_tetrimino.x += 1
                if event.key == pygame.K_DOWN and self.is_valid_position(self.current_tetrimino, adj_y=1):
                    self.current_tetrimino.y += 1
                if event.key == pygame.K_UP:
                    rotated_tetrimino = Tetrimino()
                    rotated_tetrimino.shape = [list(row) for row in zip(*self.current_tetrimino.shape[::-1])]
                    rotated_tetrimino.x = self.current_tetrimino.x
                    rotated_tetrimino.y = self.current_tetrimino.y
                    if self.is_valid_position(rotated_tetrimino):
                        self.current_tetrimino.rotate()

    def update(self):
        # Track the time for block fall speed
        self.fall_time += self.clock.get_rawtime()
        
        if self.fall_time > self.fall_speed:
            if self.is_valid_position(self.current_tetrimino, adj_y=1):
                self.current_tetrimino.y += 1
            else:
                self.place_tetrimino(self.current_tetrimino)
                self.current_tetrimino = Tetrimino()
                if not self.is_valid_position(self.current_tetrimino):
                    self.game_over = True
            self.fall_time = 0  # Reset the timer after moving

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_grid()
        self.draw_board()
        self.spawn_new_tetrimino(self.current_tetrimino)
        self.display_score()

    def display_score(self):
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

# Run the game
if __name__ == "__main__":
    game = Tetris()
    game.run()