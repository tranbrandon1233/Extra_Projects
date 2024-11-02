import pygame
import random
from pygame import Vector2

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SHOOTER_COST = 10
INITIAL_COINS = 100
INITIAL_HP = 5
COINS_PER_ROUND = 30
SHOOTER_RADIUS = 20
BALLOON_RADIUS = 15
PROJECTILE_RADIUS = 5
BALLOON_SPEED = 2
PROJECTILE_SPEED = 5
SHOOT_COOLDOWN = 60  # frames
INITIAL_BALLOONS = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Create window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Bloons-style Game")
clock = pygame.time.Clock()

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.font = pygame.font.Font(None, 36)
        
    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class Path:
    def __init__(self):
        # Define path control points
        self.points = [
            (WINDOW_WIDTH, 100),
            (600, 100),
            (500, 300),
            (300, 300),
            (200, 500),
            (0, 500)
        ]
        
    def draw(self, surface):
        pygame.draw.lines(surface, BLACK, False, self.points, 40)
        
    def get_position(self, t):
        # Linear interpolation between points based on t (0 to 1)
        if t >= 1:
            return Vector2(self.points[-1])
        
        total_segments = len(self.points) - 1
        segment = int(t * total_segments)
        t_segment = (t * total_segments) % 1
        
        start = Vector2(self.points[segment])
        end = Vector2(self.points[segment + 1])
        
        return start.lerp(end, t_segment)

class Shooter:
    def __init__(self, pos):
        self.pos = Vector2(pos)
        self.cooldown = 0
        
    def draw(self, surface):
        pygame.draw.circle(surface, BLUE, self.pos, SHOOTER_RADIUS)
        
    def can_shoot(self):
        return self.cooldown <= 0
        
    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

class Projectile:
    def __init__(self, pos, target_pos):
        self.pos = Vector2(pos)
        direction = Vector2(target_pos) - self.pos
        self.velocity = direction.normalize() * PROJECTILE_SPEED
        
    def update(self):
        self.pos += self.velocity
        
    def draw(self, surface):
        pygame.draw.circle(surface, YELLOW, self.pos, PROJECTILE_RADIUS)
        
    def is_off_screen(self):
        return (self.pos.x < 0 or self.pos.x > WINDOW_WIDTH or 
                self.pos.y < 0 or self.pos.y > WINDOW_HEIGHT)

class Balloon:
    def __init__(self, path, health):
        self.path = path
        self.t = 0  # Position along path (0 to 1)
        self.pos = self.path.get_position(0)
        self.health = health
        self.colors = [RED, YELLOW, GREEN]
        
    def update(self):
        self.t += BALLOON_SPEED / 1000
        self.pos = self.path.get_position(self.t)
        
    def draw(self, surface):
        color = self.colors[min(len(self.colors)-1, self.health-1)]
        pygame.draw.circle(surface, color, self.pos, BALLOON_RADIUS)
        
    def is_completed(self):
        return self.t >= 1
        
    def is_popped(self):
        return self.health <= 0

class Game:
    def __init__(self):
        self.reset()
        # Create buttons
        self.play_button = Button(WINDOW_WIDTH - 110, 10, 100, 40, "Play", GREEN)
        self.buy_shooter_button = Button(WINDOW_WIDTH - 110, 60, 100, 40, "Buy", BLUE)
        
    def reset(self):
        self.path = Path()
        self.shooters = []
        self.projectiles = []
        self.balloons = []
        self.coins = INITIAL_COINS
        self.hp = INITIAL_HP
        self.round = 0
        self.round_active = False
        self.placing_shooter = False
        self.balloons_to_spawn = 0
        
    def start_round(self):
        if not self.round_active:
            self.round += 1
            self.round_active = True
            self.balloons = []
            self.balloons_to_spawn = INITIAL_BALLOONS + (self.round - 1) * 3  # Increase by 3 each round
            
    def update(self):
        # Update shooters
        for shooter in self.shooters:
            shooter.update()
            if shooter.can_shoot() and self.balloons:
                # Find closest balloon
                closest = min(self.balloons, 
                            key=lambda b: (b.pos - shooter.pos).length())
                if (closest.pos - shooter.pos).length() < 200:
                    self.projectiles.append(Projectile(shooter.pos, closest.pos))
                    shooter.cooldown = SHOOT_COOLDOWN
        
        # Update projectiles
        for proj in self.projectiles:
            proj.update()
            if proj.is_off_screen():
                self.projectiles.remove(proj)
                continue
                
            # Check collision with balloons
            for balloon in self.balloons:
                if (balloon.pos - proj.pos).length() < BALLOON_RADIUS + PROJECTILE_RADIUS:
                    balloon.health -= 1
                    if balloon.is_popped():
                        self.balloons.remove(balloon)
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    break
        
        # Update balloons
        if self.round_active:
            # Spawn new balloons
            if self.balloons_to_spawn > 0 and random.random() < 0.02:
                self.balloons.append(Balloon(self.path,3+2*self.round))
                self.balloons_to_spawn -= 1
                
            for balloon in self.balloons:
                balloon.update()
                if balloon.is_completed():
                    self.balloons.remove(balloon)
                    self.hp -= 1
                    
            # Check if round is over
            if not self.balloons and self.balloons_to_spawn == 0:
                self.round_active = False
                self.coins += COINS_PER_ROUND
                
    def draw(self, surface):
        surface.fill(WHITE)
        
        # Draw path
        self.path.draw(surface)
        
        # Draw game objects
        for shooter in self.shooters:
            shooter.draw(surface)
        for proj in self.projectiles:
            proj.draw(surface)
        for balloon in self.balloons:
            balloon.draw(surface)
            
        # Draw UI
        font = pygame.font.Font(None, 36)
        hp_text = font.render(f"HP: {self.hp}", True, BLACK)
        coins_text = font.render(f"Coins: {self.coins}", True, BLACK)
        round_text = font.render(f"Round: {self.round}", True, BLACK)
        balloons_text = font.render(f"Balloons: {self.balloons_to_spawn + len(self.balloons)}", True, BLACK)
        
        surface.blit(hp_text, (10, 10))
        surface.blit(coins_text, (10, 50))
        surface.blit(round_text, (10, 90))
        surface.blit(balloons_text, (10, 130))
        
        # Draw buttons
        if not self.round_active:
            self.play_button.draw(surface)
        
        # Draw buy button (grayed out if can't afford)
        if self.coins >= SHOOTER_COST:
            self.buy_shooter_button.color = BLUE
        else:
            self.buy_shooter_button.color = GRAY
        self.buy_shooter_button.draw(surface)
            
        # Draw cursor for shooter placement
        if self.placing_shooter:
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(surface, BLUE, pos, SHOOTER_RADIUS, 2)

def main():
    game = Game()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    pos = pygame.mouse.get_pos()
                    
                    # Check play button
                    if not game.round_active and game.play_button.is_clicked(pos):
                        game.start_round()
                    
                    # Check buy button
                    elif (game.buy_shooter_button.is_clicked(pos) and 
                          game.coins >= SHOOTER_COST and 
                          not game.placing_shooter):
                        game.placing_shooter = True
                        
                    # Place shooter
                    elif game.placing_shooter:
                        # Check if position is valid (not on path)
                        path_color = screen.get_at(pos)
                        if path_color != BLACK:
                            game.shooters.append(Shooter(pos))
                            game.coins -= SHOOTER_COST
                            game.placing_shooter = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and game.placing_shooter:
                    game.placing_shooter = False
                    
        # Update game state
        if game.hp <= 0:
            game.reset()
        else:
            game.update()
            
        # Draw everything
        game.draw(screen)
        pygame.display.flip()
        
        # Control game speed
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()