import pygame
import random
import time
import string

# Initialize Pygame
pygame.init()
pygame.font.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Typing Speed Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Fonts
FONT = pygame.font.Font(None, 36)
LARGE_FONT = pygame.font.Font(None, 48)

# Game constants
WORD_SPEED_BASE = 1
WORD_SPAWN_RATE = 2000  # milliseconds
MAX_MISSED_WORDS = 5
TARGET_WORDS = 50

# Sample word list (you can expand this)
WORDS = [
    "python", "programming", "keyboard", "computer", "algorithm",
    "developer", "software", "typing", "practice", "speed",
    "coding", "learning", "game", "skills", "improve",
    "fast", "accuracy", "challenge", "score", "player"
]

class Word:
    def __init__(self, text, x, y):
        self.text = text
        self.x = x
        self.y = y
        self.speed = WORD_SPEED_BASE * (10 / max(len(text), 5))  # Longer words fall slower
        self.color = BLACK
        self.active = False  # Currently being typed

class Game:
    def __init__(self):
        self.reset_game()
        self.leaderboard = []

    def reset_game(self):
        """Reset the game while preserving the leaderboard."""
        self.words = []
        self.current_input = ""
        self.score = 0
        self.missed_words = 0
        self.correct_words = 0
        self.start_time = time.time()
        self.end_time = None
        self.last_spawn = pygame.time.get_ticks()
        self.game_over = False
        self.player_name = ""
        self.entering_name = True
        self.accuracy_count = {"correct": 0, "total": 0}
        self.final_wpm = 0
        self.final_accuracy = 0

    def spawn_word(self):
        word = random.choice(WORDS)
        x = random.randint(10, SCREEN_WIDTH - 100)
        word_obj = Word(word, x, 0)
        self.words.append(word_obj)
    
    def calculate_stats(self):
        if self.game_over:
            return self.final_wpm, self.final_accuracy
        else:
            elapsed_time = time.time() - self.start_time
            wpm = (self.correct_words / elapsed_time) * 60
            accuracy = (self.accuracy_count["correct"] / max(1, self.accuracy_count["total"])) * 100
            return wpm, accuracy

    def end_game(self):
        self.game_over = True
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.final_wpm = (self.correct_words / elapsed_time) * 60
        self.final_accuracy = (self.accuracy_count["correct"] / max(1, self.accuracy_count["total"])) * 100
        # Pass score along with other stats to update_leaderboard
        self.update_leaderboard(self.player_name, self.score, self.final_wpm, self.final_accuracy)


    def update_leaderboard(self, name, score, wpm, accuracy):
        entry = {"name": name, "score": score, "wpm": wpm, "accuracy": accuracy}
        self.leaderboard.append(entry)
        # Sort the leaderboard by `score` in descending order
        self.leaderboard.sort(key=lambda x: x["score"], reverse=True)
        # Keep only the top 10 entries
        self.leaderboard = self.leaderboard[:10]

def main():
    game = Game()
    clock = pygame.time.Clock()
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if game.entering_name:
                    if event.key == pygame.K_RETURN and game.player_name:
                        game.entering_name = False
                        game.start_time = time.time()
                    elif event.key == pygame.K_BACKSPACE:
                        game.player_name = game.player_name[:-1]
                    elif len(game.player_name) < 20 and event.unicode.isalnum():
                        game.player_name += event.unicode
                elif not game.game_over:
                    if event.key == pygame.K_RETURN:
                        matched = False
                        for word in game.words:
                            if game.current_input == word.text:
                                game.words.remove(word)
                                game.score += len(word.text)
                                game.correct_words += 1
                                game.accuracy_count["correct"] += len(word.text)
                                matched = True
                                break
                        game.accuracy_count["total"] += len(game.current_input)
                        game.current_input = ""
                    elif event.key == pygame.K_BACKSPACE:
                        game.current_input = game.current_input[:-1]
                    elif event.unicode in string.ascii_letters:
                        game.current_input += event.unicode
                elif game.game_over:
                    # Press 'R' to restart the game
                    if event.key == pygame.K_r:
                        game.reset_game()
        
        if game.entering_name:
            screen.fill(WHITE)
            name_text = FONT.render("Enter your name:", True, BLACK)
            input_text = FONT.render(game.player_name, True, BLUE)
            screen.blit(name_text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50))
            screen.blit(input_text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))
        
        elif not game.game_over:
            if current_time - game.last_spawn > WORD_SPAWN_RATE:
                game.spawn_word()
                game.last_spawn = current_time
            
            for word in game.words[:]:
                word.y += word.speed
                if word.y > SCREEN_HEIGHT:
                    game.words.remove(word)
                    game.missed_words += 1
            
            if game.missed_words >= MAX_MISSED_WORDS or game.correct_words >= TARGET_WORDS:
                game.end_game()
            
            screen.fill(WHITE)
            for word in game.words:
                color = BLUE if game.current_input == word.text[:len(game.current_input)] else BLACK
                word_surface = FONT.render(word.text, True, color)
                screen.blit(word_surface, (word.x, word.y))
            
            input_surface = FONT.render(f"> {game.current_input}", True, GREEN)
            screen.blit(input_surface, (10, SCREEN_HEIGHT - 40))
            
            stats_text = FONT.render(f"Score: {game.score} | Missed: {game.missed_words}/{MAX_MISSED_WORDS} | Words: {game.correct_words}/{TARGET_WORDS}", True, BLACK)
            screen.blit(stats_text, (10, 10))
            
        else:
            screen.fill(WHITE)
            game_over_text = LARGE_FONT.render("Game Over!", True, RED)
            stats_text = FONT.render(f"WPM: {game.final_wpm:.1f} | Accuracy: {game.final_accuracy:.1f}%", True, BLACK)
            score_text = FONT.render(f"Final Score: {game.score}", True, BLACK)
            restart_text = FONT.render("Press 'R' to Restart", True, BLUE)
            
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 100, 50))
            screen.blit(stats_text, (SCREEN_WIDTH // 2 - 150, 120))
            screen.blit(score_text, (SCREEN_WIDTH // 2 - 100, 170))
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - 100, 230))
            
            leaderboard_text = FONT.render("Top 10 Players:", True, BLUE)
            screen.blit(leaderboard_text, (50, 280))
            for i, entry in enumerate(game.leaderboard):
                text = f"{i+1}. {entry['name']}: Score {entry['score']}, {entry['wpm']:.1f} WPM, {entry['accuracy']:.1f}%"
                entry_text = FONT.render(text, True, BLACK)
                screen.blit(entry_text, (50, 320 + i * 30))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()