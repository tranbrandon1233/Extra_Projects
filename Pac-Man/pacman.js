// Get the canvas element
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set the canvas dimensions
canvas.width = 640;
canvas.height = 480;

// Define some colors
const PACMAN_COLOR = 'yellow';
const PELLET_COLOR = 'white';
const GHOST_COLOR = 'red';
const WALL_COLOR = 'blue';
const BACKGROUND_COLOR = 'black';

// Define the maze
const MAZE = [
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "X                             X",
  "X  XXX  XXX  XXX  XXX  XXX  X",
  "X  X     X     X     X     X  X",
  "X  X  XXX  XXX  XXX  XXX  X  X",
  "X  X  X     X     X     X  X  X",
  "X  XXX  XXX  XXX  XXX  XXX  X",
  "X     X     X     X     X     X",
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "X                             X",
  "X  XXX  XXX  XXX  XXX  XXX  X",
  "X  X     X     X     X     X  X",
  "X  X  XXX  XXX  XXX  XXX  X  X",
  "X  X  X     X     X     X  X  X",
  "X  XXX  XXX  XXX  XXX  XXX  X",
  "X     X     X     X     X     X",
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
];

// Define the player and ghost objects
const player = {
  x: 50,
  y: 50,
  radius: 10,
  speed: 5,
  lives: 3,
  immune: false,
};

const ghost1 = {
  x: 100,
  y: 100,
  radius: 10,
  speed: 2,
};

const ghost2 = {
  x: 150,
  y: 150,
  radius: 10,
  speed: 2,
};

// Define the pellet array
const pellets = [];

// Define the score and game over variables
let score = 0;
let gameOver = false;

// Draw the maze
function drawMaze() {
  for (let i = 0; i < MAZE.length; i++) {
    for (let j = 0; j < MAZE[i].length; j++) {
      if (MAZE[i][j] === 'X') {
        ctx.fillStyle = WALL_COLOR;
        ctx.fillRect(j * 20, i * 20, 20, 20);
      }
    }
  }
}

// Draw the player
function drawPlayer() {
  ctx.fillStyle = PACMAN_COLOR;
  ctx.beginPath();
  ctx.arc(player.x, player.y, player.radius, 0, Math.PI * 2);
  ctx.fill();
}

// Draw the ghosts
function drawGhosts() {
  ctx.fillStyle = GHOST_COLOR;
  ctx.beginPath();
  ctx.arc(ghost1.x, ghost1.y, ghost1.radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(ghost2.x, ghost2.y, ghost2.radius, 0, Math.PI * 2);
  ctx.fill();
}

// Draw the pellets
function drawPellets() {
  for (let i = 0; i < pellets.length; i++) {
    ctx.fillStyle = PELLET_COLOR;
    ctx.beginPath();
    ctx.arc(pellets[i].x, pellets[i].y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Update the player position
function updatePlayer() {
  if (player.immune) {
    setTimeout(() => {
      player.immune = false;
    }, 3000);
  }

  if (keyPressed === 'ArrowUp') {
    if (player.y - player.speed >= 0) {
      player.y -= player.speed;
    }
  } else if (keyPressed === 'ArrowDown') {
    if (player.y + player.speed <= canvas.height) {
      player.y += player.speed;
    }
  } else if (keyPressed === 'ArrowLeft') {
    if (player.x - player.speed >= 0) {
      player.x -= player.speed;
    }
  } else if (keyPressed === 'ArrowRight') {
    if (player.x + player.speed <= canvas.width) {
      player.x += player.speed;
    }
  }

  for (let i = 0; i < MAZE.length; i++) {
    for (let j = 0; j < MAZE[i].length; j++) {
      if (MAZE[i][j] === 'X') {
        if (player.x + player.radius > j * 20 && player.x - player.radius < j * 20 + 20 && player.y + player.radius > i * 20 && player.y - player.radius < i * 20 + 20) {
          if (keyPressed === 'ArrowUp') {
            player.y += player.speed;
          } else if (keyPressed === 'ArrowDown') {
            player.y -= player.speed;
          } else if (keyPressed === 'ArrowLeft') {
            player.x += player.speed;
          } else if (keyPressed === 'ArrowRight') {
            player.x -= player.speed;
          }
        }
      }
    }
  }
}

// Update the ghost positions
function updateGhosts() {
  if (ghost1.x < player.x) {
    if (ghost1.x + ghost1.speed <= canvas.width) {
      ghost1.x += ghost1.speed;
    }
  } else if (ghost1.x > player.x) {
    if (ghost1.x - ghost1.speed >= 0) {
      ghost1.x -= ghost1.speed;
    }
  }

  if (ghost1.y < player.y) {
    if (ghost1.y + ghost1.speed <= canvas.height) {
      ghost1.y += ghost1.speed;
    }
  } else if (ghost1.y > player.y) {
    if (ghost1.y - ghost1.speed >= 0) {
      ghost1.y -= ghost1.speed;
    }
  }

  if (ghost2.x < player.x) {
    if (ghost2.x + ghost2.speed <= canvas.width) {
      ghost2.x += ghost2.speed;
    }
  } else if (ghost2.x > player.x) {
    if (ghost2.x - ghost2.speed >= 0) {
      ghost2.x -= ghost2.speed;
    }
  }

  if (ghost2.y < player.y) {
    if (ghost2.y + ghost2.speed <= canvas.height) {
      ghost2.y += ghost2.speed;
    }
  } else if (ghost2.y > player.y) {
    if (ghost2.y - ghost2.speed >= 0) {
      ghost2.y -= ghost2.speed;
    }
  }

  for (let i = 0; i < MAZE.length; i++) {
    for (let j = 0; j < MAZE[i].length; j++) {
      if (MAZE[i][j] === 'X') {
        if (ghost1.x + ghost1.radius > j * 20 && ghost1.x - ghost1.radius < j * 20 + 20 && ghost1.y + ghost1.radius > i * 20 && ghost1.y - ghost1.radius < i * 20 + 20) {
          if (ghost1.x < player.x) {
            ghost1.x -= ghost1.speed;
          } else if (ghost1.x > player.x) {
            ghost1.x += ghost1.speed;
          }

          if (ghost1.y < player.y) {
            ghost1.y -= ghost1.speed;
          } else if (ghost1.y > player.y) {
            ghost1.y += ghost1.speed;
          }
        }

        if (ghost2.x + ghost2.radius > j * 20 && ghost2.x - ghost2.radius < j * 20 + 20 && ghost2.y + ghost2.radius > i * 20 && ghost2.y - ghost2.radius < i * 20 + 20) {
          if (ghost2.x < player.x) {
            ghost2.x -= ghost2.speed;
          } else if (ghost2.x > player.x) {
            ghost2.x += ghost2.speed;
          }

          if (ghost2.y < player.y) {
            ghost2.y -= ghost2.speed;
          } else if (ghost2.y > player.y) {
            ghost2.y += ghost2.speed;
          }
        }
      }
    }
  }
}

// Check for collisions
function checkCollisions() {
  for (let i = 0; i < pellets.length; i++) {
    if (Math.hypot(player.x - pellets[i].x, player.y - pellets[i].y) < player.radius + 5) {
      pellets.splice(i, 1);
      score++;
    }
  }

  if (Math.hypot(player.x - ghost1.x, player.y - ghost1.y) < player.radius + ghost1.radius) {
    if (!player.immune) {
      player.lives--;
      player.immune = true;
    }
  }

  if (Math.hypot(player.x - ghost2.x, player.y - ghost2.y) < player.radius + ghost2.radius) {
    if (!player.immune) {
      player.lives--;
      player.immune = true;
    }
  }

  if (player.lives <= 0) {
    gameOver = true;
  }
}

// Initialize the game
function init() {
  for (let i = 0; i < MAZE.length; i++) {
    for (let j = 0; j < MAZE[i].length; j++) {
      if (MAZE[i][j] === ' ') {
        pellets.push({ x: j * 20 + 10, y: i * 20 + 10 });
      }
    }
  }
}

// Draw the score and lives
function drawScoreAndLives() {
  ctx.fillStyle = 'white';
  ctx.font = '24px Arial';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(`Score: ${score}`, 10, 10);
  ctx.textAlign = 'right';
  ctx.fillText(`Lives: ${player.lives}`, canvas.width - 10, 10);
}

// Game loop
function gameLoop() {
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawMaze();
  drawPlayer();
  drawGhosts();
  drawPellets();
  drawScoreAndLives();
  updatePlayer();
  updateGhosts();
  checkCollisions();

  if (gameOver) {
    ctx.fillStyle = 'white';
    ctx.font = '48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Game Over', canvas.width / 2, canvas.height / 2);
    return;
  }

  requestAnimationFrame(gameLoop);
}

// Key press event handler
let keyPressed = '';
document.addEventListener('keydown', (e) => {
  keyPressed = e.key;
});

document.addEventListener('keyup', (e) => {
  keyPressed = '';
});

// Initialize the game and start the game loop
init();
gameLoop();