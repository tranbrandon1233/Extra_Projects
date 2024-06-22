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
  "X  X                       X  X",
  "X  X                       X  X",
  "X  X  XXX  XXX  XXX  XXX  X  X",
  "X  X                       X  X",
  "X  X                       X  X",
  "X  XXX  XXX  XXX  XXX  XXX  X",
  "X                             X",
  "X  XXX  XXX  XXX  XXX  XXX  X",
  "X  X                       X  X",
  "X  X                       X  X",
  "X  X  XXX  XXX  XXX  XXX  X  X",
  "X  X                       X  X",
  "X  X                       X  X",
  "X  XXX  XXX  XXX  XXX  XXX  X",
  "X                             X",
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
];

// Update the ghost positions
function updateGhosts() {
  // Update ghost 1 position
  switch (ghost1.direction) {
    case 'up':
      if (isWall(ghost1.x, ghost1.y - ghost1.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost1.y -= ghost1.speed;
      }
      break;
    case 'down':
      if (isWall(ghost1.x, ghost1.y + ghost1.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost1.y += ghost1.speed;
      }
      break;
    case 'left':
      if (isWall(ghost1.x - ghost1.speed, ghost1.y)) {
        // Do nothing, just stop moving
      } else {
        ghost1.x -= ghost1.speed;
      }
      break;
    case 'right':
      if (isWall(ghost1.x + ghost1.speed, ghost1.y)) {
        // Do nothing, just stop moving
      } else {
        ghost1.x += ghost1.speed;
      }
      break;
  }

  // Update ghost 2 position
  switch (ghost2.direction) {
    case 'up':
      if (isWall(ghost2.x, ghost2.y - ghost2.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost2.y -= ghost2.speed;
      }
      break;
    case 'down':
      if (isWall(ghost2.x, ghost2.y + ghost2.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost2.y += ghost2.speed;
      }
      break;
    case 'left':
      if (isWall(ghost2.x - ghost2.speed, ghost2.y)) {
        // Do nothing, just stop moving
      } else {
        ghost2.x -= ghost2.speed;
      }
      break;
    case 'right':
      if (isWall(ghost2.x + ghost2.speed, ghost2.y)) {
        // Do nothing, just stop moving
      } else {
        ghost2.x += ghost2.speed;
      }
      break;
  }

  // Update ghost 3 position
  switch (ghost3.direction) {
    case 'up':
      if (isWall(ghost3.x, ghost3.y - ghost3.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost3.y -= ghost3.speed;
      }
      break;
    case 'down':
      if (isWall(ghost3.x, ghost3.y + ghost3.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost3.y += ghost3.speed;
      }
      break;
    case 'left':
      if (isWall(ghost3.x - ghost3.speed, ghost3.y)) {
        // Do nothing, just stop moving
      } else {
        ghost3.x -= ghost3.speed;
      }
      break;
    case 'right':
      if (isWall(ghost3.x + ghost3.speed, ghost3.y)) {
        // Do nothing, just stop moving
      } else {
        ghost3.x += ghost3.speed;
      }
      break;
  }

  // Update ghost 4 position
  switch (ghost4.direction) {
    case 'up':
      if (isWall(ghost4.x, ghost4.y - ghost4.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost4.y -= ghost4.speed;
      }
      break;
    case 'down':
      if (isWall(ghost4.x, ghost4.y + ghost4.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost4.y += ghost4.speed;
      }
      break;
    case 'left':
      if (isWall(ghost4.x - ghost4.speed, ghost4.y)) {
        // Do nothing, just stop moving
      } else {
        ghost4.x -= ghost4.speed;
      }
      break;
    case 'right':
      if (isWall(ghost4.x + ghost4.speed, ghost4.y)) {
        // Do nothing, just stop moving
      } else {
        ghost4.x += ghost4.speed;
      }
      break;
  }
}

// Change the ghost direction every three seconds
setInterval(() => {
  ghost1.direction = getRandomDirection();
  ghost2.direction = getRandomDirection();
  ghost3.direction = getRandomDirection();
  ghost4.direction = getRandomDirection();
}, 1000);

// Check if a wall is at the given position
function isWall(x, y) {
  const wallX = Math.floor(x / 20);
  const wallY = Math.floor(y / 20);
  return MAZE[wallY][wallX] === 'X';
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

  ctx.beginPath();
  ctx.arc(ghost3.x, ghost3.y, ghost3.radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(ghost4.x, ghost4.y, ghost4.radius, 0, Math.PI * 2);
  ctx.fill();
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

  if (Math.hypot(player.x - ghost3.x, player.y - ghost3.y) < player.radius + ghost3.radius) {
    if (!player.immune) {
      player.lives--;
      player.immune = true;
    }
  }

  if (Math.hypot(player.x - ghost4.x, player.y - ghost4.y) < player.radius + ghost4.radius) {
    if (!player.immune) {
      player.lives--;
      player.immune = true;
    }
  }

  if (player.lives <= 0 || pellets.length === 0) {
    gameOver = true;
  }
}

// Get a random direction
function getRandomDirection() {
  const directions = ['up', 'down', 'left', 'right'];
  return directions[Math.floor(Math.random() * directions.length)];
}

// Initialize the ghosts
const ghost1 = {
  x: 100,
  y: 100,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost2 = {
  x: 150,
  y: 150,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost3 = {
  x: 200,
  y: 200,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost4 = {
  x: 250,
  y: 200,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};


// Draw the win or lose screen
function drawWinOrLoseScreen() {
  if (gameOver) {
    ctx.fillStyle = 'white';
    ctx.font = '48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Game Over', canvas.width / 2, canvas.height / 2);
  } else if (pellets.length === 0) {
    ctx.fillStyle = 'white';
    ctx.font = '48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('You Win!', canvas.width / 2, canvas.height / 2);
  }
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
    if (player.lives <= 0) {
      ctx.fillText('Game Over: You Lose!', canvas.width / 2, canvas.height / 2);
    } else if (pellets.length === 0) {
      ctx.fillText('Game Over: You Win!', canvas.width / 2, canvas.height / 2);
    }
    return;
  }

  requestAnimationFrame(gameLoop);
}

// Define the player and ghost objects
const player = {
  x: 50,
  y: 50,
  radius: 10,
  speed: 5,
  lives: 3,
  immune: false,
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