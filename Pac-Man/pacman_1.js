// Get the canvas element
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
// Fill the rest of the canvas with empty space
ctx.fillStyle = BACKGROUND_COLOR;
ctx.fillRect(0, MAZE.length * 20, canvas.width, canvas.height - MAZE.length * 20);


// Set the canvas dimensions
canvas.width = 640;
canvas.height = 480;

// Define some colors
const PACMAN_COLOR = 'yellow';
const PELLET_COLOR = 'white';
const GHOST_COLOR = 'red';
const WALL_COLOR = 'blue';
const BACKGROUND_COLOR = 'black';

// Set the canvas dimensions to fill the entire window
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Define the maze
const MAZE = [
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "X                                               X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X  X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X",
  "X                                               X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X  X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X",
  "X                                               X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X",
  "X                                               X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X  X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X",
  "X                                               X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X  X",
  "X  X                                           X  X",
  "X  X                                           X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X",
  "X                                               X",
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
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
  // Ghost 5
  switch (ghost5.direction) {
    case 'up':
      if (isWall(ghost5.x, ghost5.y - ghost5.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost5.y -= ghost5.speed;
      }
      break;
    case 'down':
      if (isWall(ghost5.x, ghost5.y + ghost5.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost5.y += ghost5.speed;
      }
      break;
    case 'left':
      if (isWall(ghost5.x - ghost5.speed, ghost5.y)) {
        // Do nothing, just stop moving
      } else {
        ghost5.x -= ghost5.speed;
      }
      break;
    case 'right':
      if (isWall(ghost5.x + ghost5.speed, ghost5.y)) {
        // Do nothing, just stop moving
      } else {
        ghost5.x += ghost5.speed;
      }
      break;
  }

  // Update ghost 6 position
  switch (ghost6.direction) {
    case 'up':
      if (isWall(ghost6.x, ghost6.y - ghost6.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost6.y -= ghost6.speed;
      }
      break;
    case 'down':
      if (isWall(ghost6.x, ghost6.y + ghost6.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost6.y += ghost6.speed;
      }
      break;
    case 'left':
      if (isWall(ghost6.x - ghost6.speed, ghost6.y)) {
        // Do nothing, just stop moving
      } else {
        ghost6.x -= ghost6.speed;
      }
      break;
    case 'right':
      if (isWall(ghost6.x + ghost6.speed, ghost6.y)) {
        // Do nothing, just stop moving
      } else {
        ghost6.x += ghost6.speed;
      }
      break;
  }

  // Update ghost 7 position
  switch (ghost7.direction) {
    case 'up':
      if (isWall(ghost7.x, ghost7.y - ghost7.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost7.y -= ghost7.speed;
      }
      break;
    case 'down':
      if (isWall(ghost7.x, ghost7.y + ghost7.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost7.y += ghost7.speed;
      }
      break;
    case 'left':
      if (isWall(ghost7.x - ghost7.speed, ghost7.y)) {
        // Do nothing, just stop moving
      } else {
        ghost7.x -= ghost7.speed;
      }
      break;
    case 'right':
      if (isWall(ghost7.x + ghost7.speed, ghost7.y)) {
        // Do nothing, just stop moving
      } else {
        ghost7.x += ghost7.speed;
      }
      break;
  }

  // Update ghost 8 position
  switch (ghost8.direction) {
    case 'up':
      if (isWall(ghost8.x, ghost8.y - ghost8.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost8.y -= ghost8.speed;
      }
      break;
    case 'down':
      if (isWall(ghost8.x, ghost8.y + ghost8.speed)) {
        // Do nothing, just stop moving
      } else {
        ghost8.y += ghost8.speed;
      }
      break;
    case 'left':
      if (isWall(ghost8.x - ghost8.speed, ghost8.y)) {
        // Do nothing, just stop moving
      } else {
        ghost8.x -= ghost8.speed;
      }
      break;
    case 'right':
      if (isWall(ghost8.x + ghost8.speed, ghost8.y)) {
        // Do nothing, just stop moving
      } else {
        ghost8.x += ghost8.speed;
      }
      break;
  }
  
  if (ghostsVulnerable) {
    // Make the ghosts blue and vulnerable
    ghost1.color = 'blue';
    ghost2.color = 'blue';
    ghost3.color = 'blue';
    ghost4.color = 'blue';
    ghost5.color = 'blue';
    ghost6.color = 'blue';
    ghost7.color = 'blue';
    ghost8.color = 'blue';
  }
}

// Change the ghost direction every three seconds
setInterval(() => {
  ghost1.direction = getRandomDirection();
  ghost2.direction = getRandomDirection();
  ghost3.direction = getRandomDirection();
  ghost4.direction = getRandomDirection();
  ghost5.direction = getRandomDirection();
  ghost6.direction = getRandomDirection();
  ghost7.direction = getRandomDirection();
  ghost8.direction = getRandomDirection();
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

  ctx.beginPath();
  ctx.arc(ghost5.x, ghost5.y, ghost5.radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(ghost6.x, ghost6.y, ghost6.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(ghost7.x, ghost7.y, ghost7.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(ghost8.x, ghost8.y, ghost8.radius, 0, Math.PI * 2);
  ctx.fill();
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

// Create additional ghost objects
const ghost5 = {
  x: 300,
  y: 300,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost6 = {
  x: 350,
  y: 350,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost7 = {
  x: 400,
  y: 400,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost8 = {
  x: 450,
  y: 450,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

// Draw the win or lose screen
function drawWinOrLoseScreen() {
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'white';
  ctx.font = '48px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  if (player.lives <= 0) {
    ctx.fillText('Game Over: You Lose!', canvas.width / 2, canvas.height / 2);
  } else if (pellets.length === 0) {
    ctx.fillText('Game Over: You Win!', canvas.width / 2, canvas.height / 2);
  }
  ctx.font = '24px Arial';
  ctx.fillText('Press Space to Restart', canvas.width / 2, canvas.height / 2 + 50);
}


// Game loop
function gameLoop() {
  if (gameOver) {
    drawWinOrLoseScreen();
  } else {
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
  }

  requestAnimationFrame(gameLoop);
}

// Key press event handler
document.addEventListener('keydown', (e) => {
  if (e.key === ' ') {
    if (gameOver) {
      restartGame();
    }
  }
  keyPressed = e.key;
});

document.addEventListener('keyup', (e) => {
  keyPressed = '';
});

// Restart the game
function restartGame() {
  player.x = 50;
  player.y = 50;
  player.lives = 3;
  player.immune = false;
  pellets.length = 0;
  score = 0;
  gameOver = false;
  init();
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
let pellets = [];

// Define the score and game over variables
let score = 0;
let gameOver = false;

function drawMaze() {
  for (let i = 0; i < MAZE.length; i++) {
    for (let j = 0; j < MAZE[i].length; j++) {
      if (MAZE[i][j] === 'X') {
        ctx.fillStyle = WALL_COLOR;
        ctx.fillRect(j * 20, i * 20, 20, 20);
      }
    }
  }
  // Fill the rest of the canvas with empty space
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, MAZE.length * 20, canvas.width, canvas.height - MAZE.length * 20);
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
    if (pellets[i].type === 'power') {
      ctx.fillStyle = 'red';
      ctx.beginPath();
      ctx.arc(pellet.x, pellet.y, pellet.radius, 0, Math.PI * 2);
      ctx.fill();
    } else {
    ctx.fillStyle = PELLET_COLOR;
    ctx.beginPath();
    ctx.arc(pellets[i].x, pellets[i].y, 5, 0, Math.PI * 2);
    ctx.fill();
    }
  }
}

// Create a Power Pellet object
const powerPellet = {
  x: 200,
  y: 200,
  radius: 10,
  type: 'power',
};

// Add the Power Pellet to the pellets array
pellets.push(powerPellet);

// Initialize the maze
function init() {
  pellets = [];
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

// Update the player position
function updatePlayer() {
  if (keyPressed === 'ArrowUp' && !isWall(player.x, player.y - player.speed - player.radius)) {
    player.y -= player.speed;
  } else if (keyPressed === 'ArrowDown' && !isWall(player.x, player.y + player.speed + player.radius)) {
    player.y += player.speed;
  } else if (keyPressed === 'ArrowLeft' && !isWall(player.x - player.speed - player.radius, player.y)) {
    player.x -= player.speed;
  } else if (keyPressed === 'ArrowRight' && !isWall(player.x + player.speed + player.radius, player.y)) {
    player.x += player.speed;
  }
}
  
  // Check for collisions between the player, ghosts, and pellets
  function checkCollisions() {
    // Check for collisions with ghosts
    if (checkGhostCollision(ghost1) || checkGhostCollision(ghost2) || checkGhostCollision(ghost3) || checkGhostCollision(ghost4)) {
      if (!player.immune) {
        player.lives--;
        player.immune = true;
        setTimeout(() => {
          player.immune = false;
        }, 2000);
      }
    }
  
    // Check for collisions with pellets
    for (let i = pellets.length - 1; i >= 0; i--) {
      if (checkPelletCollision(pellets[i])) {
        pellets.splice(i, 1);
        score++;
        if (pellets[i].type === 'power') {
          // Make the ghosts vulnerable for 10 seconds
          ghostsVulnerable = true;
          setTimeout(() => {
            ghostsVulnerable = false;
          }, 10000);
        }
      }
    }
  
    // Check if the game is over
    if (player.lives <= 0) {
      gameOver = true;
    }
  }

  
  // Helper function to check for collisions with ghosts
  function checkGhostCollision(ghost) {
    const distance = Math.sqrt(Math.pow(player.x - ghost.x, 2) + Math.pow(player.y - ghost.y, 2));
    return distance < player.radius + ghost.radius;
  }
  
  // Helper function to check for collisions with pellets
  function checkPelletCollision(pellet) {
    const distance = Math.sqrt(Math.pow(player.x - pellet.x, 2) + Math.pow(player.y - pellet.y, 2));
    return distance < player.radius + 5;
  }
// Initialize the game and start the game loop
init();
gameLoop();