// Get the canvas element
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Define some colors
const PACMAN_COLOR = 'yellow';
const PELLET_COLOR = 'white';
const GHOST_COLOR = 'black';
const WALL_COLOR = 'blue';
const BACKGROUND_COLOR = 'gray';

// Set the canvas dimensions to fill the entire window
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.style.overflow = 'hidden';


// Define the maze
const MAZE = [
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "X                                               X",
  "X  X              P                             X",
  "X  X                                            X",
  "X  X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X  X  XX",
  "X  X                                            X",
  "X  X                 P                          X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X                                               X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X              P                             X",
  "X  X                                            X",
  "X  X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X  X  XX",
  "X  X                 P                          X",
  "X  X                                            X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X                                               X",
  "XXXXXXXXXXXXX         XXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "XXXXXXXXXXXXX         XXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "X                                               X",
  "X  X               P                   P        X",
  "X  X                                         X  X",
  "X  X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  X   XXXX",
  "X  X   P                                      X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X                                          X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X                    P                     X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X                                         X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X                            P            X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X                                         X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X  P                                       X  X",
  "X  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XXX  XX    X",
  "X  X                            P            X  X",
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
];

function updateGhosts() {
  for (let i = 0; i < ghosts.length; i++) {
    // Update ghost position
    switch (ghosts[i].direction) {
      case 'up':
        if (isWall(ghosts[i].x, ghosts[i].y - ghosts[i].speed)) {
          // Do nothing, just stop moving
        } else {
          ghosts[i].y -= ghosts[i].speed;
        }
        break;
      case 'down':
        if (isWall(ghosts[i].x, ghosts[i].y + ghosts[i].speed)) {
          // Do nothing, just stop moving
        } else {
          ghosts[i].y += ghosts[i].speed;
        }
        break;
      case 'left':
        if (isWall(ghosts[i].x - ghosts[i].speed, ghosts[i].y)) {
          // Do nothing, just stop moving
        } else {
          ghosts[i].x -= ghosts[i].speed;
        }
        break;
      case 'right':
        if (isWall(ghosts[i].x + ghosts[i].speed, ghosts[i].y)) {
          // Do nothing, just stop moving
        } else {
          ghosts[i].x += ghosts[i].speed;
        }
        break;
    }
  }
}

// Draw the ghosts
function drawGhosts() {
  for (let i = 0; i < ghosts.length; i++) {
    ctx.fillStyle = ghosts[i].color || GHOST_COLOR;
    ctx.beginPath();
    ctx.arc(ghosts[i].x, ghosts[i].y, ghosts[i].radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Check for collisions with ghosts
function checkGhostCollision() {
  for (let i = 0; i < ghosts.length; i++) {
    const distance = Math.sqrt(Math.pow(player.x - ghosts[i].x, 2) + Math.pow(player.y - ghosts[i].y, 2));
    if (distance < player.radius + ghosts[i].radius) {
      if (ghosts[i].color === 'red') {
        // Delete the ghost
        ghosts.splice(i, 1);
        score += 100;
        if(ghosts.length === 0){
          gameOver = true;
        }
      } else {
        // Decrease player lives
        if(!player.immune){
        player.lives--;
        player.immune = true;
        setTimeout(() => {
          player.immune = false;
        }, 2000);
        }
      }
      return true;
    }
  }
  return false;
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

const ghost5 = {
  x: 100,
  y: 380,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost6 = {
  x: 150,
  y: 420,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost7 = {
  x: 200,
  y: 480,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

const ghost8 = {
  x: 250,
  y: 520,
  radius: 10,
  speed: 2,
  direction: getRandomDirection(),
};

// Define the ghosts array
let ghosts = [ghost1, ghost2, ghost3, ghost4, ghost5, ghost6, ghost7, ghost8];




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
  } else if (pellets.length === 0 | ghosts.length === 0) {
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
  let resetX = [100, 150, 200, 250, 100, 150, 200, 250];
  let resetY = [100, 150, 200, 200, 380, 420, 480, 520];
  player.x = 50;
  player.y = 50;
  player.lives = 3;
  player.immune = false;
  pellets.length = 0;
  score = 0;
  gameOver = false;
  ghosts = [ghost1, ghost2, ghost3, ghost4, ghost5, ghost6, ghost7, ghost8];
  for (let i = 0; i < ghosts.length; i++) {
    ghosts[i].x = resetX[i];
    ghosts[i].y = resetY[i];
    ghosts[i].direction = getRandomDirection();
    ghosts[i].color = 'black';
  } 

  init();
}

// Define the player and ghost objects
const player = {
  x: 50,
  y: 20,
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
// Update the game state
function checkCollisions() {
  // Check for collisions with ghosts
  checkGhostCollision();

  // Check for collisions with pellets
  for (let i = pellets.length - 1; i >= 0; i--) {
    checkPelletCollision(pellets[i]);
  }

  for (let i = powerPellets.length - 1; i >= 0; i--) {
    checkPelletCollision(powerPellets[i]);
  }

  // Check if the game is over
  if (player.lives <= 0) {
    gameOver = true;
  }
}
// Define the pellet array
let powerPellets = [];

// Initialize the pellets
function init() {
  pellets = [];
  powerPellets = [];
  for (let i = 0; i < MAZE.length; i++) {
    for (let j = 0; j < MAZE[i].length; j++) {
      if (MAZE[i][j] === ' ') {
        pellets.push({ x: j * 20 + 10, y: i * 20 + 10 });
      } else if (MAZE[i][j] === 'P') {
        powerPellets.push({ x: j * 20 + 10, y: i * 20 + 10 });
      }
    }
  }
}


// Draw the pellets
function drawPellets() {
  for (let i = 0; i < pellets.length; i++) {
    ctx.fillStyle = PELLET_COLOR;
    ctx.beginPath();
    ctx.arc(pellets[i].x, pellets[i].y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  for (let i = 0; i < powerPellets.length; i++) {
    ctx.fillStyle = 'green';
    ctx.beginPath();
    ctx.arc(powerPellets[i].x, powerPellets[i].y, 10, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Check for collisions with pellets
function checkPelletCollision(pellet) {
  const distance = Math.sqrt(Math.pow(player.x - pellet.x, 2) + Math.pow(player.y - pellet.y, 2));
  if (distance < player.radius + 5) {
    if (powerPellets.includes(pellet)) {
      score += 50;
      // Turn ghosts red for 5 seconds
      for (let i = 0; i < ghosts.length; i++) {
        ghosts[i].color = 'red';
      }
      setTimeout(() => {
        for (let i = 0; i < ghosts.length; i++) {
          ghosts[i].color = 'black';
        }
      }, 5000);
    } else {
      pellets.splice(pellets.indexOf(pellet), 1);
      score += 1;
    }
    return true;
  }
  return false;
}

// Check for collisions with Power Pellets
function checkPowerPelletCollision(pellet) {
  const distance = Math.sqrt(Math.pow(player.x - pellet.x, 2) + Math.pow(player.y - pellet.y, 2));
  if (distance < player.radius + 10) {
    // Turn ghosts blue for 10 seconds
    for (let i = 0; i < ghosts.length; i++) {
        if(ghosts[i].color === 'red'){
          return false;
        }
      }
    for (let i = 0; i < ghosts.length; i++) {
      ghosts[i].color = 'red';
    }
    setTimeout(() => {
      for (let i = 0; i < ghosts.length; i++) {
        ghosts[i].color = 'black';
      }
    }, 5000);
    return true;
  }
  return false;
}

// Initialize the game and start the game loop
init();
gameLoop();