// Get the canvas element
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set the canvas dimensions
canvas.width = 640;
canvas.height = 480;

// Load the bird image
const birdImage = new Image();
birdImage.src = 'assets/bird.png';
birdImage.width = 30;
birdImage.height = 30;

// Set the bird's initial position and velocity
let birdX = 50;
let birdY = canvas.height / 2;
let birdVelocity = 0;

// Set the pipe's initial position and gap size
let pipeX = canvas.width;
let pipeGap = 150;
let pipeTopHeight = Math.random() * (canvas.height - pipeGap);

// Set the score
let score = 0;

// Set the game over flag
let gameOver = false;

// Set the flag to check if the bird has passed the current pipe
let hasPassedPipe = false;

// Set the timer
let timer = 15;

// Handle mouse clicks
canvas.addEventListener('click', () => {
  birdVelocity = -10;
});

// Main game loop
function update() {
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the bird
  ctx.drawImage(birdImage, birdX, birdY, birdImage.width, birdImage.height);

  // Update the bird's position
  birdY += birdVelocity;
  birdVelocity += 0.5;

  // Check for collision with the ground
  if (birdY + birdImage.height > canvas.height) {
    gameOver = true;
  }

  // Draw the pipe
  ctx.fillStyle = 'green';
  ctx.fillRect(pipeX, 0, 50, pipeTopHeight);
  ctx.fillRect(pipeX, pipeTopHeight + pipeGap, 50, canvas.height - (pipeTopHeight + pipeGap));

  // Update the pipe's position
  pipeX -= 5;

  // Check for collision with the pipe
  if (pipeX < birdX + birdImage.width &&
      pipeX + 50 > birdX &&
      (birdY < pipeTopHeight || birdY + birdImage.height > pipeTopHeight + pipeGap)) {
    gameOver = true;
  }

  // Check if the bird has passed the pipe
  if (pipeX < birdX && !hasPassedPipe) {
    score++;
    hasPassedPipe = true;
  }

  // Check if the pipe is off the screen
  if (pipeX < -50) {
    pipeX = canvas.width;
    pipeTopHeight = Math.random() * (canvas.height - pipeGap);
    hasPassedPipe = false;
  }

  // Draw the score
  ctx.font = '24px Arial';
  ctx.fillStyle = 'black';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(`Score: ${score}`, 10, 10);

  // Draw the timer
  ctx.font = '24px Arial';
  ctx.fillStyle = 'black';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(`Time: ${timer.toFixed(1)}`, canvas.width - 10, 10);

  // Update the timer
  timer -= 1/60;
  if (timer <= 0) {
    gameOver = true;
  }

  // Check if the game is over
  if (gameOver) {
    ctx.font = '48px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`Game Over! Final Score: ${score}`, canvas.width / 2, canvas.height / 2);
  } else {
    // Request the next frame
    requestAnimationFrame(update);
  }
}

// Start the game loop
update();