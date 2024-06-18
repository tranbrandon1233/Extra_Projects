// Get the canvas element
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set the canvas dimensions
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Define the columns
const columns = [
  { x: canvas.width / 8, color: 'red', key: 'A', keyCode: 65 },
  { x: canvas.width / 8 * 3, color: 'blue', key: 'S', keyCode: 83 },
  { x: canvas.width / 8 * 5, color: 'green', key: 'D', keyCode: 68 },
  { x: canvas.width / 8 * 7, color: 'yellow', key: 'F', keyCode: 70 }
];

// Define the rectangles
let rectangles = [];

// Define the points
let points = 0;

// Define the key pressed status
const keyPressed = {};

// Add event listeners for key press and release
document.addEventListener('keydown', (e) => {
  keyPressed[e.keyCode] = true;
});

document.addEventListener('keyup', (e) => {
  keyPressed[e.keyCode] = false;
});

// Add event listener for window resize
window.addEventListener('resize', () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

// Function to generate new rectangles
function generateRectangle(column) {
  const rectangle = {
    x: column.x,
    y: 0,
    width: canvas.width / 8,
    height: Math.random() * 200 + 50,
    color: column.color,
    key: column.key,
    keyCode: column.keyCode
  };
  rectangles.push(rectangle);
  setTimeout(() => generateRectangle(column), Math.random() * 2500 + 500);
}

// Start generating rectangles for each column
columns.forEach((column) => {
  generateRectangle(column);
});

// Function to update the game state
let lastTime = Date.now();
let timer = 15;
function update() {
  const currentTime = Date.now();
  const deltaTime = (currentTime - lastTime) / 1000;
  lastTime = currentTime;

  // Update rectangle positions
  rectangles.forEach((rectangle) => {
    rectangle.y += 200 * deltaTime;
  });

  // Remove rectangles that are off the screen
  rectangles = rectangles.filter((rectangle) => rectangle.y < canvas.height + rectangle.height);

  // Update points
  columns.forEach((column) => {
    const rectangle = rectangles.find((rectangle) => rectangle.x === column.x && rectangle.y + rectangle.height > canvas.height - 50);
    if (rectangle && keyPressed[rectangle.keyCode]) {
      points += deltaTime;
    } else if (keyPressed[column.keyCode]) {
      points -= deltaTime;
    }
  });

  // Update timer
  timer -= deltaTime;
  if (timer <= 0) {
    timer = 0;
    rectangles = [];
  }
}

// Function to draw the game
function draw() {
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the columns
  columns.forEach((column) => {
    ctx.fillStyle = 'white';
    ctx.fillRect(column.x, canvas.height - 50, canvas.width / 8, 50);
    ctx.fillStyle = 'black';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(column.key, column.x + canvas.width / 16, canvas.height - 20);
  });

  // Draw the rectangles
  rectangles.forEach((rectangle) => {
    ctx.fillStyle = rectangle.color;
    ctx.fillRect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
  });

  // Draw the points
  ctx.fillStyle = 'black';
  ctx.font = '24px Arial';
  ctx.textAlign = 'left';
  ctx.fillText(`Points: ${Math.floor(points)}`, 20, 40);

  // Draw the timer
  ctx.textAlign = 'right';
  ctx.fillText(`Time: ${Math.floor(timer)}`, canvas.width - 20, 40);

  // Draw game over screen
  if (timer <= 0) {
    ctx.textAlign = 'center';
    ctx.font = '48px Arial';
    ctx.fillText('Game Over', canvas.width / 2, canvas.height / 2);
    ctx.font = '24px Arial';
    ctx.fillText(`Final Points: ${Math.floor(points)}`, canvas.width / 2, canvas.height / 2 + 50);
  }
}

// Main game loop
function loop() {
  update();
  draw();
  requestAnimationFrame(loop);
}

// Start the game
loop();