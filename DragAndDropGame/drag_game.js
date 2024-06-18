// Get the canvas element
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');

// Set the canvas dimensions
canvas.width = 800;
canvas.height = 600;

// Define the colors
var colors = ['red', 'green', 'blue', 'yellow'];

// Define the squares at the bottom of the screen
var squares = [
  { x: 100, y: canvas.height - 50, color: 'red' },
  { x: 300, y: canvas.height - 50, color: 'green' },
  { x: 500, y: canvas.height - 50, color: 'blue' },
  { x: 700, y: canvas.height - 50, color: 'yellow' }
];

// Define the falling squares
var fallingSquares = [];

// Define the score
var score = 0;

// Define the time
var time = 20;

// Define the interval for creating new falling squares
setInterval(function() {
  // Create a new falling square
  var fallingSquare = {
    x: Math.random() * (canvas.width - 50),
    y: 0,
    color: colors[Math.floor(Math.random() * colors.length)],
    dragging: false
  };

  // Add the falling square to the array
  fallingSquares.push(fallingSquare);
}, 1000);

// Define the mouse events
var mouseDown = false;
var mousePosition = { x: 0, y: 0 };

canvas.addEventListener('mousedown', function(event) {
  mouseDown = true;
  mousePosition = { x: event.clientX, y: event.clientY };

  // Check if the user is clicking on a falling square
  for (var i = 0; i < fallingSquares.length; i++) {
    var fallingSquare = fallingSquares[i];
    if (mousePosition.x > fallingSquare.x && mousePosition.x < fallingSquare.x + 50 && mousePosition.y > fallingSquare.y && mousePosition.y < fallingSquare.y + 50) {
      fallingSquare.dragging = true;
    }
  }
});

canvas.addEventListener('mousemove', function(event) {
  if (mouseDown) {
    mousePosition = { x: event.clientX, y: event.clientY };

    // Update the position of the falling square being dragged
    for (var i = 0; i < fallingSquares.length; i++) {
      var fallingSquare = fallingSquares[i];
      if (fallingSquare.dragging) {
        fallingSquare.x = mousePosition.x - 25;
        fallingSquare.y = mousePosition.y - 25;
      }
    }
  }
});

canvas.addEventListener('mouseup', function(event) {
  mouseDown = false;

  // Check if the user dropped a falling square on a square at the bottom of the screen
  for (var i = 0; i < fallingSquares.length; i++) {
    var fallingSquare = fallingSquares[i];
    if (fallingSquare.dragging) {
      fallingSquare.dragging = false;

      // Check if the falling square is on top of a square at the bottom of the screen
      for (var j = 0; j < squares.length; j++) {
        var square = squares[j];
        if (fallingSquare.x > square.x - 25 && fallingSquare.x < square.x + 75 && fallingSquare.y > square.y - 25 && fallingSquare.y < square.y + 75) {
          // Check if the colors match
          if (fallingSquare.color === square.color) {
            // Increase the score
            score++;

            // Remove the falling square from the array
            fallingSquares.splice(i, 1);
          } else {
            // Decrease the score
            score--;
          }
        }
      }
    }
  }
});

// ...

// Main loop
function update() {
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Check if the game is over
  if (time <= 0) {
    // Draw the final score
    ctx.fillStyle = 'black';
    ctx.font = '48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Game Over! Your final score is: ' + score, canvas.width / 2, canvas.height / 2);
  } else {
    // Draw the squares at the bottom of the screen
    for (var i = 0; i < squares.length; i++) {
      var square = squares[i];
      ctx.fillStyle = square.color;
      ctx.fillRect(square.x, square.y, 50, 50);
    }

    // Update and draw the falling squares
    for (var i = 0; i < fallingSquares.length; i++) {
      var fallingSquare = fallingSquares[i];

      // Update the position of the falling square
      if (!fallingSquare.dragging) {
        fallingSquare.y += 2;

        // Check if the falling square has reached the bottom of the screen
        if (fallingSquare.y > canvas.height) {
          // Decrease the score
          score--;

          // Remove the falling square from the array
          fallingSquares.splice(i, 1);
        }
      }

      // Draw the falling square
      ctx.fillStyle = fallingSquare.color;
      ctx.fillRect(fallingSquare.x, fallingSquare.y, 50, 50);
    }

    // Update the time
    time -= 1 / 60;

    // Draw the score
    ctx.fillStyle = 'black';
    ctx.font = '24px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('Score: ' + score, canvas.width - 10, 10);

    // Draw the time
    ctx.fillStyle = 'black';
    ctx.font = '24px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Time: ' + Math.floor(time), 10, 10);

    // Request the next frame
    requestAnimationFrame(update);
  }
}

// Start the main loop
update();