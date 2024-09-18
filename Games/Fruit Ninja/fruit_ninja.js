var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

var score = 0;
var time = 15;

var squares = [];

function getRandomColor() {
    return Math.random() < 0.5 ? 'green' : 'red';
}

function getRandomX() {
    return Math.random() * (canvas.width - 50);
}

function getRandomY() {
    return Math.random() * (canvas.height * 0.8);
}

function createSquare() {
    var square = {
        x: getRandomX(),
        y: canvas.height,
        color: getRandomColor(),
        speed: Math.random() * 5 + 2,
        direction: -1,
        maxHeight: getRandomY()
    };
    squares.push(square);
}

function drawSquares() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < squares.length; i++) {
        ctx.fillStyle = squares[i].color;
        ctx.fillRect(squares[i].x, squares[i].y, 50, 50);
    }
}

// Function to update the squares
function updateSquares() {
    for (var i = 0; i < squares.length; i++) {
        squares[i].y += squares[i].speed * squares[i].direction;
        if (squares[i].y < squares[i].maxHeight) {
            squares[i].direction = 1;
        }
        if (squares[i].y > canvas.height) {
            squares.splice(i, 1);
        }
    }
}

// Function to handle mouse clicks
function handleClick(event) {
    for (var i = 0; i < squares.length; i++) {
        if (event.clientX > squares[i].x && event.clientX < squares[i].x + 50 && event.clientY > squares[i].y && event.clientY < squares[i].y + 50) {
            if (squares[i].color == 'green') {
                score++;
            } else {
                score--;
            }
            squares.splice(i, 1);
        }
    }
}

// Function to draw the score
function drawScore() {
    ctx.font = '24px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('Score: ' + score, canvas.width - 20, 10);
}

// Function to draw the time
function drawTime() {
    ctx.font = '24px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Time: ' + time, 10, 10);
}

// Function to draw the game over screen
function drawGameOver() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '48px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Game Over', canvas.width / 2, canvas.height / 2);
    ctx.font = '36px Arial';
    ctx.fillText('Final Score: ' + score, canvas.width / 2, canvas.height / 2 + 50);
}

// Main game loop
function gameLoop() {
    if (time > 0) {
        if (Math.random() < 0.05) {
            createSquare();
        }
        updateSquares();
        drawSquares();
        drawScore();
        drawTime();
        time -= 1 / 60;
        requestAnimationFrame(gameLoop);
    } else {
        drawGameOver();
    }
}

canvas.addEventListener('click', handleClick);

// Start the game loop
gameLoop();