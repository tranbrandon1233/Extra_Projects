// Get the canvas element
let canvas = document.getElementById('gameCanvas');
let ctx = canvas.getContext('2d');

// Set canvas dimensions to match the window dimensions
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Define the spaceship object
let spaceship = {
    x: canvas.width / 2,
    y: canvas.height / 2,
    speed: 2,
    radius: 20
};

// Define the asteroids array
let asteroids = [];

// Function to draw the game elements
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw spaceship
    ctx.beginPath();
    ctx.arc(spaceship.x, spaceship.y, spaceship.radius, 0, 2 * Math.PI);
    ctx.fillStyle = 'blue';
    ctx.fill();

    // Draw asteroids
    for (let i = 0; i < asteroids.length; i++) {
        ctx.beginPath();
        ctx.arc(asteroids[i].x, asteroids[i].y, asteroids[i].radius, 0, 2 * Math.PI);
        ctx.fillStyle = 'green';
        ctx.fill();
    }

// Update asteroids
for (let i = 0; i < asteroids.length; i++) {
    switch (asteroids[i].direction) {
        case 'down':
            asteroids[i].y += asteroids[i].speed;
            if (asteroids[i].y > canvas.height) {
                asteroids.splice(i, 1);
                i--;
            }
            break;
        case 'left':
            asteroids[i].x -= asteroids[i].speed;
            if (asteroids[i].x < 0) {
                asteroids.splice(i, 1);
                i--;
            }
            break;
        case 'up':
            asteroids[i].y -= asteroids[i].speed;
            if (asteroids[i].y < 0) {
                asteroids.splice(i, 1);
                i--;
            }
            break;
        case 'right':
            asteroids[i].x += asteroids[i].speed;
            if (asteroids[i].x > canvas.width) {
                asteroids.splice(i, 1);
                i--;
            }
            break;
    }
}

    // Check collision with spaceship
    for (let i = 0; i < asteroids.length; i++) {
        if (Math.pow(asteroids[i].x - spaceship.x, 2) + Math.pow(asteroids[i].y - spaceship.y, 2) < Math.pow(asteroids[i].radius + spaceship.radius, 2)) {
            alert('Game Over');
            // Reset game state
            resetGame();
            return;
        }
    }

    // Add new asteroids
if (asteroids.length < 12) {
    let numberOfAsteroidsToAdd = 12 - asteroids.length;
    for (let i = 0; i < numberOfAsteroidsToAdd; i++) {
        let direction = Math.floor(Math.random() * 4); // 0: top, 1: right, 2: bottom, 3: left
        let asteroid;
        switch (direction) {
            case 0: // top
                asteroid = {
                    x: Math.random() * canvas.width,
                    y: 0,
                    speed: Math.random() * 2 + 1,
                    radius: Math.random() * 20 + 10,
                    direction: 'down'
                };
                break;
            case 1: // right
                asteroid = {
                    x: canvas.width,
                    y: Math.random() * canvas.height,
                    speed: Math.random() * 2 + 1,
                    radius: Math.random() * 20 + 10,
                    direction: 'left'
                };
                break;
            case 2: // bottom
                asteroid = {
                    x: Math.random() * canvas.width,
                    y: canvas.height,
                    speed: Math.random() * 2 + 1,
                    radius: Math.random() * 20 + 10,
                    direction: 'up'
                };
                break;
            case 3: // left
                asteroid = {
                    x: 0,
                    y: Math.random() * canvas.height,
                    speed: Math.random() * 2 + 1,
                    radius: Math.random() * 20 + 10,
                    direction: 'right'
                };
                break;
        }
        asteroids.push(asteroid);
    }
}

    requestAnimationFrame(draw);
}

// Function to reset the game state
function resetGame() {
    // Reset spaceship position
    spaceship.x = canvas.width / 2;
    spaceship.y = canvas.height / 2;

    // Clear asteroids array
    asteroids = [];

    // Restart the game
    draw();
}

// Add event listener for mouse movement
canvas.addEventListener('mousemove', function(e) {
    let rect = canvas.getBoundingClientRect();
    spaceship.x = e.clientX - rect.left - spaceship.radius / 2;
    spaceship.y = e.clientY - rect.top - spaceship.radius / 2;
});

// Start the game
requestAnimationFrame(draw);