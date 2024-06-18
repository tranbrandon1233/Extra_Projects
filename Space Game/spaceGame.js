let canvas = document.getElementById('gameCanvas');
let ctx = canvas.getContext('2d');

let spaceship = {
    x: canvas.width / 2,
    y: canvas.height / 2,
    speed: 2,
    radius: 20
};

let asteroids = [];

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
        asteroids[i].y += asteroids[i].speed;
        if (asteroids[i].y > canvas.height) {
            asteroids.splice(i, 1);
            i--;
        }
    }

    // Check collision with spaceship
    for (let i = 0; i < asteroids.length; i++) {
        if (Math.pow(asteroids[i].x - spaceship.x, 2) + Math.pow(asteroids[i].y - spaceship.y, 2) < Math.pow(asteroids[i].radius + spaceship.radius, 2)) {
            alert('Game Over');
            return;
        }
    }

    // Add new asteroids
    if (asteroids.length < 12) {
        let numberOfAsteroidsToAdd = 12 - asteroids.length;
        for (let i = 0; i < numberOfAsteroidsToAdd; i++) {
            asteroids.push({
                x: Math.random() * canvas.width,
                y: 0,
                speed: Math.random() * 2 + 1,
                radius: Math.random() * 20 + 10
            });
        }
    }

    requestAnimationFrame(draw);
}

canvas.addEventListener('mousemove', function(e) {
    spaceship.x = e.clientX;
    spaceship.y = e.clientY;
});

setInterval(function() {
    requestAnimationFrame(draw);
}, 3000);

requestAnimationFrame(draw);