// Get canvas and toolbar elements
const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const toolbar = document.getElementById('toolbar');
const pen = document.getElementById('pen');
const eraser = document.getElementById('eraser');
const red = document.getElementById('red');
const black = document.getElementById('black');
const blue = document.getElementById('blue');

// Set up canvas
canvas.width = window.innerWidth * 0.8;
canvas.height = window.innerHeight * 0.8;

// Drawing variables
let drawing = false;
let tool = 'pen';
let color = 'black';
let prevX = 0;
let prevY = 0;

// Event listeners
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    prevX = e.clientX;
    prevY = e.clientY;
});

canvas.addEventListener('mousemove', (e) => {
    if(drawing) {
        const x = e.clientX;
        const y = e.clientY;
        draw(x, y);
        prevX = x;
        prevY = y;
    }
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
});

canvas.addEventListener('mouseleave', () => {
    drawing = false;
});

// Toolbar event listeners
pen.addEventListener('click', () => {
    tool = 'pen';
});

eraser.addEventListener('click', () => {
    tool = 'eraser';
});

red.addEventListener('click', () => {
    color = 'red';
});

black.addEventListener('click', () => {
    color = 'black';
});

blue.addEventListener('click', () => {
    color = 'blue';
});

// Drawing function
function draw(x, y) {
    if(tool === 'pen') {
        ctx.strokeStyle = color;
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(x, y);
        ctx.stroke();
    } else if(tool === 'eraser') {
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(x, y);
        ctx.stroke();
    }
}