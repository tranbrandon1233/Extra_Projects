const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);
const fs = require('fs');
const path = require('path');

app.use(express.static('public'));

let videoBuffer = Buffer.alloc(0);

io.on('connection', (socket) => {
    console.log('Client connected');
    socket.on('video', (chunk) => {
        videoBuffer = Buffer.concat([videoBuffer, chunk]);
    });
    socket.on('end', () => {
        const videoPath = './uploads/video.mp4';
        fs.writeFileSync(videoPath, videoBuffer);
        io.emit('videoReady');
    });
});

app.get('/play', (req, res) => {
    res.sendFile(__dirname + '/public/play.html');
});

app.get('/video', (req, res) => {
    const videoPath = './uploads/video.mp4';
    res.sendFile(path.join(__dirname, videoPath));
});

server.listen(8001, () => {
    console.log('Streaming server listening on port 8001');
});