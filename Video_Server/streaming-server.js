const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);
const fs = require('fs');
const path = require('path');

app.use(express.static('public'));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

let videoBuffer = Buffer.alloc(0);

io.on('connection', (socket) => {
    console.log('Client connected');
    socket.on('video', (chunk) => {
        videoBuffer = Buffer.concat([videoBuffer, chunk]);
    });
    socket.on('end', () => {
        const videoPath = './uploads/video.mp4';
        fs.writeFileSync(videoPath, videoBuffer);
        io.emit('videoReady', '/uploads/video.mp4');
    });
    socket.on('comment', (comment) => {
        // Send the comment to the server on port 8000
        const socketToUploadServer = require('socket.io-client')('http://localhost:8000');
        socketToUploadServer.emit('comment', comment);
    });
});

app.get('/play', (req, res) => {
    res.sendFile(__dirname + '/public/play.html');
});

server.listen(8001, () => {
    console.log('Streaming server listening on port 8001');
});