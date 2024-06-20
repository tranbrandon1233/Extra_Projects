const express = require('express');
const multer = require('multer');
const app = express();
const upload = multer({ dest: './uploads/' });
const server = require('http').createServer(app);
const io = require('socket.io')(server);

app.use(express.static('public'));

app.post('/upload', upload.single('video'), (req, res) => {
    const filePath = req.file.path;
    const readStream = require('fs').createReadStream(filePath);
    const socket = require('socket.io-client')('http://localhost:8001');
    readStream.on('data', (chunk) => {
        socket.emit('video', chunk);
    });
    readStream.on('end', () => {
        socket.emit('end');
        res.send('Video uploaded successfully');
    });
});

io.on('connection', (socket) => {
    socket.on('comment', (comment) => {
        // Broadcast the comment to all connected clients
        io.emit('comment', comment);
    });
});

server.listen(8000, () => {
    console.log('Upload server listening on port 8000');
});