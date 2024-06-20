// Import required modules
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server); // Set up Socket.IO for real-time communication
const fs = require('fs'); // File system module
const path = require('path'); // Path module

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Serve uploaded videos from the 'uploads' directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Initialize a buffer to store the video chunks
let videoBuffer = Buffer.alloc(0);

// Handle incoming connections from clients
io.on('connection', (socket) => {
    console.log('Client connected');
    
    // Receive video chunks from the server on port 8000
    socket.on('video', (chunk) => {
        // Append the chunk to the video buffer
        videoBuffer = Buffer.concat([videoBuffer, chunk]);
    });
    
    // Receive the 'end' signal from the server on port 8000
    socket.on('end', () => {
        // Write the video buffer to a file
        const videoPath = './uploads/video.mp4';
        fs.writeFileSync(videoPath, videoBuffer);
        
        // Notify all connected clients that the video is ready
        io.emit('videoReady', '/uploads/video.mp4');
    });
    
    // Relay comments from clients to the server on port 8000
    socket.on('comment', (comment) => {
        // Connect to the server on port 8000 as a client
        const socketToUploadServer = require('socket.io-client')('http://localhost:8000');
        
        // Send the comment to the server on port 8000
        socketToUploadServer.emit('comment', comment);
    });
});

// Serve the video player page
app.get('/play', (req, res) => {
    res.sendFile(__dirname + '/public/play.html');
});

// Start the server on port 8001
server.listen(8001, () => {
    console.log('Streaming server listening on port 8001');
});