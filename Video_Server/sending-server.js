// Import required modules
const express = require('express');
const multer = require('multer');
const app = express();
const upload = multer({ dest: './uploads/' }); // Set up Multer to store uploaded files in the './uploads/' directory
const server = require('http').createServer(app);
const io = require('socket.io')(server); // Set up Socket.IO for real-time communication

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Handle video uploads
app.post('/upload', upload.single('video'), (req, res) => {
    // Get the path of the uploaded file
    const filePath = req.file.path;
    
    // Create a read stream to read the file in chunks
    const readStream = require('fs').createReadStream(filePath);
    
    // Connect to the server on port 8001 as a client
    const socket = require('socket.io-client')('http://localhost:8001');
    
    // Send the video chunks to the server on port 8001
    readStream.on('data', (chunk) => {
        socket.emit('video', chunk);
    });
    
    // Send an 'end' signal when the video upload is complete
    readStream.on('end', () => {
        socket.emit('end');
        res.send('Video uploaded successfully');
    });
});

// Handle incoming connections from clients
io.on('connection', (socket) => {
    // Relay comments from the server on port 8001 to all connected clients
    socket.on('comment', (comment) => {
        io.emit('comment', comment);
    });
});

// Start the server on port 8000
server.listen(8000, () => {
    console.log('Upload server listening on port 8000');
});