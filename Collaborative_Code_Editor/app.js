const express = require('express');
const app = express();
const http = require('http').Server(app);
const io = require('socket.io')(http);
const { spawn } = require('child_process');

app.use(express.static('public'));

let code = ''; // Store the current code
let language = 'python'; // Store the current language
let userSockets = {}; // Store user sockets

io.on('connection', (socket) => {
  console.log('A user connected');

  // Send current code to newly connected user
  socket.emit('init', code);

  // Handle code changes
  socket.on('codeChange', (newCode) => {
    code = newCode;
    socket.broadcast.emit('updateCode', newCode);
  });

  // Handle language changes
  socket.on('languageChange', (newLanguage) => {
    language = newLanguage;
    socket.broadcast.emit('updateLanguage', newLanguage);
  });

  // Handle run code
  socket.on('runCode', (userCode) => {
    let command;
    let args;
    if (language === 'python') {
      command = 'python';
      args = ['-c', userCode];
    } else if (language === 'cpp') {
      command = 'g++';
      args = ['-o', 'output', '-x', 'c++', '-'];
    }

    const compileProcess = spawn(command, args);
    let output = '';

    compileProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    compileProcess.stderr.on('data', (data) => {
      output += data.toString();
    });

    compileProcess.on('close', () => {
      if (language === 'cpp') {
        const runProcess = spawn('./output');
        runProcess.stdout.on('data', (data) => {
          output += data.toString();
        });

        runProcess.stderr.on('data', (data) => {
          output += data.toString();
        });

        runProcess.on('close', () => {
          socket.emit('runOutput', output); // Send output to the user who executed the code
        });
      } else {
        socket.emit('runOutput', output); // Send output to the user who executed the code
      }
    });
  });

  // Store user socket
  socket.on('registerUser', (userId) => {
    userSockets[userId] = socket;
  });

  socket.on('disconnect', () => {
    console.log('User disconnected');
  });
});

const PORT = process.env.PORT || 3000;
http.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});