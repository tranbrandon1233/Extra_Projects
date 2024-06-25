const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server, {
  cors: {
    origin: '*',
  },
});

let playersReady = 0;
let playersReadyToRestart = 0;
let playerSymbols = [];
let bothPlayersReady = false;
let board = ['', '', '', '', '', '', '', '', ''];
let currentPlayerSymbol = 'X';

io.on('connection', (socket) => {
  console.log('a new client connected');



  const symbol = playersReady === 1 ? 'X' : 'O';
  playerSymbols[socket.id] = symbol;
  playersReady++;
  playerSymbols.push(symbol);

  if (playersReady === 1) {
    io.emit('waiting');
  }

  if (playersReady === 2) {
    io.emit('stopWaiting');
    bothPlayersReady = true;
    io.emit('bothPlayersReady');
    io.emit('startGame', 'X');
  }

  socket.on('restart', () => {
    playersReady = 0;
    playerSymbols = [];
    bothPlayersReady = false;
    board = ['', '', '', '', '', '', '', '', ''];
    io.emit('resetGame');
    io.emit('waiting');
  });
  socket.on('getSymbol', (symbol) => {
    io.emit('getSymbol', symbol);
  });
  socket.on('playerReady', () => {
    playersReadyToRestart++;
    if (playersReadyToRestart === 2) {
      playersReadyToRestart = 0;
      board = ['', '', '', '', '', '', '', '', ''];
      io.emit('resetGame');
      io.emit('startGame', 'X');
    }
    else{
        io.emit('waiting');
    }
  });

  socket.on('makeMove', (newBoard) => {
    console.log(currentPlayerSymbol)
    console.log(playerSymbols[socket.id])
    console.log(socket.id)
    if (currentPlayerSymbol === playerSymbols[socket.id]) {
      board = newBoard;
      io.emit('updateBoard', newBoard);
      currentPlayerSymbol = currentPlayerSymbol === 'X' ? 'O' : 'X';
      checkForWin(newBoard);
      io.emit('switchTurn', currentPlayerSymbol);
      
    }
  });

  function checkForWin(newBoard) {
    const winConditions = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [0, 3, 6],
      [1, 4, 7],
      [2, 5, 8],
      [0, 4, 8],
      [2, 4, 6],
    ];

    for (const condition of winConditions) {
      if (
        newBoard[condition[0]] === newBoard[condition[1]] &&
        newBoard[condition[1]] === newBoard[condition[2]] &&
        newBoard[condition[0]] !== ''
      ) {
        io.emit('gameOver', newBoard[condition[0]]);
        return;
      }
    }

    if (!newBoard.includes('')) {
      io.emit('tie');
    }
  }

  socket.on('disconnect', () => {
    console.log('a client disconnected');
    playersReady--;
    playerSymbols.pop();
    bothPlayersReady = false;
    if (playersReady === 1) {
      io.emit('waiting');
    }
  });
});

server.listen(3001, () => {
  console.log('server started');
});