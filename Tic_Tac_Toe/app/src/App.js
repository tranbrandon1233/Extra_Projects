import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';

const socket = io.connect('http://localhost:3001');

function App() {
  const [playerSymbol, setPlayerSymbol] = useState(null);
  const [gameStarted, setGameStarted] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState(null);
  const [board, setBoard] = useState(['', '', '', '', '', '', '', '', '']);
  const [turn, setTurn] = useState(null);
  const [myTurn, setMyTurn] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [bothPlayersReady, setBothPlayersReady] = useState(false);
  const [player1Wins, setPlayer1Wins] = useState(0);
  const [player2Wins, setPlayer2Wins] = useState(0);

  useEffect(() => {
    // Initialize the game when the component mounts
    const symbols = ['X', 'O'];
    const randomSymbol = symbols[Math.floor(Math.random() * symbols.length)];
    setPlayerSymbol(randomSymbol);

    // Handle spacebar press to restart the game

    const handleKeyPress = (event) => {
      if (event.key === ' ') {
        if (gameOver) {
          socket.emit('playerReady');
        }
      }
    };

    document.addEventListener('keydown', handleKeyPress);

    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [gameOver]);

  useEffect(() => {
    socket.on('resetGame', () => {
      setGameStarted(false);
      setGameOver(false);
      setWinner(null);
      setBoard(['', '', '', '', '', '', '', '', '']);
      setTurn(null);
      setMyTurn(false);
      setWaiting(true);
    });

    socket.on('bothPlayersReady', () => {
      setBothPlayersReady(true);
      setWaiting(false);
    });

    socket.on('startGame', (symbol) => {
      setGameStarted(true);
      setGameOver(false);
      setTurn(symbol);
      setMyTurn(symbol === playerSymbol);
    });
    
    socket.on('switchTurn', (newTurn) => {
      setTurn(newTurn);
      setMyTurn(newTurn === playerSymbol);
    });

    socket.on('updateBoard', (newBoard) => {
      setBoard(newBoard);
    }); 

    socket.on('gameOver', (winner) => {
      setGameOver(true);
      setWinner(winner);
      if (winner === 'X') {
        setPlayer1Wins(player1Wins + 1);
      } else {
        setPlayer2Wins(player2Wins + 1);
      }
    });
    
    socket.on('stopWaiting', () => {
      setWaiting(false);
    });

    socket.on('tie', () => {
      setGameOver(true);
      setWinner(null);
    });


    socket.on('waiting', () => {
      setWaiting(true);
    });
  }, [playerSymbol, player1Wins, player2Wins]);

const handleCellClick = (index) => {
  console.log(playerSymbol)
  setMyTurn(playerSymbol !== turn)
  if (gameOver || !gameStarted || !myTurn || board[index] !== '' ) return;
  const newBoard = [...board];
  newBoard[index] = playerSymbol;
  socket.emit('makeMove', newBoard);

};
  
return (
    <div className="app">
      {bothPlayersReady && gameStarted && !gameOver ? (
        <div className="wins">
          <div>Player 1 Wins: {player1Wins}</div>
          <div>Player 2 Wins: {player2Wins}</div>
        </div>
      ) : null}

    {bothPlayersReady && gameStarted ? (
      gameOver ? (
        winner === playerSymbol ? (
          <div className="win-screen">You Win!</div>
        ) : winner === null ? (
          <div className="tie-screen">You Tied!</div>
        ) : (
          <div className="lose-screen">You Lose!</div>
        )
      ) : (
        <div className="game-board">
          {board.map((cell, index) => (
            <div
              key={index}
              className="cell"
              onClick={() => handleCellClick(index)}
            >
              {cell}
            </div>
          ))}
        </div>
      )
    ) : (
      <div className="waiting-screen">Waiting for other player...</div>
    )}
  </div>
);
}

export default App;