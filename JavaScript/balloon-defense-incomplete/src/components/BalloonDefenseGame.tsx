import React, { useEffect, useRef, useState } from 'react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';

interface Balloon {
  x: number;
  y: number;
  health: number;
  pathProgress: number;
}

interface Shooter {
  x: number;
  y: number;
  lastShot: number;
}

interface Projectile {
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  speed: number;
}

const pathPoints = [
  { x: 800, y: 100 },
  { x: 600, y: 100 },
  { x: 500, y: 200 },
  { x: 400, y: 300 },
  { x: 300, y: 200 },
  { x: 200, y: 100 },
  { x: 100, y: 200 },
  { x: 0, y: 200 },
];

const ANIMATION_INTERVAL = 20; // 20ms = 0.02 seconds
const MOVEMENT_SPEED = 0.02;

const BalloonDefenseGame = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastUpdateTimeRef = useRef<number>(0);
  const [gameState, setGameState] = useState({
    coins: 50,
    hp: 5,
    round: 1,
    isPlaying: false,
    gameOver: false
  });
  const [shooters, setShooters] = useState<Shooter[]>([]);
  const [balloons, setBalloons] = useState<Balloon[]>([]);
  const [projectiles, setProjectiles] = useState<(Projectile | null)[]>([]);
  const [placingShooter, setPlacingShooter] = useState(false);

  useEffect(() => {
    let animationId: number;
    
    const gameLoop = (timestamp: number) => {
      if (gameState.isPlaying && !gameState.gameOver) {
        const deltaTime = timestamp - lastUpdateTimeRef.current;
        
        if (deltaTime >= ANIMATION_INTERVAL) {
          updateGame();
          lastUpdateTimeRef.current = timestamp;
        }
      }
      animationId = requestAnimationFrame(gameLoop);
    };
    
    animationId = requestAnimationFrame(gameLoop);
    return () => cancelAnimationFrame(animationId);
  }, [gameState.isPlaying, gameState.gameOver]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    canvas.width = 800;
    canvas.height = 400;
    drawGame();
  }, []);

  useEffect(() => {
    drawGame();
  }, [shooters, balloons, projectiles, gameState]);

  const isPointOnPath = (x: number, y: number): boolean => {
    for (let i = 0; i < pathPoints.length - 1; i++) {
      const p1 = pathPoints[i];
      const p2 = pathPoints[i + 1];
      const distance = Math.abs(
        (p2.y - p1.y) * x - (p2.x - p1.x) * y + p2.x * p1.y - p2.y * p1.x
      ) / Math.sqrt((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2);
      
      if (distance < 30) return true;
    }
    return false;
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (placingShooter && gameState.coins >= 10) {
      const canvas = canvasRef.current;
      if (!canvas) return;
  
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
  
      if (!isPointOnPath(x, y)) {
        setShooters([...shooters, { x, y, lastShot: 0 }]);
        setGameState({ ...gameState, coins: gameState.coins - 10 });
      }
    }
  };
  
  const handleBuyShooterClick = () => {
    setPlacingShooter(!placingShooter);
  };

  const startRound = () => { 
    const numBalloons = gameState.round * 5;
    const newBalloons: Balloon[] = Array(numBalloons).fill(null).map((_, i) => ({
      x: pathPoints[0].x,
      y: pathPoints[0].y,
      health: 3,
      pathProgress: -i * 50 
    }));
  
    setBalloons(newBalloons);
    setGameState(prevState => ({ ...prevState, isPlaying: true }));
    gameState.isPlaying = true;
    console.log(gameState)
  };
  
  useEffect(() => {
    if (gameState.isPlaying) {
      lastUpdateTimeRef.current = performance.now();
    }
  }, [gameState.isPlaying]);
  

const getPositionOnPath = (progress: number) => {
  const segmentCount = pathPoints.length - 1;
  const segmentIndex = Math.floor(progress);
  
  if (segmentIndex >= segmentCount) {
    return pathPoints[segmentCount];
  }

  const start = pathPoints[segmentIndex];
  const end = pathPoints[segmentIndex + 1];
  const segmentProgress = progress - segmentIndex; 

  return {
    x: start.x + (end.x - start.x) * segmentProgress,
    y: start.y + (end.y - start.y) * segmentProgress
  };
};

const updateGame = () => {
  const updatedBalloons = balloons.map(balloon => {
    const newProgress = balloon.pathProgress + MOVEMENT_SPEED;
    const position = getPositionOnPath(newProgress);
    const vx = (position.x - balloon.x) * MOVEMENT_SPEED;
    const vy = (position.y - balloon.y) * MOVEMENT_SPEED;
    console.log(vx, vy);
    return {
      ...balloon,
      x: position.x+vx,
      y: position.y+vy,
      pathProgress: newProgress
    };
  });
  console.log(updatedBalloons)

  const updatedProjectiles = projectiles.map(projectile => {
    if (!projectile) return null;

    const dx = projectile.targetX - projectile.x;
    const dy = projectile.targetY - projectile.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance < projectile.speed) {
      return null;
    }
    
    const vx = (dx / distance) * projectile.speed;
    const vy = (dy / distance) * projectile.speed;
    
    return {
      ...projectile,
      x: projectile.x + vx,
      y: projectile.y + vy
    };
  });

  const currentTime = Date.now();
  shooters.forEach(shooter => {
    if (currentTime - shooter.lastShot > 1000) { 
      const nearestBalloon = updatedBalloons.find(balloon => 
        Math.sqrt((balloon.x - shooter.x) ** 2 + (balloon.y - shooter.y) ** 2) < 150
      );
      
      if (nearestBalloon) {
        updatedProjectiles.push({
          x: shooter.x,
          y: shooter.y,
          targetX: nearestBalloon.x,
          targetY: nearestBalloon.y,
          speed: 5
        });
        shooter.lastShot = currentTime;
      }
    }
  });

  // Collision detection
  updatedBalloons.forEach(balloon => {
    updatedProjectiles.forEach((projectile, index) => {
      if (projectile && Math.sqrt((balloon.x - projectile.x) ** 2 + (balloon.y - projectile.y) ** 2) < 20) {
        balloon.health--;
        updatedProjectiles[index] = null;
      }
    });
  });

  const remainingBalloons = updatedBalloons.filter(balloon => {
    if (balloon.pathProgress >= pathPoints.length - 1) {
      setGameState(prev => ({
        ...prev,
        hp: prev.hp - 1
      }));
      return false;
    }
    return balloon.health > 0;
  });

  if (remainingBalloons.length === 0 && gameState.isPlaying) {
    setGameState(prev => ({
      ...prev,
      isPlaying: false,
      coins: prev.coins + prev.round * 20,
      round: prev.round + 1
    }));
  }

  if (gameState.hp <= 0) {
    setGameState(prev => ({ ...prev, gameOver: true }));
  }
  setBalloons(remainingBalloons);
  setProjectiles(updatedProjectiles.filter(Boolean) as Projectile[]);
};


  const drawGame = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
  
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  
    ctx.beginPath();
    ctx.moveTo(pathPoints[0].x, pathPoints[0].y);
    pathPoints.forEach(point => {
      ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 40;
    ctx.stroke();
  
    shooters.forEach(shooter => {
      ctx.beginPath();
      ctx.arc(shooter.x, shooter.y, 15, 0, Math.PI * 2);
      ctx.fillStyle = '#00f';
      ctx.fill();
    });
  
    balloons.forEach(balloon => {
      ctx.beginPath();
      ctx.arc(balloon.x, balloon.y, 20, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(255, ${85 * balloon.health}, 0)`;
      ctx.fill();
    });
  
    projectiles.forEach(projectile => {
      if (projectile) {
        ctx.beginPath();
        ctx.arc(projectile.x, projectile.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#000';
        ctx.fill();
      }
    });
  };

  return (
    <Card className="max-w-4xl mx-auto">
      <div className="flex flex-col items-center gap-4">
        <div className="flex gap-4 mb-4">
          <div>Coins: {gameState.coins}</div>
          <div>HP: {gameState.hp}</div>
          <div>Round: {gameState.round}</div>
        </div>
        
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="border border-gray-300 cursor-pointer"
        />
        
        <div className="flex gap-4">
          <Button
            className='text'
            onClick={handleBuyShooterClick}
            disabled={gameState.coins < 10 || gameState.isPlaying || gameState.gameOver}
          >
            {placingShooter && gameState.coins >= 10 ? 'Cancel' : 'Buy Shooter (10 coins)'}
          </Button>

          <Button
            className='text'
            onClick={startRound}
            disabled={gameState.isPlaying || gameState.gameOver}
          >
            Start Round
          </Button>
        </div>
        
        {gameState.gameOver && (
          <div className="text-xl font-bold text-red-500">Game Over!</div>
        )}
      </div>
    </Card>
  );
};

export default BalloonDefenseGame;