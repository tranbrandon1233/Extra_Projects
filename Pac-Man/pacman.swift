import Foundation

/// Class representing the Pacman game.
class PacmanGame {
    let width = 20  // Width of the maze
    let height = 10  // Height of the maze
    var maze: [[Character]]  // 2D array representing the maze
    var player: (x: Int, y: Int)  // Player's position
    var ghosts: [(x: Int, y: Int)]  // List of ghosts' positions
    var health: Int  // Player's health
    var pelletsRemaining: Int  // Number of pellets remaining in the maze

    /// Initializes the game with a predefined maze, player, and ghosts.
    init() {
        maze = [
            ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", ".", "#", "#", ".", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", "#", ".", "#", "#", "#", ".", "#", "#", ".", "#", "#", "#", ".", "#", "#", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "#", "#", ".", "#", ".", "#", "#", "#", "#", "#", "#", ".", "#", ".", "#", "#", ".", "#"],
            ["#", ".", ".", ".", ".", "#", ".", ".", ".", "#", "#", ".", ".", ".", "#", ".", ".", ".", ".", "#"],
            ["#", ".", "#", "#", ".", "#", "#", "#", ".", "#", "#", ".", "#", "#", "#", ".", "#", "#", ".", "#"],
            ["#", ".", "#", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "#", ".", "#"],
            ["#", ".", ".", ".", ".", "#", "#", "#", "#", ".", ".", "#", "#", "#", "#", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"]
        ]
        player = (1, 1)  // Initial player position
        ghosts = [(18, 8), (18, 1), (1, 8)]  // Initial ghosts' positions
        health = 3  // Initial player health
        pelletsRemaining = 0  // Initial pellets count
        
        // Count the number of pellets in the maze
        for y in 0..<height {
            for x in 0..<width {
                if maze[y][x] == "." {
                    pelletsRemaining += 1
                }
            }
        }
        
        maze[player.y][player.x] = "P" // Place the player in the maze
        for ghost in ghosts {
            maze[ghost.y][ghost.x] = "G" // Place the ghosts in the maze
        }
    }
    /// Prints the current state of the maze, player's health, and remaining pellets.
    func printMaze() {
        for row in maze {
            print(String(row))
        }
        print("Health: \(health)")
        print("Pellets remaining: \(pelletsRemaining)")
    }
    
        /// Moves the player in the specified direction.
        /// - Parameter:
        ///     - direction: The direction to move the player ('w', 'a', 's', 'd').
    func movePlayer(_ direction: Character) {
        let (dx, dy) = getDirection(direction)
        let newX = player.x + dx
        let newY = player.y + dy
        // Check if the new position is not a wall
        if maze[newY][newX] != "#" {
            if maze[newY][newX] == "." {
                pelletsRemaining -= 1
            } else if maze[newY][newX] == "G" {
                health -= 1
            }
            maze[player.y][player.x] = " "  // Clear the old player position
            player = (newX, newY)  // Update the player position
            maze[player.y][player.x] = "P"  // Place the player in the new position
        }
        moveGhosts()  // Move the ghosts after the player moves
    }
    
    func moveGhosts() {
        for i in 0..<ghosts.count {
            maze[ghosts[i].y][ghosts[i].x] = "."  // Clear the old ghost position
            
            let directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  // Possible directions
            
            var validMoves = [(Int, Int)]() // List of valid moves
             
             // Check for valid moves
            for (dx, dy) in directions {
                let newX = ghosts[i].x + dx
                let newY = ghosts[i].y + dy
                if maze[newY][newX] != "#" && maze[newY][newX] != "G" {
                    validMoves.append((newX, newY))
                }
            }
             // Move the ghost to a random valid position
            if !validMoves.isEmpty {
                let randomMove = validMoves.randomElement()!
                ghosts[i] = randomMove
            }
            // Check if the ghost caught the player
            if ghosts[i] == player {
                health -= 1
            } else {
                maze[ghosts[i].y][ghosts[i].x] = "G" // Place the ghost in the new position
            }
        }
    }
     /// Returns the direction vector for the given input.
    /// - Parameter:
    ///     - input: The input character ('w', 'a', 's', 'd').
    /// - Returns: A tuple representing the direction vector.
    func getDirection(_ input: Character) -> (Int, Int) {
        switch input.lowercased() {
        case "w": return (0, -1)
        case "s": return (0, 1)
        case "a": return (-1, 0)
        case "d": return (1, 0)
        default: return (0, 0)
        }
    }
    
    /// Checks if the game is over.
    /// - Returns: A boolean indicating whether the game is over.
    func isGameOver() -> Bool {
        return health <= 0 || pelletsRemaining == 0
    }
    
    /// Starts the game loop.
    func play() {
        while !isGameOver() {
            printMaze()
            print("Enter move (WASD): ", terminator: "")
            if let input = readLine()?.first {
                movePlayer(input)
            }
        }
        
        printMaze()
        if health <= 0 {
            print("Game Over! You lost all your health.")
        } else {
            print("Congratulations! You won by collecting all pellets.")
        }
    }
}

let game = PacmanGame()
game.play()