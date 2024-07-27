import Foundation

/// Maze struct represents the game maze
struct Maze {
    let width: Int  // Width of the maze
    let height: Int  // Height of the maze
    var grid: [[Character]]  // 2D grid representing the maze
    var playerPosition: (x: Int, y: Int) = (0, 0)  // Player's current position
    var exitPosition: (x: Int, y: Int) = (0, 0)  // Position of the exit
    
    /// Initializer for the Maze struct
    /// It initializes the maze layout and finds the player and exit positions
    init() {
        // Layout of the maze
        let mazeLayout = [
            "############",
            "#P     #   #",
            "### ## # # #",
            "#   #  # # #",
            "# ### ## # #",
            "#     #  # #",
            "# ### # ## #",
            "#   # #  # #",
            "### # # ## #",
            "#         E#",
            "############"
        ]
        
        self.height = mazeLayout.count
        self.width = mazeLayout[0].count
        self.grid = mazeLayout.map { Array($0) }
        
        // Find player and exit positions
        for (y, row) in grid.enumerated() {
            for (x, cell) in row.enumerated() {
                if cell == "P" {
                    self.playerPosition = (x, y)
                } else if cell == "E" {
                    self.exitPosition = (x, y)
                }
            }
        }
        
        // Ensure the maze contains both 'P' (player) and 'E' (exit)
        if playerPosition == (0, 0) || exitPosition == (0, 0) {
            fatalError("Maze must contain both 'P' (player) and 'E' (exit)")
        }
    }
    
    /// Function to move the player in a given direction
    /// - Parameter direction: The direction to move the player. It can be "w", "s", "a", or "d".
    /// - Returns: A boolean indicating whether the move was successful.
    mutating func movePlayer(direction: String) -> Bool {
        let (x, y) = playerPosition
        var newX = x
        var newY = y
        
        // Determine new position based on input direction
        switch direction.lowercased() {
        case "w": newY -= 1
        case "s": newY += 1
        case "a": newX -= 1
        case "d": newX += 1
        default: return false
        }
        
        // If the move is valid, update the player's position and return true
        if isValidMove(x: newX, y: newY) {
            grid[y][x] = " "
            grid[newY][newX] = "P"
            playerPosition = (newX, newY)
            return true
        }
        // If the move is not valid, return false
        return false
    }
    
    /// Function to check if a move is valid (i.e., within the maze bounds and not a wall)
    /// - Parameters:
    ///   - x: The x-coordinate of the new position.
    ///   - y: The y-coordinate of the new position.
    /// - Returns: A boolean indicating whether the move is valid.
    func isValidMove(x: Int, y: Int) -> Bool {
        return x >= 0 && x < width && y >= 0 && y < height && grid[y][x] != "#"
    }
    
    /// Function to display the current state of the maze
    func display() {
        for row in grid {
            print(row.map { String($0) }.joined())
        }
    }
    
    /// Function to check if the player has reached the exit and won the game
    /// - Returns: A boolean indicating whether the player has won.
    func hasPlayerWon() -> Bool {
        return playerPosition == exitPosition
    }
}

// Game loop
var maze = Maze()
print("Welcome to the Maze Game!")
print("Use W (up), S (down), A (left), D (right) to move. Q to quit.")
print("Reach 'E' to win the game.")

while true {
    maze.display()
    print("Enter your move: ", terminator: "")
    if let input = readLine()?.lowercased() {
        if input == "q" {
            print("Thanks for playing!")
            break
        }
        if maze.movePlayer(direction: input) {
            if maze.hasPlayerWon() {
                maze.display()
                print("Congratulations! You've reached the exit and won the game!")
                break
            }
        } else {
            print("Invalid move. Try again.")
        }
    }
}
