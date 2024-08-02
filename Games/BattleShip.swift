import Foundation

// Define the size of the grid
let gridSize = 10

// Define the ships and their sizes
let ships = [
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2
]

// Player class to represent each player
class Player {
    var name: String
    var grid: [[String]]
    var shipsLocations: [String:[(Int, Int)]]
    
    init(name: String) {
        self.name = name
        self.grid = Array(repeating: Array(repeating: ".", count: gridSize), count: gridSize)
        self.shipsLocations = [:]
    }
    
    func placeShips() {
        for (shipName, shipSize) in ships {
            while true {
                let isHorizontal = Bool.random()
                let row = Int.random(in: 0..<gridSize)
                let col = Int.random(in: 0..<gridSize)
                if canPlaceShip(size: shipSize, row: row, col: col, horizontal: isHorizontal) {
                    placeShip(name: shipName, size: shipSize, row: row, col: col, horizontal: isHorizontal)
                    break
                }
            }
        }
    }
    
    func canPlaceShip(size: Int, row: Int, col: Int, horizontal: Bool) -> Bool {
        if horizontal {
            guard col + size <= gridSize else { return false }
            return (col..<col+size).allSatisfy { grid[row][$0] == "." }
        } else {
            guard row + size <= gridSize else { return false }
            return (row..<row+size).allSatisfy { grid[$0][col] == "." }
        }
    }
    
    func placeShip(name: String, size: Int, row: Int, col: Int, horizontal: Bool) {
        var positions: [(Int, Int)] = []
        for i in 0..<size {
            let (r, c) = horizontal ? (row, col + i) : (row + i, col)
            grid[r][c] = "S"
            positions.append((r, c))
        }
        shipsLocations[name] = positions
    }
    
    func receiveAttack(row: Int, col: Int) -> Bool {
        if grid[row][col] == "S" {
            grid[row][col] = "X"
            return true
        } else {
            grid[row][col] = "O"
            return false
        }
    }
    
    func allShipsSunk() -> Bool {
        return grid.allSatisfy { row in row.allSatisfy { $0 != "S" } }
    }
    
    func displayGrid(hideShips: Bool) {
        print("  0 1 2 3 4 5 6 7 8 9")
        for (i, row) in grid.enumerated() {
            print("\(i) ", terminator: "")
            for cell in row {
                let symbol = hideShips && cell == "S" ? "." : cell
                print("\(symbol) ", terminator: "")
            }
            print()
        }
    }
}

// Game class to manage the game flow
class BattleshipGame {
    let player1: Player
    let player2: Player
    var currentPlayer: Player
    
    init() {
        player1 = Player(name: "Player 1")
        player2 = Player(name: "Player 2")
        currentPlayer = player1
        
        player1.placeShips()
        player2.placeShips()
    }
    
    func play() {
        while true {
            print("\n\(currentPlayer.name)'s turn")
            print("\nYour grid:")
            currentPlayer.displayGrid(hideShips: false)
            print("\nOpponent's grid:")
            let opponent = currentPlayer === player1 ? player2 : player1
            opponent.displayGrid(hideShips: true)
            
            print("\nEnter column (0-9) and row (0-9) separated by a space:")
            
            guard let input = readLine()?.split(separator: " "),
                  input.count == 2,
                  let col = Int(input[0]),
                  let row = Int(input[1]),
                  (0..<gridSize).contains(row),
                  (0..<gridSize).contains(col) else {
                print("Invalid input. Try again.")
                continue
            }
            
            let hit = opponent.receiveAttack(row: row, col: col)
            
            if hit {
                print("Hit!")
                if opponent.allShipsSunk() {
                    print("\n\(currentPlayer.name) wins!")
                    return
                }
            } else {
                print("Miss!")
            }
            
            currentPlayer = opponent
        }
    }
}

// Start the game
let game = BattleshipGame()
game.play()