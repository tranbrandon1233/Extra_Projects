import Foundation

let mapWidth = 10
let mapHeight = 10
let chestProbability = 0.1
let trapProbability = 0.1

var playerX = 0
var playerY = 0
var playerHealth = 100
var playerWallet = 0
var gameMap: [[Character]] = []

/**
 Starts and runs the main game loop.
 */
func startGame() {
    initializeMap()
    
    while playerHealth > 0 {
        displayMap()
        displayPlayerStatus()
        
        if let move = readLine()?.lowercased() {
            processMove(move)
        }
    }
    
    print("Game Over! Your final score: \(playerWallet)")
}

/**
 Initializes the game map with random chest and trap placements.
 */
func initializeMap() {
    for _ in 0..<mapHeight {
        var row: [Character] = []
        for _ in 0..<mapWidth {
            if Double.random(in: 0...1) < chestProbability {
                row.append("C")  // Chest
            } else if Double.random(in: 0...1) < trapProbability {
                row.append("T")  // Trap
            } else {
                row.append(".")  // Empty space
            }
        }
        gameMap.append(row)
    }
}

/**
 Displays the current state of the game map.
 */
func displayMap() {
    for y in 0..<mapHeight {
        for x in 0..<mapWidth {
            if x == playerX && y == playerY {
                print("P", terminator: "")  // Player
            } else {
                print(gameMap[y][x], terminator: "")
            }
        }
        print()  // New line after each row
    }
}

/**
 Displays the player's current health and wallet balance.
 */
func displayPlayerStatus() {
    print("Health: \(playerHealth) | Gold: \(playerWallet)")
    print("Enter move (W/A/S/D): ", terminator: "")
}

/**
 Processes the player's move input and updates the game state accordingly.
 
 - Parameter move: A string representing the player's move (W/A/S/D).
 */
func processMove(_ move: String) {
    let newX = playerX + (move == "d" ? 1 : (move == "a" ? -1 : 0))
    let newY = playerY + (move == "s" ? 1 : (move == "w" ? -1 : 0))
    
    // Check if the new position is within bounds
    if newX >= 0 && newX < mapWidth && newY >= 0 && newY < mapHeight {
        playerX = newX
        playerY = newY
        
        // Check what's in the new position
        switch gameMap[playerY][playerX] {
        case "C":
            let gold = Int.random(in: 10...50)
            playerWallet += gold
            print("You found a treasure chest! +\(gold) gold")
            gameMap[playerY][playerX] = "."  // Remove the chest
        case "T":
            let damage = Int.random(in: 10...25)
            playerHealth -= damage
            print("Ouch! You stepped on a trap. -\(damage) health")
            gameMap[playerY][playerX] = "."  // Remove the trap
            if playerHealth <= 0 {
                break
            }
        default:
            break
        }
    } else {
        print("You can't move there!")
    }
}

// Start the game
startGame()