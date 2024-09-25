import Foundation

// Player class to represent each player in the game
class Player {
    var name: String
    var health: Int
    var isDefending: Bool
    
    // Initializer to set up a player with a name and health
    init(name: String, health: Int) {
        self.name = name
        self.health = health
        self.isDefending = false
    }
    
    // Method for attacking, returns a fixed damage value
    func attack() -> Int {
        return 20
    }
    
    // Method to set the player in defending mode
    func defend() {
        isDefending = true
    }
}

// CombatGame class to manage the game logic
class CombatGame {
    var player1: Player
    var player2: Player
    var currentPlayer: Player
    
    // Initializer to set up the game with two players
    init() {
        player1 = Player(name: "Player 1", health: 100)
        player2 = Player(name: "Player 2", health: 100)
        currentPlayer = player1
    }
    
    // Method to start the game loop
    func start() {
        // Continue the game until one player's health drops to 0 or below
        while player1.health > 0 && player2.health > 0 {
            performTurn()
            switchPlayer()
        }
        
        // Announce the winner once the game is over
        announceWinner()
    }
    
    // Method to perform a turn for the current player
    private func performTurn() {
        print("\(currentPlayer.name)'s turn. Choose action (1: Attack, 2: Defend):")
        guard let choice = readLine(), let action = Int(choice) else {
            print("Invalid input. Skipping turn.")
            return
        }
        
        // Execute the chosen action
        switch action {
        case 1:
            attack()
        case 2:
            defend()
        default:
            print("Invalid action. Skipping turn.")
        }
    }
    
    // Method to handle the attack action
    private func attack() {
        // Determine the defender based on the current player
        let defender = (currentPlayer === player1) ? player2 : player1
        var damage = currentPlayer.attack()
        
        // If the defender is defending, reduce the damage by half
        if defender.isDefending {
            damage /= 2
            defender.isDefending = false
        }
        defender.health -= damage
  
        print("\(currentPlayer.name) attacks for \(damage) damage!")
        print("\(defender.name) has \(defender.health) health points remaining.")
    }
    
    // Method to handle the defend action
    private func defend() {
        currentPlayer.defend()
        print("\(currentPlayer.name) is defending.")
    }
    
    // Method to switch the current player after each turn
    private func switchPlayer() {
        currentPlayer = (currentPlayer === player1) ? player2 : player1
    }
    
    // Method to announce the winner at the end of the game
    private func announceWinner() {
        print("Game Over!")
        if player1.health <= 0 {
            print("\(player2.name) wins!")
        } else {
            print("\(player1.name) wins!")
        }
    }
}

// Create a new game instance and start the game
let game = CombatGame()
game.start()