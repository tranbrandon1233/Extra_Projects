import Foundation

// Hangman game class
class Hangman {
    // List of words to guess
    private let words = ["swift", "programming", "computer", "algorithm", "development"]
    // The word to guess in the current game
    private var wordToGuess: String
    // Set of letters guessed by the player
    private var guessedLetters: Set<Character>
    // Number of remaining attempts
    private var remainingAttempts: Int
    
    // Initializes a new game
    init() {
        // Randomly select a word from the list
        wordToGuess = words.randomElement()!
        // Initialize the set of guessed letters as empty
        guessedLetters = []
        // Set the number of attempts to 6
        remainingAttempts = 6
    }
    
    // Main function to play the game
    func play() {
        print("Welcome to Hangman!")
        
        // Game loop
        while true {
            // Print the current game status
            printGameStatus()
            
            // Check if the game is won
            if isGameWon() {
                print("Congratulations! You've guessed the word: \(wordToGuess)")
                break
            }
            
            // Check if the game is lost
            if isGameLost() {
                print("Game over! The word was: \(wordToGuess)")
                break
            }
            
            // Get a letter guess from the player
            let guess = getPlayerGuess()
            // Process the player's guess
            processGuess(guess)
        }
    }
    
    // Prints the current game status
    private func printGameStatus() {
        // Create a string representing the word to guess, with underscores for unguessed letters
        let wordStatus = wordToGuess.map { guessedLetters.contains($0) ? String($0) : "_" }.joined(separator: " ")
        print("\nWord: \(wordStatus)")
        print("Attempts remaining: \(remainingAttempts)")
        print("Guessed letters: \(guessedLetters.sorted().map { String($0) }.joined(separator: ", "))")
    }
    
    // Checks if the game is won
    private func isGameWon() -> Bool {
        // The game is won if all letters in the word to guess have been guessed
        return wordToGuess.allSatisfy { guessedLetters.contains($0) }
    }
    
    // Checks if the game is lost
    private func isGameLost() -> Bool {
        // The game is lost if there are no remaining attempts
        return remainingAttempts == 0
    }
    
    // Gets a letter guess from the player
    private func getPlayerGuess() -> Character {
        while true {
            print("\nEnter a letter: ", terminator: "")
            // Validate the player's input
            if let input = readLine(), input.count == 1, let guess = input.lowercased().first {
                return guess
            }
            print("Invalid input. Please enter a single letter.")
        }
    }
    
    // Processes a player's guess
    private func processGuess(_ guess: Character) {
        // Check if the letter has already been guessed
        if guessedLetters.contains(guess) {
            print("You've already guessed that letter.")
            return
        }
        
        // Add the guessed letter to the set of guessed letters
        guessedLetters.insert(guess)
        
        // Check if the guessed letter is in the word to guess
        if wordToGuess.contains(guess) {
            print("Good guess!")
        } else {
            print("Wrong guess!")
            // Decrease the number of remaining attempts
            remainingAttempts -= 1
        }
    }
}

// Create a new game and play it
let game = Hangman()
game.play()
