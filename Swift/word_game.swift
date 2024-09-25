import Foundation

class WordGame {
    private var letters: [Character] = []
    private var validWords: [String] = []
    private var playerScore: Int = 0
    
    // Dictionary of valid words (you can expand this as needed)
    private let dictionary = ["cat", "dog", "bird", "fish", "rat", "bat", "hat", "mat", "van", "just", "pan"]
    
    init() {
        generateRandomLetters()
    }
    
    // Get random letters that can be used
    private func generateRandomLetters() {
        let alphabet = "abcdefghijklmnopqrstuvwxyz"
        letters = Array(alphabet.shuffled().prefix(10))
    }
    
    // Play the game
   func play() {
    print("Welcome to the Word Forming Game!")
    print("Your letters are: \(letters.map { String($0) }.joined(separator: " "))")
    
    var shouldContinue = true
    while shouldContinue {
        print("\nEnter a word (or 'q' to quit):")
        if let input = readLine()?.lowercased() {
            if input == "q" {
                shouldContinue = false
            } else if isValidWord(input) {
                if !validWords.contains(input) {
                    validWords.append(input)
                    playerScore += input.count
                    print("Correct! Your score is now \(playerScore)")
                } else {
                    print("You've already found this word.")
                }
            } else {
                print("Invalid word. Try again.")
            }
        }
    }
    
    print("\nGame Over!")
    print("Your final score: \(playerScore)")
    print("Words you found: \(validWords.joined(separator: ", "))")
}
    private func isValidWord(_ word: String) -> Bool {
        // Check if the word is in the dictionary
        guard dictionary.contains(word) else {
            return false
        }
        
        // Check if the word can be formed from the available letters
        var availableLetters = letters
        for char in word {
            if let index = availableLetters.firstIndex(of: char) {
                availableLetters.remove(at: index)
            } else {
                return false
            }
        }
        
        return true
    }
}

// Start the game
let game = WordGame()
game.play()