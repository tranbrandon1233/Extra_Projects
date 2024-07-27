import Foundation

/** Function to get a random word from a predefined list.

 - Returns: A random word from the predefined list.
*/
func getRandomWord() -> String {
    // Predefined list of words
    let words = ["apple", "beach", "chair", "dance", "eagle"]
    // Return a random word from the list
    return words.randomElement()!
}
/** Function to check the user's guess against the selected word.

 - Parameters:
   - guess: The user's guess.
   - word: The selected word.
- Returns: A string representing the result of the guess.
*/
func checkGuess(_ guess: String, against word: String) -> String {
    var result = ""
    // Convert the guess and the word into arrays of characters
    let guessChars = Array(guess)
    let wordChars = Array(word)
    
    // Check each character in the guess against the corresponding character in the word
    for i in 0..<5 {
        if guessChars[i] == wordChars[i] {
            // If the character is correct and in the correct position, add a green square to the result
            result += "🟩" 
        } else if word.contains(guessChars[i]) {
            // If the character is correct but in the wrong position, add a yellow square to the result
            result += "🟨" 
        } else {
            // If the character is not in the word, add a white square to the result
            result += "⬜" 
        }
    }
    
    // Return the result
    return result
}

/**
 Function to play the game of Wordle.
*/
func playWordle() {
    // Get a random word
    let word = getRandomWord()
    var guessCount = 0
    
    // Welcome the user to the game
    print("Welcome to Wordle! You have 5 guesses to find the 5-letter word.")
    
    // Allow the user to guess up to 5 times
    while guessCount < 5 {
        print("\nEnter your guess (5 letters):")
        // Get the user's guess, making sure it is a 5-letter word
        guard let guess = readLine()?.lowercased(), guess.count == 5 else {
            print("Invalid input. Please enter a 5-letter word.")
            continue
        }
        
        // Increment the guess count
        guessCount += 1
        // Check the guess against the word
        let result = checkGuess(guess, against: word)
        // Print the guess and the result
        print("\(guess)")
        print("\(result)")
        
        // If the guess is correct, congratulate the user and end the game
        if guess == word {
            print("Congratulations! You've guessed the word!")
            return
        }
    }
    
    // If the user has used all their guesses without guessing the word, end the game
    print("\nGame over! The word was: \(word)")
}

// Start the game
playWordle()