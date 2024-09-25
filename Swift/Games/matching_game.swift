import Foundation

// Define the words in English and Japanese
var englishWords = ["apple", "banana", "cherry", "date", "fig", "grape", "kiwi", "lemon", "mango", "orange"]
var japaneseWords = ["りんご", "バナナ", "さくらんぼ", "デーツ", "いちじく", "ぶどう", "キウイ", "レモン", "マンゴー", "オレンジ"]

// Shuffle the Japanese words
var shuffledJapaneseWords = japaneseWords.shuffled()

// Initialize the score
var score = 0

/**
 Displays the English and Japanese words to the user.
 */
func displayWords() {
    print("Match the English words with their Japanese counterparts:")
    for (index, word) in englishWords.enumerated() {
        print("\(index + 1). \(word)")
    }
    print("\nJapanese words:")
    for (index, word) in shuffledJapaneseWords.enumerated() {
        print("\(index + 1). \(word)")
    }
}

/**
 Gets user input and checks if the selected English and Japanese words match.
 
 - Returns: A tuple containing a boolean indicating if the match is correct, and the indices of the selected English and Japanese words.
 */
func getUserInput() -> (Bool, Int, Int) {
    print("\nEnter the number of the English word followed by the number of the Japanese word (e.g., 1 3): ", terminator: "")
    if let input = readLine(), let englishIndex = Int(input.split(separator: " ")[0]), let japaneseIndex = Int(input.split(separator: " ")[1]) {
        // Check if the selected words match
        return (japaneseWords[Int(englishIndex) - 1] == shuffledJapaneseWords[Int(japaneseIndex) - 1], Int(englishIndex) - 1, Int(japaneseIndex) - 1)
    }
    return (false, -1, -1)
}

// Main game loop
while true {
    displayWords()
    let (result, englishIndex, japaneseIndex) = getUserInput()
    if result {
        print("\nCorrect!")
        score += 1 // Increase score
        print("Current score: \(score)\n")
        // Remove the matched words from the lists
        shuffledJapaneseWords.remove(at: japaneseIndex)
        japaneseWords.remove(at: englishIndex)
        englishWords.remove(at: englishIndex)
    } else {
        print("\nIncorrect. Try again.")
        score -= 1  // Decrease score
        print("Current score: \(score)\n")
    }
    // Check if all words have been matched
    if shuffledJapaneseWords.count == 0 {
        break
    }
}

// Display the final score
print("\nGame over! Your final score is \(score).")
