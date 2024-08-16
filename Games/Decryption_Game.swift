import Foundation

protocol Puzzle {
    var description: String { get }
    var maxAttempts: Int { get }
    
    func validateInput(_ input: String) -> Bool
    func generateHint() -> String
}

class VigenereCipherPuzzle: Puzzle {
    let description = "Encrypt the following message using a Vigenère cipher:"
    let maxAttempts = 3
    
    let plainText: String 
    let keyword: String
    
    init(plainText: String, keyword: String) {
        self.plainText = plainText
        self.keyword = keyword
    }
    
    /**
     Validates the user's input against the correct encrypted message.
     
     - Parameter input: The user's attempted encryption
     - Returns: Boolean indicating whether the input is correct
     */
    func validateInput(_ input: String) -> Bool {
        return input.lowercased() == plainText.lowercased()
    }
    
    /**
     Generates a hint for the puzzle.
     
     - Returns: A string containing a hint about the cipher keyword
     */
    func generateHint() -> String {
        return "The keyword for this Vigenère cipher is '\(keyword)'. Use each letter of the keyword to determine the shift for the corresponding letter in the plaintext."
    }
    
    /**
     Encrypts a plaintext message using the Vigenère cipher.
     
     - Parameters:
       - plainText: The message to encrypt
       - keyword: The keyword used for encryption
     - Returns: The encrypted message
     */
    func encrypt(_ plainText: String, keyword: String) -> String {
        let alphabet = "abcdefghijklmnopqrstuvwxyz"
        var result = ""
        var keyIndex = 0
        
        for char in plainText.lowercased() {
            if let charIndex = alphabet.firstIndex(of: char) {
                let keyChar = keyword[keyword.index(keyword.startIndex, offsetBy: keyIndex % keyword.count)]
                let keyCharIndex = alphabet.distance(from: alphabet.startIndex, to: alphabet.firstIndex(of: keyChar)!)
                let encryptedIndex = (alphabet.distance(from: alphabet.startIndex, to: charIndex) + keyCharIndex) % 26
                result.append(alphabet[alphabet.index(alphabet.startIndex, offsetBy: encryptedIndex)])
                keyIndex += 1
            } else {
                result.append(char)
            }
        }
        
        return result
    }
    
    /**
     Decrypts a ciphertext message using the Vigenère cipher.
     
     - Parameters:
       - cipherText: The message to decrypt
       - keyword: The keyword used for decryption
     - Returns: The decrypted message
     */
    func decrypt(_ cipherText: String, keyword: String) -> String {
        let alphabet = "abcdefghijklmnopqrstuvwxyz"
        var result = ""
        var keyIndex = 0
        
        for char in cipherText.lowercased() {
            if let charIndex = alphabet.firstIndex(of: char) {
                let keyChar = keyword[keyword.index(keyword.startIndex, offsetBy: keyIndex % keyword.count)]
                let keyCharIndex = alphabet.distance(from: alphabet.startIndex, to: alphabet.firstIndex(of: keyChar)!)
                let decryptedIndex = (alphabet.distance(from: alphabet.startIndex, to: charIndex) - keyCharIndex + 26) % 26
                result.append(alphabet[alphabet.index(alphabet.startIndex, offsetBy: decryptedIndex)])
                keyIndex += 1
            } else {
                result.append(char)
            }
        }
        
        return result
    }
}

class EncryptionPuzzleGame {
    private var currentPuzzle: Puzzle
    private(set) var attempts: Int = 0
    
    /**
     Initializes a new game with the given puzzle.
     
     - Parameter puzzle: The puzzle to be solved in this game
     */
    init(puzzle: Puzzle) {
        self.currentPuzzle = puzzle
    }
    
    /**
     Starts the game by displaying the puzzle description.
     */
    func start() {
        print(currentPuzzle.description)
    }
    
    /**
     Processes the user's answer submission.
     
     - Parameter answer: The user's submitted answer
     - Returns: A tuple containing a boolean indicating if the answer is correct and a message with feedback
     */
    func submitAnswer(_ answer: String) -> (isCorrect: Bool, message: String) {
        attempts += 1
        let isCorrect = currentPuzzle.validateInput(answer)
        
        if isCorrect {
            return (true, "Congratulations! You've solved the puzzle.")
        } else {
            if attempts >= currentPuzzle.maxAttempts {
                return (false, "Game over. You've used all your attempts.")
            } else {
                let attemptsLeft = currentPuzzle.maxAttempts - attempts
                var message = "Incorrect answer. You have \(attemptsLeft) attempt(s) left."
                if attempts == currentPuzzle.maxAttempts - 1 {
                    message += " Here's a hint: \(currentPuzzle.generateHint())"
                }
                return (false, message)
            }
        }
    }
}

// Example usage with user input for Vigenère cipher:
let vigenerePuzzle = VigenereCipherPuzzle(plainText: "hello world", keyword: "key")
let game = EncryptionPuzzleGame(puzzle: vigenerePuzzle)
let encryptedText = vigenerePuzzle.encrypt(vigenerePuzzle.plainText, keyword: vigenerePuzzle.keyword)
print("Welcome to the Encryption Puzzle Game: Vigenère Cipher Edition!")
game.start()

print("The encrypted word is: \(encryptedText)")
print("The keyword for encryption is: \(vigenerePuzzle.keyword)")

var isSolved = false
while !isSolved && game.attempts < vigenerePuzzle.maxAttempts {
    print("\nAttempt \(game.attempts + 1) of \(vigenerePuzzle.maxAttempts)")
    print("Enter your encryption attempt:")
    
    if let userAnswer = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) {
        let (solved, message) = game.submitAnswer(userAnswer)
        print(message)
        isSolved = solved
    } else {
        print("Invalid input. Please try again.")
    }
}
