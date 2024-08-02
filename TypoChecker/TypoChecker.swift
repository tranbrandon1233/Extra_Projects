import Foundation

/// A class to check for typos in words.
class TypoChecker {
    /// A set of common words to check against.
    private let dictionary: Set<String>
    
    /// Initializes the TypoChecker with a predefined set of common words.
    init() {
        // Simple array of common words
        let words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
                     "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                     "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                     "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
                     "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"]
        dictionary = Set(words)
    }
    
    /// Checks if a given word is in the dictionary.
    /// - Parameter word: The word to check.
    /// - Returns: `true` if the word is in the dictionary, `false` otherwise.
    func checkWord(_ word: String) -> Bool {
        return dictionary.contains(word.lowercased())
    }
}

/// Appends text to a file. If the file does not exist, it creates a new one.
/// - Parameters:
///   - text: The text to append.
///   - fileName: The name of the file.
func appendToFile(_ text: String, fileName: String) {
    let fileManager = FileManager.default
    let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
    let fileURL = documentsDirectory.appendingPathComponent(fileName)
    
    do {
        if fileManager.fileExists(atPath: fileURL.path) {
            let fileHandle = try FileHandle(forWritingTo: fileURL)
            fileHandle.seekToEndOfFile()
            fileHandle.write(text.data(using: .utf8)!)
            fileHandle.closeFile()
            print("Text appended successfully to \(fileURL.path)")
        } else {
            try text.write(to: fileURL, atomically: true, encoding: .utf8)
            print("New file created and text saved successfully to \(fileURL.path)")
        }
    } catch {
        print("Error saving/appending to file: \(error.localizedDescription)")
    }
}

let typoChecker = TypoChecker()
var userInput = ""
var currentWord = ""

print("Start typing (press Enter twice to finish):")

while true {
    if let line = readLine() {
        // Break the loop if Enter is pressed twice
        if line.isEmpty && userInput.hasSuffix("\n") {
            break
        }
        
        for char in line {
            if char.isLetter {
                currentWord.append(char)
            } else {
                if !currentWord.isEmpty {
                    if !typoChecker.checkWord(currentWord) {
                        print("Warning: Possible typo in word '\(currentWord)'")
                    }
                    currentWord = ""
                }
            }
            userInput.append(char)
        }
        userInput.append("\n")
    }
}

// Check the last word if there is any
if !currentWord.isEmpty {
    if !typoChecker.checkWord(currentWord) {
        print("Warning: Possible typo in word '\(currentWord)'")
    }
}

print("Finished typing. Saving to file...")
appendToFile(userInput, fileName: "user_input.txt")
