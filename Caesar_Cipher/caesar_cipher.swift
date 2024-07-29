import Foundation

func encryptAndSave(text: String, shift: Int, fileName: String) {
    // Ensure input is valid ASCII and not too long
    guard text.utf8.count <= 200, text.utf8.allSatisfy({ $0 < 128 }) else {
        print("Invalid input: Text must be ASCII and not exceed 200 characters.")
        return
    }
    
    let encrypted = text.map { char -> Character in
        if let ascii = char.asciiValue, ascii >= 32, ascii <= 126 {
            let shifted = Int(ascii) + shift
            return Character(UnicodeScalar((shifted - 32) % 95 + 32)!)
        } else if char.isASCII {
            return char
        } else {
            return "?"
        }
    }
    
    let encryptedString = String(encrypted)
    
    do {
        try encryptedString.write(toFile: fileName, atomically: true, encoding: .utf8)
        print("Encrypted text saved to \(fileName)")
    } catch {
        print("Error saving file: \(error.localizedDescription)")
    }
}

// Example usage
let inputText = "Hello, World!"
let shiftValue = 3
let outputFileName = "encrypted.txt"

encryptAndSave(text: inputText, shift: shiftValue, fileName: outputFileName)