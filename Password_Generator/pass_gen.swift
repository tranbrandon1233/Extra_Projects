import Foundation

/**
Generates a random password of a given length with optional character sets.

- Parameters:
   - length: The length of the password to generate.
   - includeUppercase: A Boolean value indicating whether to include uppercase letters in the password.
   - includeLowercase: A Boolean value indicating whether to include lowercase letters in the password.
   - includeNumbers: A Boolean value indicating whether to include numbers in the password.
   - includeSpecialChars: A Boolean value indicating whether to include special characters in the password.

- Returns: A string representing the generated password. If no character set is selected, it returns an error message.
*/
func generatePassword(length: Int, includeUppercase: Bool = true, includeLowercase: Bool = true, includeNumbers: Bool = true, includeSpecialChars: Bool = true) -> String {
    var charset = "" // Initialize the character set string
    let uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" // Define the uppercase letters
    let lowercase = "abcdefghijklmnopqrstuvwxyz" // Define the lowercase letters
    let numbers = "0123456789" // Define the numbers
    let specialChars = "!@#$%^&*()-_=+[]{}|;:,.<>?" // Define the special characters
    
    // Add the selected character sets to the charset string
    if includeUppercase { charset += uppercase }
    if includeLowercase { charset += lowercase }
    if includeNumbers { charset += numbers }
    if includeSpecialChars { charset += specialChars }
    
    // Check if the charset string is empty
    guard !charset.isEmpty else {
        return "Error: No character set selected" // Return an error message if no character set is selected
    }
    
    var password = "" // Initialize the password string
    for _ in 0..<length { // Loop for the specified length
        if let randomChar = charset.randomElement() { // Get a random character from the charset string
            password.append(randomChar) // Append the random character to the password string
        }
    }
    
    return password // Return the generated password
}

// Example usage:
let password = generatePassword(length: 12, includeUppercase: true, includeLowercase: true, includeNumbers: true, includeSpecialChars: true) // Generate a password
print("Generated password: \(password)") // Print the generated password
