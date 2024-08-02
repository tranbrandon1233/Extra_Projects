/// Finds and prints the differences between two strings using character-by-character comparison.
/// - Parameters:
///   - str1: The first string to compare
///   - str2: The second string to compare
func findStringDifferences_Method1(_ str1: String, _ str2: String) {
    // Convert strings to arrays of characters for easier indexing
    let chars1 = Array(str1)
    let chars2 = Array(str2)
    
    // Iterate through both strings simultaneously
    for (index, (char1, char2)) in zip(chars1, chars2).enumerated() {
        if char1 != char2 {
            print("Difference at index \(index): '\(char1)' in string 1, '\(char2)' in string 2")
        }
    }
    
    // Check for extra characters in the longer string
    let maxLength = max(chars1.count, chars2.count)
    for i in min(chars1.count, chars2.count)..<maxLength {
        if i < chars1.count {
            print("Extra character in string 1 at index \(i): '\(chars1[i])'")
        } else {
            print("Extra character in string 2 at index \(i): '\(chars2[i])'")
        }
    }
}

// Example usage
let string1 = "Hello, World!"
let string2 = "Hello, Swift!"
findStringDifferences_Method1(string1, string2)