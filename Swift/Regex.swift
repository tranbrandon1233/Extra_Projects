import Foundation

// Sample text block
let sampleText = """
Here are some emails: johnmaldoe@gmail.com, janedoe@yahoo.com, 
mal123@gmail.com, notmal@gmail.com, random1234@gmail.com, 
person@mal.com, malone@gmail.com, tricky_malperson@gmail.com. 
Also, don't forget malfunctioning@gmail.com and mal@gail.com!
"""

// Define the regex pattern to match emails
let pattern = #"\b\w*mal\w*@gmail\.com\b"#

do {
    // Create a regular expression object
    let regex = try NSRegularExpression(pattern: pattern)
    
    // Find matches in the sample text
    let matches = regex.matches(in: sampleText, range: NSRange(sampleText.startIndex..., in: sampleText))
    
    // Extract the matched email addresses
    let matchedEmails = matches.map {
        String(sampleText[Range($0.range, in: sampleText)!])
    }
    
    // Print the results
    print("Matched emails:")
    matchedEmails.forEach { print($0) }
    
} catch {
    // Handle invalid regular expressions
    print("Invalid regex: \(error)")
}
