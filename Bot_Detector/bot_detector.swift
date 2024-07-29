import Foundation

func checkMessageHistory(_ messages: [String]) -> Bool {
    var criteriaCount = 0
    
    // Criterion 1: A phrase of at least three words is repeated three or more times in a single message
        for message in messages {
            let words = message.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            
            // If the number of words is less than 3, continue to the next message
            if words.count < 3 {
                continue
            }
            
            // Create a dictionary to store phrases and their counts
            var phraseCount = [String: Int]()
            
            // Loop through the words and create phrases of three 
                for i in 0...(words.count - 3) {
                    let phrase = "\(words[i]) \(words[i + 1]) \(words[i + 2])"
                    phraseCount[phrase, default: 1] += 1
                }
                
                if words.count == 3 && words.dropFirst().allSatisfy({ $0 == words.first }){
                     print("1")
                    criteriaCount += 1
                    break
                }
                else{
                    for count in phraseCount.values {
                        if count >= 3 {
                        print("1")
                            criteriaCount += 1
                            break
                        }
                    }
                }
            
            
            }
    
    // Criterion 2: A phrase of at least three words is repeated three or more times between messages
    let allText = messages.joined(separator: " ")
    let allWords = allText.split(separator: " ")
    if allWords.count >= 3 {
        for i in 0...allWords.count - 3 {
            let phrase = allWords[i..<i+3].joined(separator: " ")
            if allText.components(separatedBy: phrase).count > 3 {
                criteriaCount += 1
                print("2")
                break
            }
        }
    }
    
    // Criterion 3: Any message contains a URL to "bit.ly" or "tinyurl.com"
    if messages.contains(where: { $0.lowercased().contains("bit.ly") || $0.lowercased().contains("tinyurl.com") }) {
        criteriaCount += 1
        print("3")
    }
    
    // Criterion 4: The lengths of 35% or more of the messages are the same
    let messageLengths = messages.map { $0.count }
    let lengthCounts = Dictionary(grouping: messageLengths, by: { $0 }).mapValues { $0.count }
    if Double(lengthCounts.values.max() ?? 0) / Double(messages.count) >= 0.35 {
        criteriaCount += 1
        print("4")

    }
    
    // Criterion 5: No messages less than 100 characters long and no messages longer than 250 characters
    if messages.allSatisfy({ $0.count >= 100 && $0.count <= 250 }) {
        criteriaCount += 1
        print("5")
    }
    
    return criteriaCount >= 3
}

print(checkMessageHistory(["hi hi hi", "hi hi hi", "hi hi hi"]))
print(checkMessageHistory(["bit.ly/asd1234512345123451234512345123451234512345123451234512345123451234512345123451234512345123451", "exa.mp/les1234512345123451234512345123451234512345123451234512345123451234512345123451234512345123451", "exa.mp/les1234512345123451234512345123451234512345123451234512345123451234512345123451234512345123451"]))
print(checkMessageHistory(["11231234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890", "2131234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890", "112312345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901111"]))
print(checkMessageHistory(["bit.ly/asdf", "hi hi hi", "hi hi hi"]))
print(checkMessageHistory(["aaa", "nnn", "ccc"]))
print(checkMessageHistory(["hi", "hi hi", "hi hi hi"]))