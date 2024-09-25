import Foundation

/// Function to count letters in a sentence
/// - Parameter sentence: The sentence in which to count letters
/// - Returns: A dictionary with characters as keys and their counts as values
func countLetters(in sentence: String) -> [Character: Int] {
    var letterCounts: [Character: Int] = [:]
    // Iterate over each character in the sentence, converted to lowercase
    for char in sentence.lowercased() where char.isLetter {
        // Increment the count for the character in the dictionary
        letterCounts[char, default: 0] += 1
    }
    return letterCounts
}

/// Function to rank sentences based on letter count
/// - Parameters:
///   - sentences: An array of sentences to rank
///   - letter: The letter to rank sentences by
///   - ascending: A boolean indicating whether to rank in ascending order
/// - Returns: An array of tuples containing the sentence and its count of the specified letter
func rankSentences(sentences: [String], by letter: Character, ascending: Bool) -> [(String, Int)] {
    // Map each sentence to a tuple containing the sentence and the count of the specified letter
    let rankedSentences = sentences.map { sentence -> (String, Int) in
        let letterCounts = countLetters(in: sentence)
        return (sentence, letterCounts[letter] ?? 0)
    }
    // Sort the sentences based on the count of the specified letter
    return rankedSentences.sorted { ascending ? $0.1 < $1.1 : $0.1 > $1.1 }
}

/// Function to run main program
    func run(){
    print("Enter a block of text:")
    guard let input = readLine(strippingNewline: false) else {
        print("Invalid input")
        exit(1)
    }
    
    // Split the input into sentences
    let sentences = input.components(separatedBy: CharacterSet(charactersIn: ".!?"))
        .map { $0.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    
    // Print each sentence with its letter counts
    for (index, sentence) in sentences.enumerated() {
        print("\nSentence \(index + 1):")
        print(sentence)
        let letterCounts = countLetters(in: sentence)
        // Print each letter and its count in the sentence
        for (letter, count) in letterCounts.sorted(by: { $0.key < $1.key }) {
            print("\(letter): \(count)", terminator: " ")
        }
        print()
    }
    
    // Ranking loop
    while true {
        print("\nEnter a letter to rank sentences by (or 'q' to quit):")
        guard let letterString = readLine()?.lowercased(), let letter = letterString.first else { continue }
        
        if letter == "q" { break }
        
        print("Rank in ascending (a) or descending (d) order?")
        guard let order = readLine()?.lowercased().first else { continue }
        
        let ascending = (order == "a")
        let rankedSentences = rankSentences(sentences: sentences, by: letter, ascending: ascending)
        
        print("\nRanked sentences by '\(letter)' (\(ascending ? "ascending" : "descending")):")
        // Print each ranked sentence with its count of the specified letter
        for (index, (sentence, count)) in rankedSentences.enumerated() {
            print("\(index + 1). [\(count)] \(sentence)")
        }
    }
    
    print("Terminating program...")
}

run()