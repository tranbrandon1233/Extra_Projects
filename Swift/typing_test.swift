import Foundation

func calculateTypingStats(original: String, typed: String, time: TimeInterval) -> (wpm: Double, typos: Int) {
    let originalWords = original.split(separator: " ")
    let typedWords = typed.split(separator: " ")
    let wordCount = Double(originalWords.count)
    
    let minutes = time / 60.0
    let wpm = wordCount / minutes
    
    var typos = 0
    for (index, word) in typedWords.enumerated() {
        if index < originalWords.count && word != originalWords[index] {
            typos += 1
        }
    }
    
    return (wpm: wpm, typos: typos)
}

let paragraph = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."

print("Please type the following paragraph without using backspace:")
print(paragraph)
print("\nPress Enter when you're ready to start typing.")
_ = readLine()

let startTime = Date()

print("Start typing now:")
let typedText = readLine() ?? ""

let endTime = Date()
let totalTime = endTime.timeIntervalSince(startTime)

let (wpm, typos) = calculateTypingStats(original: paragraph, typed: typedText, time: totalTime)

print("\nTyping statistics:")
print(String(format: "Total time: %.2f seconds", totalTime))
print(String(format: "Words per minute: %.2f", wpm))
print("Number of typos: \(typos)")