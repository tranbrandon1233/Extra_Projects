import Foundation

func multiplyLargeNumbers(_ numbers: [String]) -> String {
    // Helper function to multiply two numbers represented as strings
    func multiplyTwoStrings(_ num1: String, _ num2: String) -> String {
        let n1 = Array(num1)
        let n2 = Array(num2)
        let len1 = n1.count
        let len2 = n2.count
        var result = Array(repeating: 0, count: len1 + len2)
        
        for i in stride(from: len1 - 1, through: 0, by: -1) {
            for j in stride(from: len2 - 1, through: 0, by: -1) {
                let mul = (Int(String(n1[i])) ?? 0) * (Int(String(n2[j])) ?? 0)
                let sum = mul + result[i + j + 1]
                result[i + j + 1] = sum % 10
                result[i + j] += sum / 10
            }
        }
        
        // Convert result array to string
        var resultString = result.reduce("") { $0 + String($1) }
        
        // Remove leading zeros
        while resultString.first == "0" && resultString.count > 1 {
            resultString.removeFirst()
        }
        
        return resultString
    }
    
    // Validate and clean input numbers
    let cleanedNumbers = numbers.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                                .filter { !$0.isEmpty && $0.range(of: #"^[0-9]+$"#, options: .regularExpression) != nil }
                                .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "0")) }
                                .map { $0.isEmpty ? "0" : $0 }
    
    // If any number is zero, the result is zero
    if cleanedNumbers.contains("0") {
        return "0"
    }
    
    // Multiply all numbers in the array
    var result = "1"
    for number in cleanedNumbers {
        result = multiplyTwoStrings(result, number)
    }
    
    return result
}

// Example usage
let numbers = ["00123", "456", "789", ""]
let result = multiplyLargeNumbers(numbers)
print(result)  // Output: "44253432"