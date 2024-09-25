/**
 Filters an array of integers to keep only those containing a specific digit.

 This function uses modulo and division operations to check if each number contains the specified digit.

 - Parameters:
   - numbers: An array of integers to filter.
   - digit: A single-digit integer (0-9) to filter by.

 - Returns: An array of integers from the input that contain the specified digit.
           Returns an empty array if the input is invalid or empty.

 - Complexity: O(n * log m), where n is the number of elements in the array and m is the maximum number in the array.
 */
func filterNumbers(_ numbers: [Int], containingDigit digit: Int) -> [Int] {
    // Validate the input digit
    guard (0...9).contains(digit) else {
        print("Invalid digit. Please provide a single-digit number (0-9).")
        return []
    }
    
    return numbers.filter { number in
        var n = abs(number) // Use absolute value to handle negative numbers
        
        // Special case: if the number is 0 and we're looking for 0
        if n == 0 && digit == 0 {
            return true
        }
        
        // Check each digit of the number
        while n > 0 {
            if n % 10 == digit { // If the last digit matches, return true
                return true
            }
            n /= 10 // Remove the last digit
        }
        return false // If we've checked all digits and found no match, return false
    }
}

// Example usage:
let numbers1: [Int] = [12, 23, 45, 62, 78, 92, 102, 222]
let result1 = filterNumbers(numbers1, containingDigit: 2)
print("Result 1:", result1) // Output: [12, 23, 62, 92, 102, 222]

// Test with an empty array
let emptyArray: [Int] = []
let result2 = filterNumbers(emptyArray, containingDigit: 2)
print("Result 2:", result2) // Output: []

// Test with invalid digit
let result3 = filterNumbers(numbers1, containingDigit: 109)
print("Result 3:", result3) // Output: [] (with an error message)

// Test with invalid digit
let result4 = filterNumbers(numbers1, containingDigit: -12)
print("Result 4:", result4) // Output: [] (with an error message)