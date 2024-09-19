import Foundation

func findHighestNumberUsingCountingSort(_ arr: [Int]) -> Int {
    // Check if the array is empty
    guard !arr.isEmpty else {
        print("Array is empty.")
        return 0
    }

    // Check if all elements are zero
    if arr.allSatisfy({ $0 == 0 }) {
        print("All elements are zero.")
        return 0
    }

    // Extract all digits from the array, ensuring to handle negative numbers correctly
    var digits: [Int] = []
    for num in arr {
        var n = abs(num)  // Use absolute value to handle negative numbers
        if n == 0 {
            digits.append(0)  // Handle case where the number is 0
        } else {
            while n > 0 {
                digits.append(n % 10)
                n /= 10
            }
        }
    }

    // Implement Counting Sort to sort digits in descending order
    func countingSortDescending(_ array: inout [Int]) {
        let maxDigit = 9
        var count = Array(repeating: 0, count: maxDigit + 1)
        var output = Array(repeating: 0, count: array.count)

        // Count occurrences of each digit
        for digit in array {
            count[digit] += 1
        }

        // Modify count array for cumulative count
        for i in stride(from: maxDigit - 1, through: 0, by: -1) {
            count[i] += count[i + 1]
        }

        // Build output array
        for i in 0..<array.count {
            output[count[array[i]] - 1] = array[i]
            count[array[i]] -= 1
        }

        // Copy the array to the original array
        array = output
    }

    countingSortDescending(&digits)

    // Combine sorted digits to form the highest number
    var highestNumberStr = ""
    for digit in digits {
        highestNumberStr += String(digit)
    }

    // Return the calculated number, if not possible return 0
    return Int(highestNumberStr) ?? 0
}

// Testing case
let arr = [56, 23, 89, 12]
let highestNumber = findHighestNumberUsingCountingSort(arr)
print(highestNumber) // Expected to be 98653221

let arrWithZeros = [0, 0, 0]
let zeroNumber = findHighestNumberUsingCountingSort(arrWithZeros)
print(zeroNumber) // Expected to be 0

let emptyArr = [Int]()
let emptyNumber = findHighestNumberUsingCountingSort(emptyArr)
print(emptyNumber) // Expected to be 0

let negativeArr = [-56, -23, -89, -12]
let negativeNumber = findHighestNumberUsingCountingSort(negativeArr)
print(negativeNumber) // Expected to handle negatives and output: 98653221