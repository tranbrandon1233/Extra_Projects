
import Foundation

import Foundation

func calculateAndRankScores1(_ scores: [String: [Double]]) -> [(String, Double)] {
    // Calculate final scores
    var finalScores: [String: Double] = [:]
    for (student, exams) in scores {
        let finalScore = exams[0] * 0.2 + exams[1] * 0.3 + exams[2] * 0.5
        finalScores[student] = finalScore
    }
    
    // Convert to array of tuples
    var rankedScores = finalScores.map { ($0.key, $0.value) }
    
    // Quick Sort implementation
    func quickSort(_ arr: inout [(String, Double)], low: Int, high: Int) {
        if low < high {
            let pivotIndex = partition(&arr, low: low, high: high)
            quickSort(&arr, low: low, high: pivotIndex - 1)
            quickSort(&arr, low: pivotIndex + 1, high: high)
        }
    }
    
    func partition(_ arr: inout [(String, Double)], low: Int, high: Int) -> Int {
        let pivot = arr[high].1
        var i = low - 1
        
        for j in low..<high {
            if arr[j].1 >= pivot {
                i += 1
                arr.swapAt(i, j)
            }
        }
        
        arr.swapAt(i + 1, high)
        return i + 1
    }
    
    // Sort the array
    quickSort(&rankedScores, low: 0, high: rankedScores.count - 1)
    
    return rankedScores
}

// Test the functions
let studentScores = [
    "Alice": [85.0, 90.0, 95.0],
    "Bob": [75.0, 80.0, 85.0],
    "Charlie": [95.0, 90.0, 88.0],
    "David": [70.0, 75.0, 80.0],
    "Eve": [88.0, 92.0, 90.0]
]

// Compare execution times
func timeFunction(_ function: () -> Void) -> TimeInterval {
    let start = DispatchTime.now()
    function()
    let end = DispatchTime.now()
    let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanoTime) / 1_000_000_000
}

print("Method 1 results:")
let result1 = calculateAndRankScores1(studentScores)
for (student, score) in result1 {
    print("\(student): \(score)")
}

let method1Time = timeFunction { _ = calculateAndRankScores1(studentScores) }

print("\nExecution time:")
print("Method 1 (Quick Sort): \(method1Time) seconds")

func calculateAndRankScores2(_ scores: [String: [Double]]) -> [(String, Double)] {
    // Calculate final scores and store in array of tuples
    let rankedScores = scores.map { (student, exams) in
        let finalScore = exams[0] * 0.2 + exams[1] * 0.3 + exams[2] * 0.5
        return (student, finalScore)
    }
    
    // Merge Sort implementation
    func mergeSort(_ arr: [(String, Double)]) -> [(String, Double)] {
        guard arr.count > 1 else { return arr }
        
        let mid = arr.count / 2
        let left = mergeSort(Array(arr[..<mid]))
        let right = mergeSort(Array(arr[mid...]))
        
        return merge(left, right)
    }
    
    // This function takes two sorted arrays (left and right) as input and merges them into a single sorted array.
    func merge(_ left: [(String, Double)], _ right: [(String, Double)]) -> [(String, Double)] {
        // Initialize indices for left and right arrays
        var leftIndex = 0
        var rightIndex = 0
        
        // Initialize the result array
        var result: [(String, Double)] = []
        
        // Loop until we've reached the end of either the left or right array
        while leftIndex < left.count && rightIndex < right.count {
            // If the current element of the left array is greater than or equal to the current element of the right array, append it to the result array
            if left[leftIndex].1 >= right[rightIndex].1 {
                result.append(left[leftIndex])
                leftIndex += 1
            } else {
                // Otherwise, append the current element of the right array to the result array
                result.append(right[rightIndex])
                rightIndex += 1
            }
        }
        
        // If there are remaining elements in the left array, append them to the result array
        result.append(contentsOf: left[leftIndex...])
        
        // If there are remaining elements in the right array, append them to the result array
        result.append(contentsOf: right[rightIndex...])
        
        // Return the merged and sorted array
        return result
    }

    // Call the mergeSort function on the rankedScores array to sort it
    return mergeSort(rankedScores)
    }


print("\nMethod 2 results:")
let result2 = calculateAndRankScores2(studentScores)
for (student, score) in result2 {
    print("\(student): \(score)")
}

let method2Time = timeFunction { _ = calculateAndRankScores2(studentScores) }

print("\nExecution time:")
print("Method 2 (Merge Sort): \(method2Time) seconds")