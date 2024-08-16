import Foundation

/// Generates random data points using a normal distribution
/// - Parameters:
///   - count: Number of data points to generate
///   - minValue: Minimum value of the range
///   - maxValue: Maximum value of the range
/// - Returns: Array of generated data points
func generateDataPoints(count: Int, minValue: Double, maxValue: Double) -> [Double] {
    var dataPoints: [Double] = []
    
    for _ in 0..<count {
        // Generate a random number from a normal distribution
        let u1 = Double.random(in: 0...1)
        let u2 = Double.random(in: 0...1)
        let z = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        
        // Scale the number to fit our range
        let scaled = z * (maxValue - minValue) / 6 // 6 sigma covers 99.7% of data
        let clamped = max(minValue, min(maxValue, scaled))
        
        dataPoints.append(clamped)
    }
    
    return dataPoints
}

/// Calculates the mean of an array of numbers
/// - Parameter data: Array of numbers
/// - Returns: Mean of the numbers
func calculateMean(_ data: [Double]) -> Double {
    return data.reduce(0, +) / Double(data.count)
}

/// Calculates the standard deviation of an array of numbers
/// - Parameter data: Array of numbers
/// - Returns: Standard deviation of the numbers
func calculateStandardDeviation(_ data: [Double]) -> Double {
    let mean = calculateMean(data)
    let variance = data.map { pow($0 - mean, 2) }.reduce(0, +) / Double(data.count)
    return sqrt(variance)
}

/// Calculates the kurtosis of an array of numbers
/// - Parameter data: Array of numbers
/// - Returns: Kurtosis of the numbers
func calculateKurtosis(_ data: [Double]) -> Double {
    let mean = calculateMean(data)
    let std = calculateStandardDeviation(data)
    let n = Double(data.count)
    
    let fourthMoment = data.map { pow(($0 - mean) / std, 4) }.reduce(0, +) / n
    return fourthMoment - 3 // Excess kurtosis
}

/// Plots the data points using ASCII characters
/// - Parameters:
///   - data: Array of data points
///   - bins: Number of bins for the histogram
///   - height: Height of the ASCII plot
func plotASCII(data: [Double], bins: Int = 50, height: Int = 20) {
    let minValue = data.min()!
    let maxValue = data.max()!
    let binWidth = (maxValue - minValue) / Double(bins)
    
    var histogram = [Int](repeating: 0, count: bins)
    
    // Count the number of data points in each bin
    for value in data {
        let index = Int((value - minValue) / binWidth)
        histogram[index == bins ? index - 1 : index] += 1
    }
    
    let maxCount = histogram.max()!
    
    // Plot the histogram
    for row in (0..<height).reversed() {
        let threshold = Double(row) / Double(height) * Double(maxCount)
        for count in histogram {
            if Double(count) > threshold {
                print("█", terminator: "")
            } else {
                print(" ", terminator: "")
            }
        }
        print()
    }
    
    // Print x-axis
    print(String(repeating: "-", count: bins))
    print(String(format: "%.2f", minValue), terminator: "")
    print(String(repeating: " ", count: bins - 12), terminator: "")
    print(String(format: "%.2f", maxValue))
}

// Generate data points
let dataPoints = generateDataPoints(count: 1000, minValue: -10, maxValue: 10)

// Calculate statistics
let mean = calculateMean(dataPoints)
let standardDeviation = calculateStandardDeviation(dataPoints)
let kurtosis = calculateKurtosis(dataPoints)

print("Mean: \(mean)")
print("Standard Deviation: \(standardDeviation)")
print("Kurtosis: \(kurtosis)")

// Plot the results
print("\nDistribution of data points:")
plotASCII(data: dataPoints)