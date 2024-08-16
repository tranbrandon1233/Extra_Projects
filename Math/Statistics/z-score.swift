import Foundation

/// Generates random data points using inverse transform sampling
/// - Parameters:
///   - count: Number of data points to generate
///   - minValue: Minimum value of the range
///   - maxValue: Maximum value of the range
/// - Returns: Array of generated data points
func generateDataPoints(count: Int, minValue: Double, maxValue: Double) -> [Double] {
    var dataPoints: [Double] = []
    
    for _ in 0..<count {
        let u = Double.random(in: 0...1)
        
        // Custom cumulative distribution function (CDF)
        let x = minValue + (maxValue - minValue) * (1 - sqrt(1 - u))
        
        dataPoints.append(x)
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

/// Calculates the z-score for a given value
/// - Parameters:
///   - value: The value to calculate the z-score for
///   - mean: The mean of the distribution
///   - standardDeviation: The standard deviation of the distribution
/// - Returns: The z-score of the value
func calculateZScore(value: Double, mean: Double, standardDeviation: Double) -> Double {
    return (value - mean) / standardDeviation
}

/// Calculates the cumulative probability for a given value
/// - Parameters:
///   - value: The value to calculate the probability for
///   - minValue: The minimum value of the range
///   - maxValue: The maximum value of the range
/// - Returns: The cumulative probability
func cumulativeProbability(value: Double, minValue: Double, maxValue: Double) -> Double {
    // Custom cumulative distribution function (CDF)
    return 1 - sqrt(1 - (value - minValue) / (maxValue - minValue))
}

// Generate data points
let dataPoints = generateDataPoints(count: 1000, minValue: -10, maxValue: 10)

// Calculate mean and standard deviation
let mean = calculateMean(dataPoints)
let standardDeviation = calculateStandardDeviation(dataPoints)

print("Mean: \(mean)")
print("Standard Deviation: \(standardDeviation)")

// Example usage
let exampleValue = 9.9
let zScore = calculateZScore(value: Double(exampleValue), mean: mean, standardDeviation: standardDeviation)
let probability = cumulativeProbability(value: exampleValue, minValue: -10, maxValue: 10)

print("Z-score for \(exampleValue): \(zScore)")
print("Probability of a value <= \(exampleValue): \(probability)")