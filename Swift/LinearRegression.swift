import Foundation

// Function to calculate predicted win percentage
func predictWinPercentage(ortg: Double, drtg: Double) -> Double {
    // Coefficients derived from historical data analysis (example values)
    let intercept = 0.5
    let ortgCoefficient = 0.03
    let drtgCoefficient = -0.03
    
    // Linear regression formula
    let predictedWinPercentage = intercept + (ortgCoefficient * ortg) + (drtgCoefficient * drtg)
    
    // Ensure the win percentage is between 0 and 1
    return max(0, min(1, predictedWinPercentage))
}

// Example historical data for the Lakers
let historicalData: [(season: String, wlPercentage: Double, ortg: Double, drtg: Double)] = [
    ("2022-23", 0.573, 113.9, 112.8),
    ("2021-22", 0.402, 110.3, 113.3),
    ("2020-21", 0.583, 111.7, 107.1),
    // Add more historical data as needed
]

// Calculate the average ORtg and DRtg from historical data
let averageOrtg = historicalData.map { $0.ortg }.reduce(0, +) / Double(historicalData.count)
let averageDrtg = historicalData.map { $0.drtg }.reduce(0, +) / Double(historicalData.count)

// Predict the win percentage for the current season
let predictedWinPercentage = predictWinPercentage(ortg: averageOrtg, drtg: averageDrtg)

// Output the result
print("Predicted Win Percentage for the current season: \(predictedWinPercentage * 100)%")