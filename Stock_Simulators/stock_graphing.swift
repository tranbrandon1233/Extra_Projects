import Foundation

/// Simulates a Gaussian random walk for a stock price and creates an ASCII graph
/// - Parameters:
///   - initialPrice: The starting price of the stock
///   - periods: The number of time periods to simulate
///   - volatility: The annualized volatility of the stock (standard deviation of returns)
/// - Returns: A tuple containing the ASCII graph string and the final stock price
func simulateGaussianStockPrice(initialPrice: Double, periods: Int, volatility: Double) -> (graph: String, finalPrice: Double) {
    var prices: [Double] = [initialPrice]
    var graph = ""
    
    // Generate price changes using a normal distribution
    let dailyVolatility = volatility / sqrt(252.0) // Assuming 252 trading days per year
    for _ in 1..<periods {
        let change = Double.random(in: 0...1).gaussianInverse() * dailyVolatility
        let newPrice = prices.last! * exp(change)
        prices.append(newPrice)
    }
    
    // Find min and max prices for scaling
    let minPrice = prices.min()!
    let maxPrice = prices.max()!
    let priceRange = maxPrice - minPrice
    
    // Create ASCII graph
    for i in 0..<periods {
        let normalizedPrice = (prices[i] - minPrice) / priceRange
        let graphPosition = Int(normalizedPrice * 20) // Scale to 20 rows
        let line = String(repeating: " ", count: graphPosition) + (i == 0 ? "O" : (prices[i] > prices[i-1] ? "+" : (prices[i] < prices[i-1] ? "-" : "=")))
        graph += String(format: "%6.2f |", prices[i]) + line + "\n"
    }
    
    // Add x-axis
    graph += "       " + String(repeating: "-", count: periods) + "\n"
    graph += "       " + (0..<periods).map { String($0 % 10) }.joined()
    
    return (graph, prices.last!)
}

// Extension to generate standard normal random variables
extension Double {
    /// Generates a random number from a standard normal distribution using the Box-Muller transform
    func gaussianInverse() -> Double {
        let u1 = Double.random(in: 0...1)
        let u2 = Double.random(in: 0...1)
        return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    }
}

// Example usage
let (graph, finalPrice) = simulateGaussianStockPrice(initialPrice: 100, periods: 30, volatility: 0.2)
print(graph)
print("Final price: \(finalPrice)")