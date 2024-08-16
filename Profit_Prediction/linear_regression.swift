import Foundation

/// Struct to represent customer purchase data.
struct PurchaseData {
    /// Date of the record
    let date: Date
    /// Total amount of purchases on this date
    let totalPurchases: Double
    /// Number of transactions on this date
    let numberOfTransactions: Int
    /// Optional: Age of the customer
    let age: Int?
    /// Optional: Location of the customer
    let location: String?
}

/// Enum for handling errors related to data processing.
enum DataError: Error {
    case missingData
    case invalidDataFormat
    case outlierDetected
    case insufficientData
}

/// Parses a date string into a `Date` object.
/// - Parameter string: A string representing the date in "yyyy-MM-dd" format.
/// - Returns: A `Date` object if the format is valid; otherwise, `nil`.
func parseDate(from string: String) -> Date? {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd"
    return formatter.date(from: string)
}

/// Validates purchase data and handles outliers.
/// - Parameters:
///   - totalPurchases: The total purchase amount.
///   - numberOfTransactions: The number of transactions.
/// - Throws: `DataError.outlierDetected` if outliers are detected.
func validateData(totalPurchases: Double, numberOfTransactions: Int) throws {
    // Validate total purchases
    guard totalPurchases >= 0 else {
        throw DataError.outlierDetected
    }
    
    // Validate number of transactions
    guard numberOfTransactions > 0 else {
        throw DataError.outlierDetected
    }
}

/// Predicts future purchases using linear regression based on historical data.
/// - Parameters:
///   - pastData: An array of tuples where each tuple contains past total purchases and number of transactions.
/// - Returns: A predicted future purchase amount, or `nil` if insufficient data.
func predictFuturePurchases(pastData: [(totalPurchases: Double, numberOfTransactions: Int)]) -> Double? {
    guard pastData.count > 1 else { return nil } // Ensure enough data points for regression
    
    // Prepare data for linear regression
    let xs = pastData.map { Double($0.numberOfTransactions) }
    let ys = pastData.map { $0.totalPurchases }
    
    // Perform linear regression to fit a line to the data
    let model = LinearRegression(xs: xs, ys: ys)
    let futureTransactions = xs.max()! + 10 // Predict for future transactions
    return model.predict(x: futureTransactions)
}

/// Linear regression model to predict future values.
struct LinearRegression {
    private let slope: Double
    private let intercept: Double
    
    init(xs: [Double], ys: [Double]) {
        // Compute means
        let meanX = xs.reduce(0, +) / Double(xs.count)
        let meanY = ys.reduce(0, +) / Double(ys.count)
        
        // Compute slope and intercept
        let numerator = zip(xs, ys).map { ($0 - meanX) * ($1 - meanY) }.reduce(0, +)
        let denominator = xs.map { ($0 - meanX) * ($0 - meanX) }.reduce(0, +)
        
        slope = numerator / denominator
        intercept = meanY - slope * meanX
    }
    
    func predict(x: Double) -> Double {
        return slope * x + intercept
    }
}

// Example data for testing
let pastData = [(totalPurchases: 1.0, numberOfTransactions: 1),
                (totalPurchases: 2.0, numberOfTransactions: 2),
                (totalPurchases: 3.0, numberOfTransactions: 3)]

do {
    // Validate the input data
    if pastData.isEmpty {
        throw DataError.missingData
    }
    
    // Predict future purchases with the past data
    if let futurePurchases = predictFuturePurchases(pastData: pastData) {
        print("Predicted future purchases: $\(futurePurchases)")
    } else {
        print("Error: Insufficient data for prediction")
    }
} catch DataError.missingData {
    print("Error: Missing required data")
} catch DataError.outlierDetected {
    print("Error: Detected outlier in data")
} catch DataError.insufficientData {
    print("Error: Insufficient data for prediction")
} catch {
    print("Unexpected error: \(error)")
}