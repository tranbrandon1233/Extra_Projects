import Foundation

// Function to perform compound interest calculation
func calculateCompoundInterest(principal: Double, rate: Double, years: Int, compoundingFrequency: Int) -> (Double, Double, Double) {
    let ratePerPeriod = rate / Double(compoundingFrequency)
    let periods = years * compoundingFrequency
    let amount = principal * pow((1 + ratePerPeriod), Double(periods))
    let totalContributions = principal
    let totalInterest = amount - totalContributions
    return (amount, totalContributions, totalInterest)
}

// Function to format currency
func formatCurrency(value: Double, currencySymbol: String) -> String {
    let formatter = NumberFormatter()
    formatter.numberStyle = .currency
    formatter.currencySymbol = currencySymbol
    return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
}

// Main Program
var calculations: [(String, String, String)] = []  // Keep track of all calculations

print("Compound Interest Calculator\n")

repeat {
    print("Enter Principal Amount:")
    let principal = Double(readLine()!) ?? 0.0

    print("Enter Annual Interest Rate (as a percentage):")
    let rate = (Double(readLine()!) ?? 0.0) / 100

    print("Enter Number of Years:")
    let years = Int(readLine()!) ?? 0

    print("Enter Number of Times Interest is Compounded per Year:")
    let compoundingFrequency = Int(readLine()!) ?? 1

    print("Enter Currency Symbol (e.g., $, €, £):")
    let currencySymbol = readLine() ?? "$"

    let (futureValue, totalContributions, totalInterest) = calculateCompoundInterest(
        principal: principal,
        rate: rate,
        years: years,
        compoundingFrequency: compoundingFrequency
    )

    let formattedFutureValue = formatCurrency(value: futureValue, currencySymbol: currencySymbol)
    let formattedContributions = formatCurrency(value: totalContributions, currencySymbol: currencySymbol)
    let formattedInterest = formatCurrency(value: totalInterest, currencySymbol: currencySymbol)

    print("\nResults:")
    print("Future Value: \(formattedFutureValue)")
    print("Total Contributions: \(formattedContributions)")
    print("Total Interest Earned: \(formattedInterest)\n")

    calculations.append((formattedFutureValue, formattedContributions, formattedInterest))

    print("Would you like to perform another calculation? (y/n):")
} while readLine()?.lowercased() == "y"

print("\nSession Summary:")
for (index, calculation) in calculations.enumerated() {
    print("Calculation #\(index + 1):")
    print("Future Value: \(calculation.0), Total Contributions: \(calculation.1), Total Interest Earned: \(calculation.2)\n")
}