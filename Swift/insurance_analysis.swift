import Foundation

// Function to read input and return as a String
func readInput(prompt: String) -> String {
    print(prompt, terminator: "")
    return readLine() ?? ""
}

// Function to read and parse input as a Double
func readDoubleInput(prompt: String) -> Double? {
    print(prompt, terminator: "")
    if let input = readLine(), let value = Double(input) {
        return value
    }
    return nil
}

// Function to read and parse input as an Int
func readIntInput(prompt: String) -> Int? {
    print(prompt, terminator: "")
    if let input = readLine(), let value = Int(input) {
        return value
    }
    return nil
}

// Function to calculate risk distribution
func calculateRiskDistribution(policies: [(String, Double)]) -> [String: Double] {
    var riskDistribution: [String: Double] = [:]
    for policy in policies {
        riskDistribution[policy.0, default: 0.0] += policy.1
    }
    return riskDistribution
}

// Function to calculate capital allocation
func calculateCapitalAllocation(riskDistribution: [String: Double], totalRisk: Double, totalCapital: Double) -> [String: Double] {
    var capitalAllocation: [String: Double] = [:]
    for (policyType, risk) in riskDistribution {
        capitalAllocation[policyType] = (risk / totalRisk) * totalCapital
    }
    return capitalAllocation
}

// Function to generate risk adjustment suggestions
func generateSuggestions(riskDistribution: [String: Double], totalRisk: Double, maxRiskThreshold: Double) -> [String] {
    var suggestions: [String] = []
    for (policyType, risk) in riskDistribution {
        if risk / totalRisk > maxRiskThreshold {
            suggestions.append("Reduce exposure to \(policyType) policies.")
        } else if risk / totalRisk < 0.05 {
            suggestions.append("Consider increasing exposure to \(policyType) policies.")
        }
    }
    return suggestions
}

// Function to check compliance with regulatory requirements
func checkCompliance(riskDistribution: [String: Double], totalRisk: Double, maxIndividualRisk: Double) -> Bool {
    for (_, risk) in riskDistribution {
        if risk / totalRisk > maxIndividualRisk {
            return false
        }
    }
    return true
}

// Main function
func main() {
    guard let numberOfPolicies = readIntInput(prompt: "Enter the number of policies: ") else {
        print("Invalid number of policies input.")
        return
    }

    var policies: [(String, Double)] = []

    for _ in 1...numberOfPolicies {
        let policyType = readInput(prompt: "Enter policy type: ")
        guard let risk = readDoubleInput(prompt: "Enter policy risk (as a percentage): ") else {
            print("Invalid risk input.")
            continue
        }
        policies.append((policyType, risk))
    }

    let totalRisk = policies.reduce(0.0) { $0 + $1.1 }
    let riskDistribution = calculateRiskDistribution(policies: policies)
    let averageRisk = totalRisk / Double(numberOfPolicies)

    print("Total Risk: \(totalRisk)")
    print("Average Risk: \(averageRisk)")
    print("Risk Distribution: \(riskDistribution)")

    let totalCapital = 1000000.0 // 1 million
    let capitalAllocation = calculateCapitalAllocation(riskDistribution: riskDistribution, totalRisk: totalRisk, totalCapital: totalCapital)

    print("Capital Allocation: \(capitalAllocation)")

    let suggestions = generateSuggestions(riskDistribution: riskDistribution, totalRisk: totalRisk, maxRiskThreshold: 0.2)
    print("Suggestions: \(suggestions)")

    let complianceFlag = checkCompliance(riskDistribution: riskDistribution, totalRisk: totalRisk, maxIndividualRisk: 0.25)
    if complianceFlag {
        print("Portfolio is compliant with regulatory requirements.")
    } else {
        print("Portfolio is not compliant with regulatory requirements.")
    }
}

main()