import Foundation

/// Represents a company with its financial data and calculated score
struct Company {
    let name: String
    let profit: Double
    let debt: Double
    let peRatio: Double
    var score: Double = 0
}

/// Calculates scores for companies based on their financial data and ranks them
/// - Parameters:
///   - companies: Array of Company structs containing financial data
/// - Returns: Array of Company structs sorted by their calculated scores in descending order
func rankCompanies(_ companies: [Company]) -> [Company] {
    let maxProfit = companies.map { $0.profit }.max() ?? 0
    let minDebt = companies.map { $0.debt }.min() ?? 0
    let minPERatio = companies.map { $0.peRatio }.min() ?? 0
    
    // Calculate scores for each company
    var rankedCompanies = companies.map { company in
        var rankedCompany = company
        
        // Normalize and calculate scores
        rankedCompany.score += (company.profit / maxProfit) * 100
        rankedCompany.score += (1 - (company.debt - minDebt) / (companies.map { $0.debt }.max()! - minDebt)) * 100
        rankedCompany.score += (1 - (company.peRatio - minPERatio) / (companies.map { $0.peRatio }.max()! - minPERatio)) * 100
        
        return rankedCompany
    }
    
    // Sort companies by score in descending order
    rankedCompanies.sort { $0.score > $1.score }
    
    return rankedCompanies
}

// Test cases
let testCompanies = [
    Company(name: "Company A", profit: 1000000, debt: 500000, peRatio: 15),
    Company(name: "Company B", profit: 2000000, debt: 750000, peRatio: 12),
    Company(name: "Company C", profit: 1500000, debt: 250000, peRatio: 18)
]

let rankedCompanies = rankCompanies(testCompanies)

// Print results
for (index, company) in rankedCompanies.enumerated() {
    print("\(index + 1). \(company.name) - Score: \(company.score)")
}