import Foundation

// Struct to represent season data
struct SeasonData {
    let year: String
    let winLossPercentage: Double
    let srs: Double
    let offensiveRating: Double
    let defensiveRating: Double
    let gamesPlayed: Int
    let playoffStatus: Bool
}

// Last 20 seasons of Lakers data (extracted from https://www.basketball-reference.com/teams/LAL/)
let lakersHistory: [SeasonData] = [
    SeasonData(year: "2024-25", winLossPercentage: 0.692, srs: 0.04, offensiveRating: 117.8, defensiveRating: 116.9, gamesPlayed: 13, playoffStatus: false),
    SeasonData(year: "2023-24", winLossPercentage: 0.573, srs: 1.07, offensiveRating: 115.9, defensiveRating: 115.3, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2022-23", winLossPercentage: 0.524, srs: 0.43, offensiveRating: 114.5, defensiveRating: 113.9, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2021-22", winLossPercentage: 0.402, srs: -3.08, offensiveRating: 110.3, defensiveRating: 113.3, gamesPlayed: 82, playoffStatus: false),
    SeasonData(year: "2020-21", winLossPercentage: 0.583, srs: 2.77, offensiveRating: 109.9, defensiveRating: 107.1, gamesPlayed: 72, playoffStatus: true),
    SeasonData(year: "2019-20", winLossPercentage: 0.732, srs: 6.28, offensiveRating: 112.0, defensiveRating: 106.3, gamesPlayed: 71, playoffStatus: true),
    SeasonData(year: "2018-19", winLossPercentage: 0.451, srs: -1.33, offensiveRating: 107.8, defensiveRating: 109.5, gamesPlayed: 82, playoffStatus: false),
    SeasonData(year: "2017-18", winLossPercentage: 0.427, srs: -1.44, offensiveRating: 106.5, defensiveRating: 108.0, gamesPlayed: 82, playoffStatus: false),
    SeasonData(year: "2016-17", winLossPercentage: 0.317, srs: -6.29, offensiveRating: 106.0, defensiveRating: 113.0, gamesPlayed: 82, playoffStatus: false),
    SeasonData(year: "2015-16", winLossPercentage: 0.207, srs: -8.92, offensiveRating: 101.6, defensiveRating: 111.6, gamesPlayed: 82, playoffStatus: false),
    SeasonData(year: "2014-15", winLossPercentage: 0.256, srs: -6.17, offensiveRating: 103.4, defensiveRating: 110.6, gamesPlayed: 82, playoffStatus: false),
    SeasonData(year: "2013-14", winLossPercentage: 0.329, srs: -5.33, offensiveRating: 104.2, defensiveRating: 110.6, gamesPlayed: 82, playoffStatus: false),
    SeasonData(year: "2012-13", winLossPercentage: 0.549, srs: 1.48, offensiveRating: 107.8, defensiveRating: 106.6, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2011-12", winLossPercentage: 0.621, srs: 1.96, offensiveRating: 106.0, defensiveRating: 104.4, gamesPlayed: 66, playoffStatus: true),
    SeasonData(year: "2010-11", winLossPercentage: 0.695, srs: 6.01, offensiveRating: 111.0, defensiveRating: 104.3, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2009-10", winLossPercentage: 0.695, srs: 4.78, offensiveRating: 108.8, defensiveRating: 103.7, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2008-09", winLossPercentage: 0.793, srs: 7.11, offensiveRating: 112.8, defensiveRating: 104.7, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2007-08", winLossPercentage: 0.695, srs: 7.34, offensiveRating: 113.0, defensiveRating: 105.5, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2006-07", winLossPercentage: 0.512, srs: 0.24, offensiveRating: 108.6, defensiveRating: 108.6, gamesPlayed: 82, playoffStatus: true),
    SeasonData(year: "2005-06", winLossPercentage: 0.549, srs: 2.53, offensiveRating: 108.4, defensiveRating: 105.7, gamesPlayed: 82, playoffStatus: true)
]

// Function to calculate averages
func calculateAverage(for metric: (SeasonData) -> Double) -> Double {
    let total = lakersHistory.reduce(0.0) { $0 + metric($1) }
    return total / Double(lakersHistory.count)
}

// Function to calculate playoff likelihood
func calculatePlayoffLikelihood() -> Double {
    let playoffSeasons = lakersHistory.filter { $0.playoffStatus }.count
    return Double(playoffSeasons) / Double(lakersHistory.count) * 100
}

// Predict performance for a given season
// Predict performance for a given season
func predictPerformance(games: Int) -> (predictedWins: Int, predictedLosses: Int, playoffLikelihood: Double) {
    let avgWinLossPercentage = calculateAverage { $0.winLossPercentage }
    let avgORtg = calculateAverage { $0.offensiveRating }
    let avgDRtg = calculateAverage { $0.defensiveRating }
    let avgSRS = calculateAverage { $0.srs }

    // Weighted formula for W/L%
    let adjustedWinLossPercentage = (avgWinLossPercentage * 0.5) +
                                     ((avgORtg - avgDRtg) / 100 * 0.3) +
                                     (avgSRS / 10 * 0.2)
    let predictedWins = Int(round(adjustedWinLossPercentage * Double(games)))
    let predictedLosses = games - predictedWins

    // Calculate playoff likelihood based on predicted W/L%
    let playoffLikelihood: Double
    switch adjustedWinLossPercentage {
    case 0.600...:
        playoffLikelihood = 90.0 // High chance of playoffs
    case 0.500..<0.600:
        playoffLikelihood = 70.0 // Moderate chance
    case 0.450..<0.500:
        playoffLikelihood = 40.0 // Low chance
    default:
        playoffLikelihood = 10.0 // Very low chance
    }

    return (predictedWins, predictedLosses, playoffLikelihood)
}

// Predict for an 82-game season
let prediction = predictPerformance(games: 82)
print("Predicted Wins: \(prediction.predictedWins)")
print("Predicted Losses: \(prediction.predictedLosses)")
print("Playoff Likelihood: \(String(format: "%.2f", prediction.playoffLikelihood))%")