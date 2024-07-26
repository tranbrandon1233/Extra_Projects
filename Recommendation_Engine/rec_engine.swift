import Foundation

struct UserInteraction {
    let userId: String
    let productId: String
    let interactionType: String // "view", "purchase", "rating"
    let rating: Double? // Only for "rating" interactions
}

// Create the user interactions with the products
let userInteractions: [UserInteraction] = [
    UserInteraction(userId: "user1", productId: "product1", interactionType: "rating", rating: 4.0),
    UserInteraction(userId: "user1", productId: "product2", interactionType: "rating", rating: 5.0),
    UserInteraction(userId: "user2", productId: "product1", interactionType: "rating", rating: 5.0),
    UserInteraction(userId: "user2", productId: "product3", interactionType: "rating", rating: 3.0),
    UserInteraction(userId: "user3", productId: "product2", interactionType: "rating", rating: 4.0),
    UserInteraction(userId: "user3", productId: "product3", interactionType: "rating", rating: 2.0),
    UserInteraction(userId: "user4", productId: "product1", interactionType: "rating", rating: 3.0)
]

// Create a user-item matrix from the interactions
func createUserItemMatrix(interactions: [UserInteraction]) -> [String: [String: Double]] {
    var matrix: [String: [String: Double]] = [:]
    
    for interaction in interactions {
        if let rating = interaction.rating {
            if matrix[interaction.userId] == nil {
                matrix[interaction.userId] = [:]
            }
            matrix[interaction.userId]?[interaction.productId] = rating
        }
    }
    
    return matrix
}

// Calculate the cosine similarity between two users
func cosineSimilarity(user1: [String: Double], user2: [String: Double]) -> Double {
    let commonItems = Set(user1.keys).intersection(Set(user2.keys))
    guard !commonItems.isEmpty else { return 0.0 }
    
    var dotProduct = 0.0
    var normA = 0.0
    var normB = 0.0
    
    for item in commonItems {
        dotProduct += user1[item]! * user2[item]!
        normA += pow(user1[item]!, 2)
        normB += pow(user2[item]!, 2)
    }
    
    return dotProduct / (sqrt(normA) * sqrt(normB))
}

// Recommend top N products for a user based on collaborative filtering
func recommendProducts(userId: String, matrix: [String: [String: Double]], topN: Int) -> [String] {
    guard let userRatings = matrix[userId] else { return [] }
    
    var similarityScores: [(String, Double)] = []
    for (otherUserId, otherUserRatings) in matrix where otherUserId != userId {
        let similarity = cosineSimilarity(user1: userRatings, user2: otherUserRatings)
        similarityScores.append((otherUserId, similarity))
    }
    
    let sortedScores = similarityScores.sorted { $0.1 > $1.1 }
    var recommendedProducts: [String] = []
    
    for (otherUserId, _) in sortedScores {
        let otherUserRatings = matrix[otherUserId]!
        for (productId, _) in otherUserRatings where userRatings[productId] == nil {
            if !recommendedProducts.contains(productId) {
                recommendedProducts.append(productId)
                if recommendedProducts.count >= topN {
                    return recommendedProducts
                }
            }
        }
    }
    
    return recommendedProducts
}

// Recommend top N products for a user from a cold start
func recommendProductsWithColdStart(userId: String, matrix: [String: [String: Double]], topN: Int) -> [String] {
    // Check if the user is new
    if matrix[userId] == nil {
        // Return popular products or ask for onboarding information
        return ["popularProduct1", "popularProduct2", "popularProduct3"]
    }
    
    return recommendProducts(userId: userId, matrix: matrix, topN: topN)
}

let userItemMatrix = createUserItemMatrix(interactions: userInteractions)
print(userItemMatrix)

let recommendationsForUser1 = recommendProducts(userId: "user1", matrix: userItemMatrix, topN: 2)
print(recommendationsForUser1)

let recommendationsForNewUser = recommendProductsWithColdStart(userId: "newUser", matrix: userItemMatrix, topN: 3)
print(recommendationsForNewUser)

let recommendationsForExistingUser = recommendProductsWithColdStart(userId: "user3", matrix: userItemMatrix, topN: 3)
print(recommendationsForExistingUser)
