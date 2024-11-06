import Foundation

// Define infinity for unreachable paths
let maxsize = Double.infinity

// Number of cities
let N = 4

// Array to store the final minimum cost path
var finalPath = [Int](repeating: -1, count: N + 1)

// Stores the minimum cost of the path
var finalRes = maxsize

// Function to copy the current path solution to the final solution
func copyToFinal(_ currPath: [Int]) {
    for i in 0..<N {
        finalPath[i] = currPath[i]
    }
    finalPath[N] = currPath[0]  // Complete the cycle by returning to the starting city
}

// Function to find the minimum edge cost ending at vertex `i`
func firstMin(_ adj: [[Double]], _ i: Int) -> Double {
    var min = maxsize
    for k in 0..<N where adj[i][k] < min && i != k {
        min = adj[i][k]
    }
    return min
}

// Function to find the second minimum edge cost ending at vertex `i`
func secondMin(_ adj: [[Double]], _ i: Int) -> Double {
    var first = maxsize
    var second = maxsize
    for j in 0..<N where i != j {
        if adj[i][j] <= first {
            second = first
            first = adj[i][j]
        } else if adj[i][j] <= second && adj[i][j] != first {
            second = adj[i][j]
        }
    }
    return second
}

// Recursive function to explore potential solutions for TSP using Branch and Bound
func TSPRec(_ adj: [[Double]], _ currBound: Double, _ currWeight: Double, _ level: Int, _ currPath: inout [Int], _ visited: inout [Bool]) {
    // Base case: when all cities are visited, check if we can complete the cycle
    if level == N {
        if adj[currPath[level - 1]][currPath[0]] != 0 {  // Check if return edge exists
            let currRes = currWeight + adj[currPath[level - 1]][currPath[0]]
            if currRes < finalRes {
                copyToFinal(currPath)  // Update final path and result
                finalRes = currRes
            }
        }
        return
    }
    
    // Explore next city options from the current city in the path
    for i in 0..<N where adj[currPath[level - 1]][i] != 0 && !visited[i] {
        var temp = currBound
        var currentWeight = currWeight + adj[currPath[level - 1]][i]
        
        // Calculate the new bound for the current path at level 1 or other levels
        if level == 1 {
            temp -= (firstMin(adj, currPath[level - 1]) + firstMin(adj, i)) / 2
        } else {
            temp -= (secondMin(adj, currPath[level - 1]) + firstMin(adj, i)) / 2
        }
        
        // If the new path's lower bound is promising, continue exploring it
        if temp + currentWeight < finalRes {
            currPath[level] = i
            visited[i] = true
            
            // Recursive call to explore further levels
            TSPRec(adj, temp, currentWeight, level + 1, &currPath, &visited)
        }
        
        // Backtrack: undo changes to currWeight, currBound, and visited array
        currentWeight -= adj[currPath[level - 1]][i]
        temp = currBound
        visited = [Bool](repeating: false, count: N)
        for j in 0..<level where currPath[j] != -1 {
            visited[currPath[j]] = true
        }
    }
}

// Function to initiate the TSP solution process
func TSP(_ adj: [[Double]]) {
    // Initialize bound, path, and visited arrays
    var currBound = 0.0
    var currPath = [Int](repeating: -1, count: N + 1)
    var visited = [Bool](repeating: false, count: N)
    
    // Calculate the initial bound based on minimum edges for each city
    for i in 0..<N {
        currBound += (firstMin(adj, i) + secondMin(adj, i))
    }
    currBound = ceil(currBound / 2)  // Round off to ensure integer bound
    
    // Set starting city and initiate recursive search
    visited[0] = true
    currPath[0] = 0
    
    TSPRec(adj, currBound, 0, 1, &currPath, &visited)
}

// Adjacency matrix representing the travel costs between cities
let adj = [
    [0.0, 10.0, 15.0, 20.0],
    [10.0, 0.0, 35.0, 25.0],
    [15.0, 35.0, 0.0, 30.0],
    [20.0, 25.0, 30.0, 0.0]
]

// Run the TSP algorithm and display the results
TSP(adj)
print("Minimum cost:", finalRes)
print("Path Taken:", finalPath)