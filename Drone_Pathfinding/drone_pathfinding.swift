import Foundation

struct Coordinate: Hashable {
    let x: Int
    let y: Int
    let z: Int
}

struct EnvironmentData {
    let buildingHeights: [[Int]]
    let noFlyZones: [Coordinate]
    let weatherConditions: [String: Double]
}

struct DroneState {
    let position: Coordinate
    let batteryLife: Double
    let payloadWeight: Double
}

func findOptimalPath(start: Coordinate, destination: Coordinate, droneState: DroneState, environment: EnvironmentData) -> [Coordinate] {
    var openSet = Set<Coordinate>([start])
    var cameFrom = [Coordinate: Coordinate]()
    var gScore = [start: 0.0]
    var fScore = [start: heuristicCost(start, destination)]
    
    let maxAltitude = 120 // Maximum allowed altitude in meters
    
    while !openSet.isEmpty {
        let current = openSet.min(by: { fScore[$0, default: .infinity] < fScore[$1, default: .infinity] })!
        
        if current == destination {
            return reconstructPath(cameFrom, current)
        }
        
        openSet.remove(current)
        
        for dx in -1...1 {
            for dy in -1...1 {
                for dz in -1...1 {
                    let neighbor = Coordinate(x: current.x + dx, y: current.y + dy, z: current.z + dz)
                    
                    if isValidMove(neighbor, environment: environment, maxAltitude: maxAltitude) {
                        let tentativeGScore = gScore[current, default: .infinity] + calculateMoveCost(from: current, to: neighbor, destination: destination, droneState: droneState, environment: environment)
                        
                        if tentativeGScore < gScore[neighbor, default: .infinity] {
                            cameFrom[neighbor] = current
                            gScore[neighbor] = tentativeGScore
                            fScore[neighbor] = tentativeGScore + heuristicCost(neighbor, destination)
                            
                            if !openSet.contains(neighbor) {
                                openSet.insert(neighbor)
                            }
                        }
                    }
                }
            }
        }
    }
    
    return [] // No path found
}

func reconstructPath(_ cameFrom: [Coordinate: Coordinate], _ current: Coordinate) -> [Coordinate] {
    var totalPath = [current]
    var currentNode = current
    
    while let parent = cameFrom[currentNode] {
        totalPath.insert(parent, at: 0)
        currentNode = parent
    }
    
    return totalPath
}

func isValidMove(_ position: Coordinate, environment: EnvironmentData, maxAltitude: Int) -> Bool {
    if position.x < 0 || position.y < 0 || position.z < 0 ||
       position.x >= environment.buildingHeights.count ||
       position.y >= environment.buildingHeights[0].count ||
       position.z > maxAltitude {
        return false
    }
    
    if position.z <= environment.buildingHeights[position.x][position.y] {
        return false
    }
    
    if environment.noFlyZones.contains(where: { $0.x == position.x && $0.y == position.y }) {
        return false
    }
    
    return true
}

func calculateMoveCost(from current: Coordinate, to next: Coordinate, destination: Coordinate, droneState: DroneState, environment: EnvironmentData) -> Double {
    let distance = sqrt(pow(Double(next.x - current.x), 2) +
                        pow(Double(next.y - current.y), 2) +
                        pow(Double(next.z - current.z), 2))
    
    let altitudeCost = Double(next.z) / 10.0 // Higher altitudes cost more
    let windSpeed = environment.weatherConditions["windSpeed"] ?? 0
    let windCost = windSpeed * Double(abs(next.z - current.z)) / 10.0
    
    return distance + altitudeCost + windCost
}

func heuristicCost(_ current: Coordinate, _ destination: Coordinate) -> Double {
    return sqrt(pow(Double(destination.x - current.x), 2) +
                pow(Double(destination.y - current.y), 2) +
                pow(Double(destination.z - current.z), 2))
}

// Test the algorithm
let path = findOptimalPath(
    start: Coordinate(x: 0, y: 0, z: 10),
    destination: Coordinate(x: 5, y: 5, z: 10),
    droneState: DroneState(position: Coordinate(x: 0, y: 0, z: 10), batteryLife: 100.0, payloadWeight: 2.0),
    environment: EnvironmentData(
        buildingHeights: [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 20, 20, 0, 0],
            [0, 0, 20, 20, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ],
        noFlyZones: [Coordinate(x: 2, y: 2, z: 0)],
        weatherConditions: ["windSpeed": 5.0]
    )
)

for coord in path {
    print("(\(coord.x), \(coord.y), \(coord.z))")
}