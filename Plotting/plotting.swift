import Foundation

/// A structure representing a point in 2D space.
struct Point {
    let x: Double
    let y: Double
}

/// Draws a coordinate plane with given points and dimensions, and calculates and draws the line of best fit.
/// - Parameters:
///   - points: An array of `Point` structures representing the points to be plotted.
///   - width: The width of the coordinate plane.
///   - height: The height of the coordinate plane.
func drawCoordinatePlane(points: [Point], width: Int, height: Int) {
    // Check for valid width and height
    guard width > 0, height > 0 else {
        print("Error: Width and height must be positive integers.")
        return
    }
    
    // Check for non-empty points array
    guard !points.isEmpty else {
        print("Error: Points array cannot be empty.")
        return
    }
    
    // Initialize the plane with empty spaces
    var plane = Array(repeating: Array(repeating: " ", count: width), count: height)
    
    // Draw the vertical axis
    for i in 0..<height {
        plane[i][width/2] = "│"
    }

    // Draw the horizontal axis
    for i in 0..<width {
        plane[height/2][i] = "─"
    }
    // Draw the origin
    plane[height/2][width/2] = "┼"
    
    // Plot the points on the plane
    for point in points {
        let x = Int(point.x * Double(width/2) / 10.0) + width/2
        let y = height/2 - Int(point.y * Double(height/2) / 10.0)
        if x >= 0 && x < width && y >= 0 && y < height {
            plane[y][x] = "●"
        }
    }
    
    // Calculate the line of best fit using the least squares method
    let n = Double(points.count)
    let sumX = points.reduce(0.0) { $0 + $1.x }
    let sumY = points.reduce(0.0) { $0 + $1.y }
    let sumXY = points.reduce(0.0) { $0 + $1.x * $1.y }
    let sumX2 = points.reduce(0.0) { $0 + $1.x * $1.x }
    
    let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
    let intercept = (sumY - slope * sumX) / n
    
    // Draw the line of best fit on the plane
    for x in 0..<width {
        let realX = (Double(x) - Double(width/2)) * 10.0 / Double(width/2)
        let y = slope * realX + intercept
        let planeY = height/2 - Int(y * Double(height/2) / 10.0)
        if planeY >= 0 && planeY < height && plane[planeY][x] == " " {
            plane[planeY][x] = "·"
        }
    }
    
    // Print the plane
    for row in plane {
        print(row.joined())
    }
    
    // Print the equation of the line of best fit
    print("Line of best fit: y = \(slope)x + \(intercept)")
}

// Example usage
let points = [
    Point(x: -8, y: -8),
    Point(x: -6, y: -6),
    Point(x: -4, y: -2),
    Point(x: -2, y: -1),
    Point(x: 0, y: 1),
    Point(x: 2, y: 3),
    Point(x: 4, y: 6),
    Point(x: 6, y: 8),
    Point(x: 8, y: 10)
]

drawCoordinatePlane(points: points, width: 61, height: 31)
