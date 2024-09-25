import Foundation

struct Point: Hashable {
    let x: Int
    let y: Int
}

func method2(size: Int, points: [Point], lines: [(Point, Point)]) -> String {
    let actualSize = size % 2 == 0 ? size + 1 : size
    var grid = Array(repeating: Array(repeating: " ", count: actualSize), count: actualSize)
    let center = actualSize / 2
    
    // Draw axes
    for i in 0..<actualSize {
        grid[center][i] = "-"
        grid[i][center] = "|"
    }
    grid[center][center] = "+"
    
    // Draw points
    for point in points {
        let x = center + point.x
        let y = center - point.y
        if x >= 0 && x < actualSize && y >= 0 && y < actualSize {
            grid[y][x] = "o"
        }
    }
    
    // Draw lines
    for (start, end) in lines {
        drawLine(from: start, to: end, on: &grid, size: actualSize)
    }
    
    return grid.map { $0.joined() }.joined(separator: "\n")
}

private func drawLine(from start: Point, to end: Point, on grid: inout [[String]], size: Int) {
    let center = size / 2
    var x1 = start.x + center
    var y1 = center - start.y
    let x2 = end.x + center
    let y2 = center - end.y
    
    let dx = abs(x2 - x1)
    let dy = -abs(y2 - y1)
    let sx = x1 < x2 ? 1 : -1
    let sy = y1 < y2 ? 1 : -1
    var err = dx + dy
    
    while true {
        if x1 >= 0 && x1 < size && y1 >= 0 && y1 < size && grid[y1][x1] != "o" {
            grid[y1][x1] = "*"
        }
        
        if x1 == x2 && y1 == y2 { break }
        let e2 = 2 * err
        if e2 >= dy {
            err += dy
            x1 += sx
        }
        if e2 <= dx {
            err += dx
            y1 += sy
        }
    }
}

// Example usage
let result = method2(size: 21,
                     points: [Point(x: 3, y: 4), Point(x: -2, y: -1)],
                     lines: [(Point(x: 3, y: 4), Point(x: -2, y: -1))])
print(result)