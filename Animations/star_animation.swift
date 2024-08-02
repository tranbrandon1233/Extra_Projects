import Foundation

/// Creates an animation of a spinning star using ASCII art.
func createSpinningStar() {
    let starPoints = 5 // Number of points on the star
    let frames = 20 // Number of frames in the animation
    let delay: TimeInterval = 0.1 // Delay between frames in seconds
    
    // Precompute all frames of the star animation
    let starFrames = (0..<frames).map { i -> String in
        let angle = Double(i) / Double(frames) * 2 * Double.pi // Calculate the angle for the current frame
        return drawStar(points: starPoints, angle: angle) // Draw the star with the calculated angle
    }
    
    // Infinite loop to keep the animation running
    while true {
        for frame in starFrames {
            print("\u{001B}[2J\u{001B}[H\(frame)") // Clear the screen and print the current frame
            Thread.sleep(forTimeInterval: delay) // Wait for the specified delay before the next frame
        }
    }
}

/// Draws a star with the given number of points and rotation angle.
/// - Parameters:
///   - points: The number of points on the star.
///   - angle: The rotation angle of the star.
/// - Returns: A string representing the star in ASCII art.
func drawStar(points: Int, angle: Double) -> String {
    let radius: Double = 10 // Radius of the star
    var result = ""
    
    // Loop through each point in the grid
    for y in -Int(radius)...Int(radius) {
        for x in -Int(radius)...Int(radius) {
            let r = sqrt(Double(x*x + y*y)) // Calculate the distance from the center
            var a = atan2(Double(y), Double(x)) // Calculate the angle from the center
            a = (a + angle).truncatingRemainder(dividingBy: 2 * Double.pi) // Adjust the angle for rotation
            // Check if the point is on the star
            if r < radius && abs(a - 2 * Double.pi / Double(points) * round(a / (2 * Double.pi / Double(points)))) < 0.1 {
                result += "*" // Add a star point
            } else {
                result += " " // Add a space
            }
        }
        result += "\n" // New line after each row
    }
    return result
}

// Start the spinning star animation
createSpinningStar()