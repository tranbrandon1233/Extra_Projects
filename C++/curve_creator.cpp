#include <iostream>
#include <vector>
#include <random>
#include <ctime>

const int WIDTH = 20;  // Width of the grid
const int HEIGHT = 20; // Height of the grid

// Structure to represent a point in 2D space
struct Point {
    int x, y;
};

int main() {
    std::srand(std::time(nullptr)); // Seed the random number generator with the current time
    auto grid = std::vector<std::vector<char>>(HEIGHT, std::vector<char>(WIDTH, ' ')); // Create the grid
    
    Point current = {0, HEIGHT / 2}; // Start point in the middle of the leftmost column
    grid[current.y][current.x] = '#'; // Mark the start point on the grid
    
    int yDirection = 0; // Direction of movement along the y-axis
    int ySteps = 0; // Number of steps taken in the current y-direction
    
    // Loop until the curve reaches the rightmost column
    while (current.x < WIDTH - 1) {
        int move = std::rand() % 3; // Randomly decide the next move: 0: right, 1: up, 2: down
        
        // Move right if the move is 0 or if the current point is at the top or bottom edge
        if (move == 0 || current.y == 0 || current.y == HEIGHT - 1) {
            current.x++;
        } 
        // Move up if the move is 1 and the current direction is not down
        else if (move == 1 && yDirection <= 0) {
            current.y--;
            yDirection = -1;
            ySteps++;
        } 
        // Move down if the move is 2 and the current direction is not up
        else if (move == 2 && yDirection >= 0) {
            current.y++;
            yDirection = 1;
            ySteps++;
        }
        
        // Reset the y-direction and y-steps if the curve has moved sufficiently in the current y-direction
        if (ySteps == 2 * yDirection) {
            yDirection = 0;
            ySteps = 0;
        }
        
        grid[current.y][current.x] = '#'; // Mark the current point on the grid
    }
    
    // Print the grid
    for (const auto& row : grid) {
        for (char cell : row) {
            std::cout << cell;
        }
        std::cout << std::endl;
    }
    return 0;
}