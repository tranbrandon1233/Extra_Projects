#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

// Simple test framework
#define TEST(name) void name()
#define ASSERT(condition) if (!(condition)) { std::cerr << "Assertion failed: " << #condition << " in " << __FUNCTION__ << " at line " << __LINE__ << std::endl; assert(condition); }

/**
 * @brief A 2D vector class for representing positions and velocities.
 */
class Vector2 {
public:
    float x, y;

    Vector2() : x(0), y(0) {}
    Vector2(float x, float y) : x(x), y(y) {}

    // Vector addition
    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }

    // Vector subtraction
    Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }

    // Scalar multiplication
    Vector2 operator*(float scalar) const {
        return Vector2(x * scalar, y * scalar);
    }

    // Array subscript operator
    float& operator[](int index) {
        return index == 0 ? x : y;
    }

    const float& operator[](int index) const {
        return index == 0 ? x : y;
    }
};

/**
 * @brief Represents an obstacle in the game world.
 */
class Obstacle {
public:
    Vector2 position;
    float width, height;
    int number;

    Obstacle(const Vector2& pos, float w, float h, int num)
        : position(pos), width(w), height(h), number(num) {}

    /**
     * @brief Checks if the ball collides with the obstacle and decrements the number if so.
     * @param ballPosition The position of the ball.
     * @param radius The radius of the ball.
     * @return True if the number was decremented, false otherwise.
     */
    bool checkAndDecrementNumber(const Vector2& ballPosition, float radius) {
        // Check if the ball is colliding with the obstacle
        float minX = position.x - width / 2;
        float maxX = position.x + width / 2;
        float minY = position.y - height / 2;
        float maxY = position.y + height / 2;

        if (ballPosition.x + radius >= minX && ballPosition.x - radius <= maxX &&
            ballPosition.y + radius >= minY && ballPosition.y - radius <= maxY) {
            // Collision detected, decrement the number
            number--;
            return true;
        }
        return false;
    }
};

// Global score variable
int score = 0;

/**
 * @brief Updates the position of a moving ball, handling collisions with obstacles and boundaries.
 * @param ballPosition The current position of the ball.
 * @param ballVelocity The current velocity of the ball.
 * @param obstacles A vector of pointers to obstacles in the game world.
 * @param radius The radius of the ball.
 */
void updateBallPosition(Vector2& ballPosition, Vector2& ballVelocity, std::vector<Obstacle*>& obstacles, float radius) {
    Vector2 nextPosition = ballPosition + ballVelocity;

    // Check collisions with obstacles
    for (size_t i = 0; i < obstacles.size(); i++) {
        Obstacle* obstacle = obstacles[i];
        float minX = obstacle->position.x - obstacle->width / 2;
        float maxX = obstacle->position.x + obstacle->width / 2;
        float minY = obstacle->position.y - obstacle->height / 2;
        float maxY = obstacle->position.y + obstacle->height / 2;

        if (nextPosition.x - radius <= maxX && nextPosition.x + radius >= minX &&
            nextPosition.y - radius <= maxY && nextPosition.y + radius >= minY) {
            // Check if the ball collided with the obstacle and decrement the number
            bool numberDecremented = obstacle->checkAndDecrementNumber(nextPosition, radius);
            if (numberDecremented) {
                // Increment the score if the obstacle's number was decremented
                score++;
            }

            // Calculate the overlap between the ball and the obstacle
            float overlapX = std::min(nextPosition.x + radius, maxX) - std::max(nextPosition.x - radius, minX);
            float overlapY = std::min(nextPosition.y + radius, maxY) - std::max(nextPosition.y - radius, minY);

            // Move the ball out of the obstacle
            if (overlapX > overlapY) {
                ballPosition.y -= overlapY * (nextPosition.y > obstacle->position.y ? 1 : -1);
            }
            else {
                ballPosition.x -= overlapX * (nextPosition.x > obstacle->position.x ? 1 : -1);
            }

            // Update the ball velocity
            Vector2 normal;
            if (overlapX > overlapY) {
                normal = Vector2(0, (nextPosition.y > obstacle->position.y) ? 1 : -1);
            }
            else {
                normal = Vector2((nextPosition.x > obstacle->position.x) ? 1 : -1, 0);
            }

            float dotProduct = ballVelocity.x * normal.x + ballVelocity.y * normal.y;
            ballVelocity.x -= 2 * dotProduct * normal.x;
            ballVelocity.y -= 2 * dotProduct * normal.y;
        }
    }

    // Handle collision with the bottom boundary
    if (nextPosition.y - radius <= -260) {
        ballPosition.y = -260 + radius + 34;
        ballVelocity = Vector2(0, 0);
    }
    else {
        // Handle collision with left and right boundaries
        if (nextPosition.x - radius <= -220 || nextPosition.x + radius >= 220) {
            ballVelocity.x = -ballVelocity.x;
            nextPosition.x = ballPosition.x + ballVelocity.x;
        }

        // Handle collision with the top boundary
        if (nextPosition.y + radius >= 260) {
            ballVelocity.y = -ballVelocity.y;
            nextPosition.y = ballPosition.y + ballVelocity.y;
        }

        ballPosition = nextPosition;
    }
}


// Utility function for floating-point comparison
bool nearlyEqual(float a, float b, float epsilon = 0.0001f) {
    return std::abs(a - b) <= epsilon;
}

// Test cases for Vector2 class
TEST(testVector2Operations) {
    Vector2 v1(3, 4);
    Vector2 v2(1, 2);

    // Test addition
    Vector2 sum = v1 + v2;
    ASSERT(nearlyEqual(sum.x, 4) && nearlyEqual(sum.y, 6));

    // Test subtraction
    Vector2 diff = v1 - v2;
    ASSERT(nearlyEqual(diff.x, 2) && nearlyEqual(diff.y, 2));

    // Test scalar multiplication
    Vector2 scaled = v1 * 2;
    ASSERT(nearlyEqual(scaled.x, 6) && nearlyEqual(scaled.y, 8));

    // Test left-hand scalar multiplication
    Vector2 leftScaled =  v1 * 2;
    ASSERT(nearlyEqual(leftScaled.x, 6) && nearlyEqual(leftScaled.y, 8));

    // Test array subscript operator
    ASSERT(nearlyEqual(v1[0], 3) && nearlyEqual(v1[1], 4));
}

// Test cases for Obstacle class
TEST(testObstacleCollision) {
    Obstacle obstacle(Vector2(0, 0), 10, 10, 1);

    // Test collision detection
    ASSERT(obstacle.checkAndDecrementNumber(Vector2(4, 4), 1));
    ASSERT(obstacle.number == 0);

    // Test no collision
    ASSERT(!obstacle.checkAndDecrementNumber(Vector2(10, 10), 1));
    ASSERT(obstacle.number == 0);
}

// Test cases for updateBallPosition function
TEST(testUpdateBallPosition) {
    Vector2 ballPosition(0, 0);
    Vector2 ballVelocity(1, 1);
    std::vector<Obstacle*> obstacles;
    obstacles.push_back(new Obstacle(Vector2(5, 5), 2, 2, 1));
    float radius = 0.5f;

    // Test movement without collision
    updateBallPosition(ballPosition, ballVelocity, obstacles, radius);
    std::cout << "Velocity: X: " << ballVelocity.x << ", Y: " << ballVelocity.y << std::endl; // Velocity should be reflected
    std::cout << "Position: X: " << ballPosition.x << ", Y: " << ballPosition.y << std::endl; // Velocity should be reflected

    // Test collision with obstacle
    ballPosition = Vector2(4, 4);
    updateBallPosition(ballPosition, ballVelocity, obstacles, radius);
    std::cout << "Velocity: X: " << ballVelocity.x << ", Y: " << ballVelocity.y << std::endl; // Velocity should be reflected
    std::cout << "Position: X: " << ballPosition.x << ", Y: " << ballPosition.y << std::endl; // Velocity should be reflected

    // Test collision with boundary
    ballPosition = Vector2(219, 0);
    ballVelocity = Vector2(2, 0);
    updateBallPosition(ballPosition, ballVelocity, obstacles, radius);
    std::cout << "Velocity: X: " << ballVelocity.x << ", Y: " << ballVelocity.y << std::endl; // Velocity should be reflected
    std::cout << "Position: X: " << ballPosition.x << ", Y: " << ballPosition.y << std::endl; // Velocity should be reflected

    // Clean up
    for (Obstacle* obstacle : obstacles) {
        delete obstacle;
    }
}

// Run all tests
int main() {

    testVector2Operations();
    testObstacleCollision();
    testUpdateBallPosition();

    return 0;
}