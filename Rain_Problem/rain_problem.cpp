#include <iostream>
#include <vector>
#include <algorithm>

int trapRainWater(const std::vector<int>& height) {
    if (height.empty()) {
        return 0;
    }

    int left = 0;
    int right = height.size() - 1;
    int leftMax = 0;
    int rightMax = 0;
    int trappedWater = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                trappedWater += (leftMax - height[left]);
            }
            ++left;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                trappedWater += (rightMax - height[right]);
            }
            --right;
        }
    }

    return trappedWater;
}

int main() {
    std::vector<int> height = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
    std::cout << "Trapped Rainwater: " << trapRainWater(height) << std::endl;
    return 0;
}