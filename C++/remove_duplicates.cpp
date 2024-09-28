#include <vector>
#include <iostream>
using namespace std;

/**
 * @brief Removes duplicates from a sorted vector in-place.
 * 
 * @param nums A reference to a vector of integers.
 * @return The new length of the vector after removing duplicates.
 */
int removeDuplicates(vector<int>& nums) {
    int index = 0; // Initialize index to track the position of unique elements
    for (int i = 0; i < nums.size(); i++) {
        // If the current element is the first element or different from the previous unique element
        if (index == 0 || nums[i] != nums[index - 1]) {
            nums[index] = nums[i]; // Update the element at the current index
            index++; // Move to the next position for unique elements
        }
    }
    nums.resize(index); // Resize the vector to contain only unique elements
    return index; // Return the new length of the vector
}

int main()
{
    vector<int> nums = {1, 1, 2, 3, 3, 4, 4, 5}; // Initialize the vector with duplicate elements
    int size = nums.size(); // Get the original size of the vector

    cout << "Original array: \n";
    for (int i = 0; i < size; i++) {
        cout << nums[i] << " "; // Print the original array
    }
    cout << "\n";

    size = removeDuplicates(nums); // Remove duplicates and get the new size

    cout << "Modified array: \n";
    for (int i = 0; i < size; i++) {
        cout << nums[i] << " "; // Print the modified array with unique elements
    }
    return 0;
}
