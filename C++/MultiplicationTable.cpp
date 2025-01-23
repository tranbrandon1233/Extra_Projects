#include <iostream>
#include <vector>

// Function to generate a multiplication table from 1 to n
std::vector<std::vector<int>> get_multiplication_table(int n) {
    // Check if n is less than 1
    if (n < 1) {
        return std::vector<std::vector<int>>(); // Return an empty vector
    }

    // Create a 2D vector with n+1 rows
    std::vector<std::vector<int>> table(n + 1);

    // Fill the table with multiplication results
    for (int i = 1; i <= n; ++i) {
        // Reserve space for table row
        table[i].reserve(i + 1);
        for (int j = 1; j <= i; ++j) {
            table[i].push_back(i * j);
        }
    }

    return table;
}

// Main function to demonstrate the usage of get_multiplication_table
int main() {
    int n = -5; // Example input for n
    std::vector<std::vector<int>> result = get_multiplication_table(n);

    // Print the multiplication table
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= i; ++j) {
            std::cout << result[i][j - 1] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}