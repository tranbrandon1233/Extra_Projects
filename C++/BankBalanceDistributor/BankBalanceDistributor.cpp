#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // Open the input file
    std::ifstream inputFile("balance.in");
    if (!inputFile) {
        std::cerr << "Error opening input file." << std::endl;
        return 1;
    }

    // Read the number of customers and the total amount of money
    int n, m;
    inputFile >> n >> m;

    // Read the balance of each customer
    std::vector<int> balances(n);
    for (int i = 0; i < n; ++i) {
        inputFile >> balances[i];
    }

    // Close the input file
    inputFile.close();

    // Distribute the money
    int remainingMoney = m;
    int customersWithNoMoney = 0;

    for (int i = 0; i < n && remainingMoney > 0; ++i) {
        if (remainingMoney >= balances[i]) {
            remainingMoney -= balances[i];
            balances[i] = 0;
            customersWithNoMoney++;
        }
        else {
            balances[i] -= remainingMoney;
            remainingMoney = 0;
        }
    }

    // Open the output file
    std::ofstream outputFile("balance.out");
    if (!outputFile) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }

    // Write the number of customers with no money left
    outputFile << customersWithNoMoney << std::endl;

    // Write the final balance of each customer
    for (const int& balance : balances) {
        outputFile << balance << std::endl;
    }

    // Close the output file
    outputFile.close();

    return 0;
}