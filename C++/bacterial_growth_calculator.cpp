#include <iostream>
#include <vector>
#include <cmath>

class BacterialColony {
public:
    double growthRate;  // Growth rate in colonies per hour
    double temperature; // Temperature in degrees Celsius
    std::vector<double> nutrientLevels; // Nutrient levels as a percentage

    BacterialColony(double rate, double temp, std::vector<double> nutrients)
        : growthRate(rate), temperature(temp), nutrientLevels(nutrients) {}

    double calculateGrowth() {
        double totalNutrients = 0.0;

        // Safely sum the nutrient levels by iterating over the vector
        for (double nutrient : nutrientLevels) {
            totalNutrients += nutrient;
        }

        return growthRate * exp((temperature - 37) / 10) * totalNutrients; // Assumes optimal growth at 37°C
    }
};

int main() {
    std::vector<double> nutrients = { 50, 75 };
    BacterialColony colony(100, 35, nutrients); // Growth rate of 100 colonies/hour at 35°C

    std::cout << "Estimated growth: " << colony.calculateGrowth() << " colonies" << std::endl;

    return 0;
}