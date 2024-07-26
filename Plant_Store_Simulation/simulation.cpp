#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <functional>

class Customer {
public:
    double arrival;
    double spend;

    Customer(double arrival) {
        this->arrival = arrival;
        this->spend = generate_spend();
    }

    bool operator<(const Customer& other) const {
        return this->arrival > other.arrival;
    }

private:
    double generate_spend() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 100);
        return dis(gen);
    }
};

class PlantShop {
public:
    int customers;
    double revenue;
    std::priority_queue<Customer> queue;

    PlantShop() {
        customers = 0;
        revenue = 0;
    }

    void gen_customers(double time, double lambda) {
        double clock = 0;
        while (clock < time) {
            double next_arrival = clock + generate_exp(lambda);
            queue.push(Customer(next_arrival));
            clock = next_arrival;
        }
    }

    void simulation(double time, double lambda) {
        double clock = 0;
        gen_customers(time, lambda);

        while (clock < time && !queue.empty()) {
            Customer customer = queue.top();
            queue.pop();
            customers++;
            revenue += customer.spend;
            std::cout << "Customer spent $" << customer.spend << std::endl;
            clock = customer.arrival;
        }

        std::cout << "Total customers: " << customers << std::endl;
        std::cout << "Total revenue: $" << revenue << std::endl;
    }

private:
    double generate_exp(double lambda) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::exponential_distribution<> dis(lambda);
        return dis(gen);
    }
};

int main() {
    PlantShop shop; // Create a PlantShop object

    // Simulate 8 hours (28800 seconds) with an average of 1 customer every 5 minutes (300 seconds)
    double simulation_time = 28800.0; 
    double arrival_rate = 1.0 / 300.0; 

    shop.simulation(simulation_time, arrival_rate); // Run the simulation

    return 0;
}