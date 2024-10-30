#include <iostream>
#include <vector>
#include <map>
#include <ctime>
#include <regex>
#include <mutex>
#include <iomanip>
#include <fstream>
#include <random>
#include <sstream>

using namespace std;

mutex mtx;
ofstream logFile("transactions.log", ios::app);

// UUID generator for unique car identification (for production, consider using a robust UUID library)
string generateUUID() {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 15);
    stringstream ss;
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            ss << '-';
        }
        else {
            ss << hex << dis(gen);
        }
    }
    return ss.str();
}

struct Car {
    string uuid;
    string licensePlate;
    string type;
    bool isRented;
    time_t rentTime;
};

map<string, double> hourlyRates = { {"SUV", 70.0 / 24}, {"Compact", 40.0 / 24}, {"Luxury", 100.0 / 24} };
vector<Car> fleet;

bool isValidLicensePlate(const string& licensePlate) {
    regex pattern("[A-Z]{3}-[0-9]{4}");
    return regex_match(licensePlate, pattern);
}

void addCar() {
    lock_guard<mutex> lock(mtx);
    string licensePlate, type;
    cout << "Enter license plate number (AAA-1234): ";
    cin >> licensePlate;
    if (!isValidLicensePlate(licensePlate)) {
        cout << "Invalid license plate format." << endl;
        return;
    }
    for (auto& car : fleet) {
        if (car.licensePlate == licensePlate) {
            cout << "Car with license plate number " << licensePlate << " already exists." << endl;
            return;
        }
	}
    cout << "Enter car type (SUV, Compact, Luxury): ";
    cin >> type;
    if (hourlyRates.find(type) == hourlyRates.end()) {
        cout << "Invalid car type." << endl;
        return;
    }
    string uuid = generateUUID();
    fleet.push_back({ uuid, licensePlate, type, false, 0 });
    cout << "Car added successfully with UUID: " << uuid << endl;
}

void logTransaction(const Car& car, const string& transactionType, double cost = 0.0) {
    time_t now = time(0);
    tm ltm;
    localtime_s(&ltm, &now);
    logFile << put_time(&ltm, "%Y-%m-%d %H:%M:%S") << " - UUID: " << car.uuid
        << ", License Plate: " << car.licensePlate << ", Transaction Type: " << transactionType;
    if (cost > 0) logFile << ", Cost: $" << fixed << setprecision(2) << cost;
    logFile << endl;
}

void rentCar() {
    lock_guard<mutex> lock(mtx);
    string licensePlate;
    cout << "Enter license plate number to rent: ";
    cin >> licensePlate;
    for (auto& car : fleet) {
        if (car.licensePlate == licensePlate && !car.isRented) {
            car.isRented = true;
            car.rentTime = time(0);
            cout << "Car rented successfully." << endl;
            logTransaction(car, "rent");
            return;
        }
    }
    cout << "Car not found or already rented." << endl;
}

void returnCar() {
    lock_guard<mutex> lock(mtx);
    string licensePlate;
    cout << "Enter license plate number to return: ";
    cin >> licensePlate;
    for (auto& car : fleet) {
        if (car.licensePlate == licensePlate && car.isRented) {
            car.isRented = false;
            time_t now = time(0);
            double duration = difftime(now, car.rentTime) / 3600.0; // Calculate duration in hours
            double cost = (duration < 24) ? hourlyRates[car.type] * ceil(duration)
                : hourlyRates[car.type] * 24 * ceil(duration / 24);
            cout << "Car returned successfully. Rental cost: $" << fixed << setprecision(2) << cost << endl;
            logTransaction(car, "return", cost);
            return;
        }
    }
    cout << "Car not found or not rented." << endl;
}

void displayFleet() {
    lock_guard<mutex> lock(mtx);
    cout << "Fleet Status:" << endl;
    for (const auto& car : fleet) {
        cout << "UUID: " << car.uuid << ", License Plate: " << car.licensePlate
            << ", Type: " << car.type << ", Status: " << (car.isRented ? "Rented" : "Available") << endl;
    }
}

int main() {
    int choice;
    do {
        cout << "\nCar Rental System Menu:\n1. Add Car\n2. Rent Car\n3. Return Car\n4. Display Fleet\n5. Exit\nEnter your choice: ";
        cin >> choice;
        switch (choice) {
        case 1: addCar(); break;
        case 2: rentCar(); break;
        case 3: returnCar(); break;
        case 4: displayFleet(); break;
        case 5: cout << "Exiting program." << endl; break;
        default: cout << "Invalid choice." << endl;
        }
    } while (choice != 5);

    return 0;
}