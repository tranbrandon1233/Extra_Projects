#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <string>
#include <algorithm>
#include <chrono>
#include <limits>

// Enum class for order status with clear validation
enum class OrderStatus { Pending = 0, InProgress, Served };

// Structure to represent an order item
struct OrderItem {
    std::string name;
    int quantity;
};

// Structure to represent an order
struct Order {
    int id;
    int tableNumber;
    std::vector<OrderItem> items;
    OrderStatus status = OrderStatus::Pending;
};

// Shared resources
std::vector<Order> orders;
std::mutex orderMutex;
std::condition_variable orderCV;
bool running = true;
int nextOrderId = 1;

// Template function for validated input
template <typename T>
T getInput(const std::string& prompt) {
    T value;
    while (true) {
        std::cout << prompt;
        if (std::cin >> value && value > 0) {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return value;
        }
        std::cout << "Invalid input. Please try again.\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

// Function to validate string input for order item names
std::string getValidatedString(const std::string& prompt) {
    std::string input;
    while (true) {
        std::cout << prompt;
        std::getline(std::cin, input);
        if (!input.empty()) return input;
        std::cout << "Input cannot be empty. Please try again.\n";
    }
}

// Function to check if an order ID exists
bool orderExists(int id) {
    return std::any_of(orders.begin(), orders.end(),
        [id](const Order& order) { return order.id == id; });
}

// Function to place a new order with validation
void placeOrder() {
    std::lock_guard<std::mutex> lock(orderMutex);
    Order order;
    order.id = nextOrderId++;
    order.tableNumber = getInput<int>("Enter table number: ");
    int itemCount = getInput<int>("Enter the number of items: ");

    for (int i = 0; i < itemCount; ++i) {
        OrderItem item;
        item.name = getValidatedString("Enter item " + std::to_string(i + 1) + " name: ");
        item.quantity = getInput<int>("Enter item quantity: ");
        order.items.push_back(item);
    }

    orders.push_back(order);
    std::cout << "Order placed successfully! Order ID: " << order.id << "\n";
    orderCV.notify_one();
}

// Function to update order status with enhanced validation
void updateOrderStatus() {
    std::lock_guard<std::mutex> lock(orderMutex);
    int orderId = getInput<int>("Enter the order ID to update: ");

    auto it = std::find_if(orders.begin(), orders.end(),
        [orderId](const Order& order) { return order.id == orderId; });

    if (it != orders.end()) {
        int status;
        std::cout << "Enter new status (0 = Pending, 1 = InProgress, 2 = Served): ";
        std::cin >> status;

        if (status >= 0 && status <= 2) {
            if (it->status == OrderStatus::Pending && status == static_cast<int>(OrderStatus::Served)) {
                std::cout << "Invalid transition: Cannot skip 'InProgress' status.\n";
            }
            else {
                it->status = static_cast<OrderStatus>(status);
                std::cout << "Order status updated successfully!\n";
            }
        }
        else {
            std::cout << "Invalid status.\n";
        }
    }
    else {
        std::cout << "Order not found.\n";
    }
}

// Function to view all orders with optional filtering by status
void viewOrders() {
    std::lock_guard<std::mutex> lock(orderMutex);
    std::cout << "Filter by status? (0 = No, 1 = Yes): ";
    int filter;
    std::cin >> filter;

    OrderStatus statusFilter = OrderStatus::Pending;
    if (filter == 1) {
        int status;
        std::cout << "Enter status to filter (0 = Pending, 1 = InProgress, 2 = Served): ";
        std::cin >> status;
        if (status >= 0 && status <= 2) {
            statusFilter = static_cast<OrderStatus>(status);
        }
        else {
            std::cout << "Invalid status. Showing all orders.\n";
        }
    }

    for (const auto& order : orders) {
        if (!filter || order.status == statusFilter) {
            std::string orderStatus = "";
            switch (static_cast<int>(order.status)) {
                case 0:
                    orderStatus = "Pending";
                case 1:
                    orderStatus = "InProgress";
                case 2:
                    orderStatus = "Served";
            }
            std::cout << "Order ID: " << order.id
                << ", Table: " << order.tableNumber
                << ", Status: " << orderStatus << "\n";
        }
    }
}

// Kitchen thread function with detailed comments
void kitchenThread() {
    while (running) {
        std::unique_lock<std::mutex> lock(orderMutex);
        orderCV.wait(lock, [] { return !orders.empty() || !running; });

        if (!running) break;

        auto it = std::find_if(orders.begin(), orders.end(),
            [](const Order& o) { return o.status == OrderStatus::Pending; });

        if (it != orders.end()) {
            it->status = OrderStatus::InProgress;
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::seconds(3));
            lock.lock();
            it->status = OrderStatus::Served;
            std::cout << "Order " << it->id << " is now served.\n";
        }
    }
}

// Menu function for user interaction
void menu() {
    while (running) {
        std::cout << "\nMenu:\n"
            << "1. Place Order\n"
            << "2. Update Order Status\n"
            << "3. View Orders\n"
            << "4. Exit\n"
            << "Enter your choice: ";
        int choice = getInput<int>("");

        switch (choice) {
        case 1:
            placeOrder();
            break;
        case 2:
            updateOrderStatus();
            break;
        case 3:
            viewOrders();
            break;
        case 4:
            running = false;
            orderCV.notify_all();
            break;
        default:
            std::cout << "Invalid choice. Please try again.\n";
        }
    }
}

// Main function to start the program
int main() {
    std::thread kitchen(kitchenThread);
    menu();
    kitchen.join();
    return 0;
}