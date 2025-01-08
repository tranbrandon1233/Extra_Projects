#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <limits>

// Forward declaration of ElevatorSystem
class ElevatorSystem;

class Elevator {
private:
    int id;
    std::atomic<int> currentFloor;
    std::atomic<bool> isMoving;
    std::atomic<bool> isUnderMaintenance;
    std::priority_queue<int, std::vector<int>, std::greater<int>> upQueue; // Upward requests
    std::priority_queue<int> downQueue; // Downward requests
    mutable std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::atomic<bool> isRunning;
    std::atomic<int> currentDirection; // 1 = Up, -1 = Down, 0 = Idle

public:
    Elevator(int elevatorId)
        : id(elevatorId),
          currentFloor(0),
          isMoving(false),
          isUnderMaintenance(false),
          isRunning(true),
          currentDirection(0) {}

    void operate() {
        std::thread([this]() {
            while (isRunning) {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCondition.wait(lock, [this]() {
                    return !(upQueue.empty() && downQueue.empty()) || !isRunning;
                });

                if (!isRunning) break;

                if (!isUnderMaintenance) {
                    isMoving = true;
                    int nextFloor = currentFloor;

                    // Dynamically prioritize based on direction or proximity when idle
                    if (currentDirection == 1 && !upQueue.empty()) {
                        nextFloor = upQueue.top();
                        upQueue.pop();
                    } else if (currentDirection == -1 && !downQueue.empty()) {
                        nextFloor = downQueue.top();
                        downQueue.pop();
                    } else if (!upQueue.empty() || !downQueue.empty()) {
                        if (!upQueue.empty()) nextFloor = upQueue.top();
                        if (!downQueue.empty() && (upQueue.empty() || abs(downQueue.top() - currentFloor) < abs(nextFloor - currentFloor))) {
                            nextFloor = downQueue.top();
                            currentDirection = -1;
                        } else {
                            currentDirection = 1;
                        }
                    } else {
                        currentDirection = 0; // Idle state
                    }

                    if (nextFloor == currentFloor) {
                        isMoving = false;
                        continue;
                    }

                    lock.unlock();

                    std::cout << "Elevator " << id << " moving from floor "
                              << currentFloor << " to floor " << nextFloor << std::endl;

                    std::this_thread::sleep_for(std::chrono::seconds(std::abs(nextFloor - currentFloor)));
                    currentFloor = nextFloor;

                    std::cout << "Elevator " << id << " reached floor " << currentFloor << std::endl;

                    isMoving = false;
                }
            }
        }).detach();
    }

    void addDestination(int floor) {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (floor > currentFloor) {
            upQueue.push(floor);
        } else if (floor < currentFloor) {
            downQueue.push(floor);
        }
        queueCondition.notify_one(); // Wake up the elevator if it's idle
    }

    void setUnderMaintenance(bool status, ElevatorSystem& system);

    void stop() {
        isRunning = false;
        queueCondition.notify_one();
    }

    int getId() const { return id; }
    bool isAvailable() const { return !isMoving && !isUnderMaintenance; }
    int getCurrentFloor() const { return currentFloor; }
    int getCurrentDirection() const { return currentDirection; }

    int upQueueSize() const {
        std::lock_guard<std::mutex> lock(queueMutex);
        return upQueue.size();
    }

    int downQueueSize() const {
        std::lock_guard<std::mutex> lock(queueMutex);
        return downQueue.size();
    }
};

class ElevatorSystem {
private:
    std::vector<std::unique_ptr<Elevator>> elevators;

public:
    ElevatorSystem(int numberOfElevators) {
        for (int i = 0; i < numberOfElevators; ++i) {
            auto elevator = std::make_unique<Elevator>(i);
            elevator->operate();
            elevators.push_back(std::move(elevator));
        }
    }

    void redistributeRequests(const std::vector<int>& requests) {
        for (int floor : requests) {
            requestElevator(floor);
        }
    }

    void requestElevator(int requestedFloor) {
        auto bestElevator = findBestElevator(requestedFloor);
        if (bestElevator) {
            std::cout << "Allocating Elevator " << bestElevator->getId()
                      << " to floor " << requestedFloor << std::endl;
            bestElevator->addDestination(requestedFloor);
        } else {
            std::cout << "No elevators available. Please wait." << std::endl;
        }
    }

    Elevator* findBestElevator(int requestedFloor) {
        Elevator* bestElevator = nullptr;
        int minScore = std::numeric_limits<int>::max();

        for (auto& elevator : elevators) {
            if (elevator->isAvailable()) {
                int distance = std::abs(elevator->getCurrentFloor() - requestedFloor);
                bool isDirectionCompatible = (requestedFloor > elevator->getCurrentFloor() && elevator->getCurrentDirection() >= 0) ||
                                              (requestedFloor < elevator->getCurrentFloor() && elevator->getCurrentDirection() <= 0) ||
                                              elevator->getCurrentDirection() == 0;

                int load = elevator->upQueueSize() + elevator->downQueueSize();
                int score = (isDirectionCompatible ? 0 : 1000) + distance + load * 10;

                if (score < minScore) {
                    minScore = score;
                    bestElevator = elevator.get();
                }
            }
        }
        return bestElevator;
    }

    void shutdown() {
        for (auto& elevator : elevators) {
            elevator->stop();
        }
    }
};

void Elevator::setUnderMaintenance(bool status, ElevatorSystem& system) {
    isUnderMaintenance = status;
    if (status) {
        std::vector<int> activeRequests;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            while (!upQueue.empty()) {
                activeRequests.push_back(upQueue.top());
                upQueue.pop();
            }
            while (!downQueue.empty()) {
                activeRequests.push_back(downQueue.top());
                downQueue.pop();
            }
        }
        system.redistributeRequests(activeRequests);
    }
}

int main() {
    ElevatorSystem elevatorSystem(3);

    elevatorSystem.requestElevator(5);
    elevatorSystem.requestElevator(2);
    elevatorSystem.requestElevator(8);

    std::this_thread::sleep_for(std::chrono::seconds(10));

    elevatorSystem.shutdown();

    return 0;
}