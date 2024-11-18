import Foundation

// FitnessTracker class to simulate tracking steps and progress
class FitnessTracker {
    
    var stepCount: Int
    var stepGoal: Int
    
    init(stepGoal: Int) {
        self.stepCount = 0
        self.stepGoal = stepGoal
    }
    
    // Method to simulate a step (increment the step count)
    func addSteps(_ steps: Int) {
        stepCount += steps

    }
    
    // Method to display current progress
    func displayProgress() {
        let percentage = Double(stepCount) / Double(stepGoal) * 100
        print("\nStep Count: \(stepCount) steps")
        print("Goal: \(stepGoal) steps")
        print("Progress: \(String(format: "%.2f", percentage))%")
        
        if stepCount >= stepGoal {
            print("Congratulations! You've reached your step goal!")
        } else {
            print("Keep going! You have \(stepGoal - stepCount) steps left to reach your goal.")
        }
    }
    
    // Method to display the main menu
    func displayMenu() {
        print("\nFitness Tracker")
        print("1. Add Steps")
        print("2. Display Progress")
        print("3. Quit")
    }
}

// Function to prompt user input
func promptUserInput(message: String) -> String? {
    print(message)
    return readLine()
}

// Main function to run the fitness tracker
func runFitnessTracker() {
    print("Welcome to the Fitness Tracker!")
    
    // Set up the step goal
    var stepGoal: Int = 0
    while true {
        if let input = promptUserInput(message: "Please set your daily step goal (e.g., 10000):"),
           let goal = Int(input), goal > 0 {
            stepGoal = goal
            break
        } else {
            print("Invalid input. Please enter a valid number for your step goal.")
        }
    }
    
    let tracker = FitnessTracker(stepGoal: stepGoal)
    
    while true {
        tracker.displayMenu()
        
        if let choice = promptUserInput(message: "Choose an option:") {
            switch choice {
            case "1":
                // Add steps
                if let input = promptUserInput(message: "How many steps did you take?"),
                   let steps = Int(input), steps > 0 {
                    tracker.addSteps(steps)
                } else {
                    print("Invalid input. Please enter a valid number of steps.")
                }
            case "2":
                // Display progress
                tracker.displayProgress()
            case "3":
                // Quit the app
                print("Thank you for using the Fitness Tracker! Goodbye!")
                return
            default:
                print("Invalid choice. Please choose 1, 2, or 3.")
            }
        }
    }
}

runFitnessTracker()