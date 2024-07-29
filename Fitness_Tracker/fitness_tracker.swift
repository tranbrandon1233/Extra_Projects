import Foundation

// Data Structures

/// Represents an exercise activity
struct ExerciseActivity {
    var name: String
    var duration: Int // in minutes
    var caloriesBurned: Int // in kcal
}

// Fitness Tracker Class

/// Manages exercise activities, allowing logging, updating, removing, and displaying activities
class FitnessTracker {
    private var activities: [ExerciseActivity] = []
    
    /**
     Logs a new exercise activity.
     
     - Parameters:
        - name: The name of the exercise.
        - duration: The duration of the exercise in minutes.
        - caloriesBurned: The calories burned during the exercise.
     */
    func logActivity(name: String, duration: Int, caloriesBurned: Int) {
        let activity = ExerciseActivity(name: name, duration: duration, caloriesBurned: caloriesBurned)
        activities.append(activity)
        print("🌟 Activity '\(name)' logged successfully. Great job!")
    }
    
    /**
     Removes an exercise activity by index.
     
     - Parameter index: The index of the activity to remove.
     */
    func removeActivity(index: Int) {
        if index >= 0 && index < activities.count {
            let removedActivity = activities.remove(at: index)
            print("🗑️ Activity '\(removedActivity.name)' removed successfully. Stay focused!")
        } else {
            print("⚠️ Invalid index. Please check the index and try again.")
        }
    }
    
    /**
     Updates an existing exercise activity by index.
     
     - Parameters:
        - index: The index of the activity to update.
        - duration: The new duration of the exercise in minutes.
        - caloriesBurned: The new calories burned during the exercise.
     */
    func updateActivity(index: Int, duration: Int, caloriesBurned: Int) {
        if index >= 0 && index < activities.count {
            activities[index].duration = duration
            activities[index].caloriesBurned = caloriesBurned
            print("🔄 Activity '\(activities[index].name)' updated successfully. Keep pushing!")
        } else {
            print("⚠️ Invalid index. Please check the index and try again.")
        }
    }
    
    /// Displays all logged exercise activities in a user-friendly format.
    func displayActivities() {
        if activities.isEmpty {
            print("No activities logged yet. Time to get moving! 🏃‍♂️🏃‍♀️")
        } else {
            print("\n🏋️‍♂️ Logged Exercise Activities:")
            for (index, activity) in activities.enumerated() {
                print("\(index + 1). 🏅 Name: \(activity.name), Duration: \(activity.duration) minutes, Calories Burned: \(activity.caloriesBurned) kcal")
            }
        }
    }
}

// Main Script Execution

let fitnessTracker = FitnessTracker()

while true {
    print("\n🏋️‍♂️ Fitness Tracker Menu:")
    print("1. Log Exercise Activity 📝")
    print("2. Remove Exercise Activity 🗑️")
    print("3. Update Exercise Activity 🔄")
    print("4. Display All Activities 📋")
    print("5. Exit 🚪")
    print("Enter your choice: ", terminator: "")
    
    if let choice = readLine(), let option = Int(choice) {
        switch option {
        case 1:
            print("Enter exercise name: ", terminator: "")
            let name = readLine() ?? ""
            print("Enter duration (in minutes): ", terminator: "")
            if let durationStr = readLine(), let duration = Int(durationStr) {
                print("Enter calories burned: ", terminator: "")
                if let caloriesStr = readLine(), let caloriesBurned = Int(caloriesStr) {
                    fitnessTracker.logActivity(name: name, duration: duration, caloriesBurned: caloriesBurned)
                } else {
                    print("⚠️ Invalid calories burned. Please enter a number.")
                }
            } else {
                print("⚠️ Invalid duration. Please enter a number.")
            }
        case 2:
            fitnessTracker.displayActivities()
            print("Enter index of exercise to remove: ", terminator: "")
            if let indexStr = readLine(), let index = Int(indexStr) {
                fitnessTracker.removeActivity(index: index - 1)
            } else {
                print("⚠️ Invalid index. Please enter a number.")
            }
        case 3:
            fitnessTracker.displayActivities()
            print("Enter index of exercise to update: ", terminator: "")
            if let indexStr = readLine(), let index = Int(indexStr) {
                print("Enter new duration (in minutes): ", terminator: "")
                if let durationStr = readLine(), let duration = Int(durationStr) {
                    print("Enter new calories burned: ", terminator: "")
                    if let caloriesStr = readLine(), let caloriesBurned = Int(caloriesStr) {
                        fitnessTracker.updateActivity(index: index - 1, duration: duration, caloriesBurned: caloriesBurned)
                    } else {
                        print("⚠️ Invalid calories burned. Please enter a number.")
                    }
                } else {
                    print("⚠️ Invalid duration. Please enter a number.")
                }
            } else {
                print("⚠️ Invalid index. Please enter a number.")
            }
        case 4:
            fitnessTracker.displayActivities()
        case 5:
            print("Exiting Fitness Tracker. Stay healthy and active! 🚶‍♂️🚶‍♀️")
            exit(0)
        default:
            print("⚠️ Invalid choice. Please try again.")
        }
    }
}
