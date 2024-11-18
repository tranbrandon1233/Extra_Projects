import Foundation
struct Exercise {
    var category: String
    var name: String
    var caloriesPerMinute: Double
}

func getCategories(from exercises: [Exercise]) -> [String] {
    var categories: [String] = []
    var categorySet: Set<String> = []
    for exercise in exercises {
        if !categorySet.contains(exercise.category) {
            categorySet.insert(exercise.category)
            categories.append(exercise.category)
        }
    }
    return categories
}

func filterExercises(by category: String, from exercises: [Exercise]) -> [Exercise] {
    return exercises.filter { $0.category == category }
}

func calculateMinutesNeeded(targetCalories: Double, caloriesPerMinute: Double) -> Int {
    return Int(ceil(targetCalories / caloriesPerMinute))
}

func runValidations() {
    print("Starting Exercise Program Validations...")
    
    // Test Data
    let exercises = [
        Exercise(category: "Cardio", name: "Walking", caloriesPerMinute: 4.2),
        Exercise(category: "Cardio", name: "Running", caloriesPerMinute: 10.5),
        Exercise(category: "Legs", name: "Squats", caloriesPerMinute: 6.0)
    ]
    
    // Test 1: Get Categories
    do {
        let result = getCategories(from: exercises)
        assert(result == ["Cardio", "Legs"],
               """
               getCategories failed:
               Expected: ["Cardio", "Legs"]
               Got: \(result)
               """)
        print("✅ Categories test passed")
    }
    
    // Test 2: Filter Exercises
    do {
        let result = filterExercises(by: "Cardio", from: exercises)
        assert(result.count == 2,
               "Expected 2 cardio exercises, got \(result.count)")
        
        let exerciseNames = result.map { $0.name }
        assert(exerciseNames == ["Walking", "Running"],
               """
               Filtered exercises don't match:
               Expected: ["Walking", "Running"]
               Got: \(exerciseNames)
               """)
        print("✅ Filter exercises test passed")
    }
    
    // Test 3: Calculate Minutes Needed
    do {
        let result = calculateMinutesNeeded(targetCalories: 120, caloriesPerMinute: 6)
        assert(result == 20,
               """
               Minutes calculation incorrect:
               Expected: 20
               Got: \(result)
               """)
        print("✅ Calculate minutes test passed")
    }
    
    // Test 4: Integration Flow
    do {
        // Test Categories
        let categories = getCategories(from: exercises)
        assert(categories == ["Cardio", "Legs"],
               "Integration test: Categories don't match expected result")
        
        // Test Filtering
        let filteredExercises = filterExercises(by: "Cardio", from: exercises)
        assert(filteredExercises.count == 2,
               "Integration test: Wrong number of filtered exercises")
        
        let filteredNames = filteredExercises.map { $0.name }
        assert(filteredNames == ["Walking", "Running"],
               "Integration test: Filtered exercise names don't match")
        
        // Test Calculation
        let minutesNeeded = calculateMinutesNeeded(targetCalories: 100, 
                                                 caloriesPerMinute: filteredExercises[0].caloriesPerMinute)
        assert(minutesNeeded == 24,
               """
               Integration test: Minutes calculation incorrect
               Expected: 24
               Got: \(minutesNeeded)
               """)
        print("✅ Integration flow test passed")
    }
    
    print("\n🎉 All validations completed successfully!")
}

func demonstrateUsage() {
    print("\nDemonstrating Exercise Program Usage:")
    
    let exercises = [
        Exercise(category: "Cardio", name: "Walking", caloriesPerMinute: 4.2),
        Exercise(category: "Cardio", name: "Running", caloriesPerMinute: 10.5),
        Exercise(category: "Legs", name: "Squats", caloriesPerMinute: 6.0)
    ]
    
    print("\nAvailable Categories:")
    let categories = getCategories(from: exercises)
    categories.forEach { print("- \($0)") }
    
    print("\nCardio Exercises:")
    let cardioExercises = filterExercises(by: "Cardio", from: exercises)
    cardioExercises.forEach { exercise in
        print("- \(exercise.name) (\(exercise.caloriesPerMinute) cal/min)")
    }
    
    let targetCalories = 100.0
    print("\nTo burn \(targetCalories) calories:")
    cardioExercises.forEach { exercise in
        let minutes = calculateMinutesNeeded(targetCalories: targetCalories, 
                                           caloriesPerMinute: exercise.caloriesPerMinute)
        print("- \(exercise.name): \(minutes) minutes")
    }
}

// Run program
print("Exercise Program Validation and Demo\n")
runValidations()

print("\nWould you like to see a demonstration? (y/n)")
if let response = readLine(), response.lowercased() == "y" {
    demonstrateUsage()
}