import Foundation

// Define a custom type for days of the week
enum DayOfWeek: String, CaseIterable {
    case Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
}

// Sample data representing steps taken each day of the week
var stepsData: [DayOfWeek: [Int]] = [
    .Monday: [3000, 5000, 2000],
    .Tuesday: [4000, 3000, 4000],
    .Wednesday: [7000, 2000],
    .Thursday: [3000, 3000, 3000],
    .Friday: [4000, 4000, 2000],
    .Saturday: [1000, 1000, 5000, 2000],
    .Sunday: [5000, 2000]
]

// Function to calculate the total number of steps taken each day
func calculateDailySteps(data: [DayOfWeek: [Int]]) -> [DayOfWeek: Int] {
    var dailySteps: [DayOfWeek: Int] = [:]
    for (day, steps) in data {
        dailySteps[day] = steps.reduce(0, +)
    }
    return dailySteps
}

// Function to print the daily steps report
func printDailyReport(dailySteps: [DayOfWeek: Int]) {
    for (day, totalSteps) in dailySteps {
        print("\(day.rawValue): \(totalSteps) steps")
    }
}

// Function to calculate the total number of steps taken in a week
func calculateTotalWeeklySteps(dailySteps: [DayOfWeek: Int]) -> Int {
    return dailySteps.values.reduce(0, +)
}

// Function to print the weekly steps report
func printWeeklyReport(totalWeeklySteps: Int) {
    print("Total steps for the week: \(totalWeeklySteps)")
}

// Main function to generate the weekly report
func generateWeeklyReport(data: [DayOfWeek: [Int]]) {
    let dailySteps = calculateDailySteps(data: data)
    printDailyReport(dailySteps: dailySteps)
    let totalWeeklySteps = calculateTotalWeeklySteps(dailySteps: dailySteps)
    printWeeklyReport(totalWeeklySteps: totalWeeklySteps)
}

// Generating the weekly report with sample data
generateWeeklyReport(data: stepsData)
