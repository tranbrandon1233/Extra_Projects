import Foundation

// Function to get a valid date input from the user
func getDateInput() -> Date? {
    print("Enter a date in the format YYYY-MM-DD:")
    if let input = readLine(), let date = DateFormatter.yyyyMMdd.date(from: input) {
        return date
    } else {
        print("Invalid date format. Please try again.")
        return getDateInput()
    }
}

// Extension to create a custom date formatter
extension DateFormatter {
    static let yyyyMMdd: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone.current
        formatter.locale = Locale.current
        return formatter
    }()
}

// Get the current date
let currentDate = Date()

// Get the user input date
guard let inputDate = getDateInput() else {
    print("Failed to get a valid date input.")
    exit(1)
}

// Calculate the difference between dates
let calendar = Calendar.current
let components = calendar.dateComponents([.day, .hour, .minute, .second], from: inputDate, to: currentDate)

// Extract the differences
let dayDifference = abs(components.day ?? 0)
let hourDifference = abs(components.hour ?? 0)
let minuteDifference = abs(components.minute ?? 0)
let secondDifference = abs(components.second ?? 0)

// Calculate total differences
let totalSeconds = calendar.dateComponents([.second], from: inputDate, to: currentDate).second ?? 0
let totalMinutes = totalSeconds / 60
let totalHours = totalMinutes / 60
let totalDays = totalHours / 24

// Determine if the input date is before or after the current date
let comparisonResult = inputDate.compare(currentDate)
let timeDescription = comparisonResult == .orderedAscending ? "ago" : "from now"

// Print the results
print("\nDifference between the date and today:")
print("\(dayDifference) days, \(hourDifference) hours, \(minuteDifference) minutes, and \(secondDifference) seconds \(timeDescription)")
print("\nTotal difference:")
print("Days: \(abs(totalDays))")
print("Hours: \(abs(totalHours))")
print("Minutes: \(abs(totalMinutes))")
print("Seconds: \(abs(totalSeconds))")