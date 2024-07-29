import Foundation

enum ReminderFrequency {
    case daily
    case weekly
    case monthly
}

func nextOccurrence(from date: Date, frequency: ReminderFrequency, timeZone: TimeZone) -> Date {
    let calendar = Calendar.current
    calendar.timeZone = timeZone
    
    switch frequency {
    case .daily:
        return calendar.date(byAdding: .day, value: 1, to: date)!
    case .weekly:
        return calendar.date(byAdding: .weekOfYear, value: 1, to: date)!
    case .monthly:
        return calendar.date(byAdding: .month, value: 1, to: date)!
    }
}

// Usage example
let now = Date()
let timeZone = TimeZone.current
let nextDate = nextOccurrence(from: now, frequency: .weekly, timeZone: timeZone)

print("Next occurrence: \(nextDate)")