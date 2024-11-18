import Foundation

class ActivityTracker {
    private var activityLog: [Date: [String: Double]] = [:]

    func recordActivity(date: Date, activity: String, duration: Double) {
        guard duration > 0 else {
            print("Invalid duration")
            return
        }
        if activityLog[date] == nil {
            activityLog[date] = [:]
        }
        activityLog[date]?[activity, default: 0] += duration
    }

    func totalActivityTime(activity: String) -> Double {
        var total = 0.0
        for (_, dayLog) in activityLog {
            total += dayLog[activity, default: 0]
        }
        return total
    }

    func averageDailyTime(activity: String) -> Double {
        let daysLogged = activityLog.keys.count
        guard daysLogged > 0 else { return 0.0 }
        return totalActivityTime(activity: activity) / Double(daysLogged)
    }

    func dayWithMaxActivity(activity: String) -> Date? {
        var maxDay: Date? = nil
        var maxTime = 0.0
        for (date, dayLog) in activityLog {
            if let time = dayLog[activity], time > maxTime {
                maxTime = time
                maxDay = date
            }
        }
        return maxDay
    }
    
    func clearAllLogs() {
        activityLog.removeAll()
    }
}