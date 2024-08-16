import Foundation

// Function to input classes and subjects
func inputClassesAndSubjects() -> (classes: [String], subjects: [String]) {
    let classes = ["Tenth Grade", "Ninth Grade", "Eighth Grade", "Seventh Grade"]
    let subjects = ["Physics", "Arabic", "English", "Chemistry", "Math", "Biology", "History", "Geography"]
    return (classes, subjects)
}

// Function to create schedule
func createSchedule(classes: [String], subjects: [String]) -> [String: [String: String]] {
    var schedule = [String: [String: String]]()
    var subjectTeachers = [String: String]()

    // Assign teachers to subjects
    for subject in subjects {
        print("Enter teacher for \(subject): ", terminator: "")
        let teacher = readLine() ?? ""
        subjectTeachers[subject] = teacher
    }

    // Create schedule ensuring no conflict
    for (index, className) in classes.enumerated() {
        schedule[className] = [String: String]()
        for (subjectIndex, _) in subjects.enumerated() {
            let slot = (index + subjectIndex) % subjects.count
            schedule[className]?[subjects[slot]] = subjectTeachers[subjects[slot]]
        }
    }
    return schedule
}

// Function to print the schedule
func printSchedule(schedule: [String: [String: String]]) {
    for (className, subjects) in schedule {
        print("\nSchedule for \(className):")
        for (subject, teacher) in subjects {
            print("\(subject): \(teacher)")
        }
    }
}

let (classes, subjects) = inputClassesAndSubjects()
let schedule = createSchedule(classes: classes, subjects: subjects)
printSchedule(schedule: schedule)