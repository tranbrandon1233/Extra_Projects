import Foundation

class PatientManagementSystem {
    private var patients: [String: [String: Any]] = [:]
    
    func addPatient() {
        print("Enter patient name:")
        guard let name = readLine(), !name.isEmpty else {
            print("Invalid name. Patient not added.")
            return
        }
        
        print("Enter patient age:")
        guard let ageString = readLine(), let age = Int(ageString) else {
            print("Invalid age. Patient not added.")
            return
        }
        
        print("Enter patient gender:")
        guard let gender = readLine(), !gender.isEmpty else {
            print("Invalid gender. Patient not added.")
            return
        }
        
        print("Enter patient medical history:")
        guard let medicalHistory = readLine() else {
            print("Invalid medical history. Patient not added.")
            return
        }
        
        let patientInfo: [String: Any] = [
            "age": age,
            "gender": gender,
            "medicalHistory": medicalHistory
        ]
        patients[name] = patientInfo
        print("Patient added successfully.")
    }
    
    func getPatientInfo() {
        print("Enter patient name to retrieve information:")
        guard let name = readLine(), !name.isEmpty else {
            print("Invalid name.")
            return
        }
        
        guard let patientInfo = patients[name] else {
            print("Patient not found.")
            return
        }
        
        print("Patient Information:")
        print("Name: \(name)")
        print("Age: \(patientInfo["age"] as? Int ?? 0)")
        print("Gender: \(patientInfo["gender"] as? String ?? "")")
        print("Medical History: \(patientInfo["medicalHistory"] as? String ?? "")")
    }
    
    func run() {
        while true {
            print("\nPatient Management System")
            print("1. Add Patient")
            print("2. Get Patient Information")
            print("3. Exit")
            print("Enter your choice:")
            
            guard let choice = readLine(), let option = Int(choice) else {
                print("Invalid input. Please try again.")
                continue
            }
            
            switch option {
            case 1:
                addPatient()
            case 2:
                getPatientInfo()
            case 3:
                print("Exiting the system.")
                return
            default:
                print("Invalid option. Please try again.")
            }
        }
    }
}

// Usage
let system = PatientManagementSystem()
system.run()