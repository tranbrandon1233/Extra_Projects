enum ValidationResult {
    case success
    case failure(String)
}

import Foundation

// Enum for Vehicle Types
enum VehicleType {
    case PassengerAircraft
    case FighterJet
    case HighSpeedTrain
    case Car
    case Bicycle
}

// Vehicle Class
class Vehicle {
    private var miles: Int = 0
    private var type: VehicleType
    private var customSpeed: Int?
    private var fuelEfficiency: Double // miles per gallon
    
    // Default Speeds for Vehicles (miles per hour)
    private let defaultSpeeds: [VehicleType: Int] = [
        .PassengerAircraft: 575,
        .FighterJet: 1320,
        .HighSpeedTrain: 217,
        .Car: 100,
        .Bicycle: 10
    ]
    
    // Default Fuel Efficiencies (miles per gallon)
    private let defaultEfficiencies: [VehicleType: Double] = [
        .PassengerAircraft: 0.2,   // 0.2 mpg
        .FighterJet: 0.1,         // 0.1 mpg
        .HighSpeedTrain: 10.0,    // 10 mpg
        .Car: 25.0,               // 25 mpg
        .Bicycle: Double.infinity // Infinite efficiency (human-powered)
    ]
    
    init(type: VehicleType, customSpeed: Int? = nil) {
        self.type = type
        self.customSpeed = customSpeed
        self.fuelEfficiency = defaultEfficiencies[type] ?? 0
    }
    
    func startEngine(minutes: Int) throws {
        guard minutes >= 0 else {
            throw VehicleError.invalidRuntime
        }
        
        let speed = customSpeed ?? (defaultSpeeds[type] ?? 0)
        guard speed > 0 else {
            throw VehicleError.invalidSpeed
        }
        
        self.miles += speed * (minutes / 60)
    }
    
    func returnMiles() -> Int {
        return self.miles
    }
    
    func calculateFuelConsumption() -> Double {
        guard fuelEfficiency > 0 else {
            return 0.0 // Fuel consumption is zero for human-powered vehicles
        }
        return Double(miles) / fuelEfficiency
    }
    
    func resetMileage() {
        self.miles = 0
    }
}

// Custom Error Types
enum VehicleError: Error, CustomStringConvertible {
    case invalidRuntime
    case invalidSpeed
    
    var description: String {
        switch self {
        case .invalidRuntime:
            return "Runtime must be zero or positive."
        case .invalidSpeed:
            return "Speed must be a positive value."
        }
    }
}

class VehicleValidator {
    static func validateAll() {
        print("Starting Vehicle Validation Suite...")
        print("------------------------------------")
        
        let results = [
            validateMileageCalculation(),
            validateFuelConsumption(),
            validateNegativeRuntime(),
            validateInvalidSpeed(),
            validateResetMileage(),
            validateCustomSpeedInitialization()
        ]
        
        print("\nValidation Summary:")
        print("------------------------------------")
        var failureCount = 0
        for (index, result) in results.enumerated() {
            switch result {
            case .success:
                print("Test \(index + 1): ✅ Passed")
            case .failure(let message):
                print("Test \(index + 1): ❌ Failed - \(message)")
                failureCount += 1
            }
        }
        print("\nTotal Results: \(results.count - failureCount) passed, \(failureCount) failed")
    }
    
    static func validateMileageCalculation() -> ValidationResult {
        let car = Vehicle(type: .Car)
        do {
            try car.startEngine(minutes: 120)
            let miles = car.returnMiles()
            guard miles == 200 else {
                return .failure("Car traveled \(miles) miles instead of expected 200 miles")
            }
            return .success
        } catch {
            return .failure("Unexpected error: \(error)")
        }
    }
    
    static func validateFuelConsumption() -> ValidationResult {
        let car = Vehicle(type: .Car)
        do {
            try car.startEngine(minutes: 120)
            let consumption = car.calculateFuelConsumption()
            guard consumption == 8.0 else {
                return .failure("Car consumed \(consumption) gallons instead of expected 8.0 gallons")
            }
            return .success
        } catch {
            return .failure("Unexpected error: \(error)")
        }
    }
    
    static func validateNegativeRuntime() -> ValidationResult {
        let car = Vehicle(type: .Car)
        do {
            try car.startEngine(minutes: -10)
            return .failure("Expected invalidRuntime error but no error was thrown")
        } catch let error as VehicleError {
            guard error == .invalidRuntime else {
                return .failure("Expected invalidRuntime error but got \(error)")
            }
            return .success
        } catch {
            return .failure("Unexpected error type: \(error)")
        }
    }
    
    static func validateInvalidSpeed() -> ValidationResult {
        let bicycle = Vehicle(type: .Bicycle, customSpeed: 0)
        do {
            try bicycle.startEngine(minutes: 60)
            return .failure("Expected invalidSpeed error but no error was thrown")
        } catch let error as VehicleError {
            guard error == .invalidSpeed else {
                return .failure("Expected invalidSpeed error but got \(error)")
            }
            return .success
        } catch {
            return .failure("Unexpected error type: \(error)")
        }
    }
    
    static func validateResetMileage() -> ValidationResult {
        let car = Vehicle(type: .Car)
        do {
            try car.startEngine(minutes: 120)
            car.resetMileage()
            let miles = car.returnMiles()
            guard miles == 0 else {
                return .failure("Mileage should be 0 after reset but was \(miles)")
            }
            return .success
        } catch {
            return .failure("Unexpected error: \(error)")
        }
    }
    
    static func validateCustomSpeedInitialization() -> ValidationResult {
        let customSpeedCar = Vehicle(type: .Car, customSpeed: 150)
        do {
            try customSpeedCar.startEngine(minutes: 60)
            let miles = customSpeedCar.returnMiles()
            guard miles == 150 else {
                return .failure("Car traveled \(miles) miles instead of expected 150 miles")
            }
            return .success
        } catch {
            return .failure("Unexpected error: \(error)")
        }
    }
}

// Usage
VehicleValidator.validateAll()