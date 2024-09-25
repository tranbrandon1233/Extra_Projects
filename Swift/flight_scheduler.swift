import Foundation

enum FlightStatus {
    case onTime
    case delayed
    case cancelled
}

struct Flight {
    let flightNumber: String
    var departureTime: Date
    var arrivalTime: Date
    var status: FlightStatus
    var assignedGate: String?
}
 
struct Passenger {
    let name: String
    let passportNumber: String
    var boardingPass: String?
    var notified: Bool = false
}

class AirportManagementSystem {
    private var flights: [String: Flight] = [:]
    private var passengers: [String: Passenger] = [:]
    private var gates: Set<String> = Set()
    private var occupiedGates: Set<String> = Set()
    
    func addGate(_ gate: String) {
        gates.insert(gate)
    }
    
    func removeGate(_ gate: String) {
        gates.remove(gate)
        occupiedGates.remove(gate)
    }
    
    func addFlight(_ flight: Flight) {
        flights[flight.flightNumber] = flight
    }
    
    func assignGate(to flightNumber: String, gate: String) {
        guard var flight = flights[flightNumber] else {
            print("Flight not found.")
            return
        }
        if !gates.contains(gate) {
            print("Gate \(gate) does not exist.")
            return
        }
        flight.assignedGate = gate
        occupiedGates.insert(gate)
        flights[flightNumber] = flight
    }
    
    func cancelFlight(_ flightNumber: String) {
        guard var flight = flights[flightNumber] else {
            print("Flight not found.")
            return
        }
        flight.status = .cancelled
        flights[flightNumber] = flight
        notifyPassengers(of: flightNumber, message: "Your flight \(flightNumber) has been cancelled.")
    }
    
    func rescheduleFlight(_ flightNumber: String, newDepartureTime: Date, newArrivalTime: Date) {
        guard var flight = flights[flightNumber] else {
            print("Flight not found.")
            return
        }
        flight.departureTime = newDepartureTime
        flight.arrivalTime = newArrivalTime
        flight.status = .delayed
        flights[flightNumber] = flight
        notifyPassengers(of: flightNumber, message: "Your flight \(flightNumber) has been rescheduled.")
    }
    
    func checkInPassenger(_ passenger: Passenger, for flightNumber: String) {
        guard let _ = flights[flightNumber] else {
            print("Flight not found.")
            return
        }
        var updatedPassenger = passenger
        updatedPassenger.boardingPass = "BP-\(flightNumber)-\(passenger.passportNumber)"
        passengers[passenger.passportNumber] = updatedPassenger
    }
    
    private func notifyPassengers(of flightNumber: String, message: String) {
        for (passportNumber, var passenger) in passengers {
            if passenger.boardingPass?.contains(flightNumber) == true {
                passenger.notified = true
                passengers[passportNumber] = passenger
                print("Notification to \(passenger.name): \(message)")
            }
        }
    }
    
    func generateFlightReport() -> [String: (status: FlightStatus, gate: String?)] {
        var report: [String: (status: FlightStatus, gate: String?)] = [:]
        for (flightNumber, flight) in flights {
            report[flightNumber] = (flight.status, flight.assignedGate)
        }
        return report
    }
    
    func generatePassengerReport() -> [String: (boardingPass: String?, notified: Bool)] {
        var report: [String: (boardingPass: String?, notified: Bool)] = [:]
        for (passportNumber, passenger) in passengers {
            report[passportNumber] = (passenger.boardingPass, passenger.notified)
        }
        return report
    }
    
    // Functions for test cases
    
    /**
     Executes the gate management test case.
          */
    func testGateManagement() {
        print("\n--- Gate Management Test ---")
        addGate("Gate 1")
        addGate("Gate 2")
        addGate("Gate 3")
        print("Gates added: \(gates)")
        print("Total gates: \(gates.count)")
    }
    
    /**
     Executes the flight scheduling test case.
      */
    func testFlightScheduling() {
        print("\n--- Flight Scheduling Test ---")
        let now = Date()
        let flight1 = Flight(flightNumber: "AA123", departureTime: now, arrivalTime: now.addingTimeInterval(3600), status: .onTime)
        let flight2 = Flight(flightNumber: "BB456", departureTime: now, arrivalTime: now.addingTimeInterval(7200), status: .delayed)
        let flight3 = Flight(flightNumber: "CC789", departureTime: now, arrivalTime: now.addingTimeInterval(1800), status: .cancelled)
        
        addFlight(flight1)
        addFlight(flight2)
        addFlight(flight3)
        
        // Print flight details
        for (flightNumber, flight) in flights {
            print("Flight \(flightNumber):")
            print("  Departure: \(flight.departureTime)")
            print("  Arrival: \(flight.arrivalTime)")
            print("  Status: \(flight.status)")
        }
    }
    
    /**
     Executes the gate assignment test case.
     */
    func testGateAssignment() {
        print("\n--- Gate Assignment Test ---")
        assignGate(to: "AA123", gate: "Gate 1")
        assignGate(to: "BB456", gate: "Gate 2")
        
        // Print gate assignments
        for (flightNumber, flight) in flights {
            if let gate = flight.assignedGate {
                print("Flight \(flightNumber) assigned to \(gate)")
            }
        }
    }
    
    /**
     Executes the passenger check-in test case.
     */
    func testPassengerCheckIn() {
        print("\n--- Passenger Check-in Test ---")
        let passenger1 = Passenger(name: "John Doe", passportNumber: "P123456")
        let passenger2 = Passenger(name: "Jane Smith", passportNumber: "P789101")
        
        checkInPassenger(passenger1, for: "AA123")
        checkInPassenger(passenger2, for: "BB456")
        
        // Print passenger check-in information
        for (passportNumber, passenger) in passengers {
            print("Passenger: \(passenger.name)")
            print("  Passport: \(passportNumber)")
            print("  Boarding Pass: \(passenger.boardingPass ?? "Not issued")")
        }
    }
    
    /**
     Executes the flight cancellation and rescheduling test case.
     */
    func testFlightCancellationAndRescheduling() {
        print("\n--- Flight Cancellation and Rescheduling Test ---")
        cancelFlight("CC789")
        
        let now = Date()
        rescheduleFlight("BB456", newDepartureTime: now.addingTimeInterval(3600), newArrivalTime: now.addingTimeInterval(7200))
        
        // Print updated flight information
        for (flightNumber, flight) in flights {
            print("Flight \(flightNumber):")
            print("  Status: \(flight.status)")
            print("  Departure: \(flight.departureTime)")
            print("  Arrival: \(flight.arrivalTime)")
        }
    }
}

// Create an instance of AirportManagementSystem
let airportSystem = AirportManagementSystem()

// Execute test cases
airportSystem.testGateManagement()
airportSystem.testFlightScheduling()
airportSystem.testGateAssignment()
airportSystem.testPassengerCheckIn()
airportSystem.testFlightCancellationAndRescheduling()

// Generate and print reports
print("\n--- Flight Report ---")
let flightReport = airportSystem.generateFlightReport()
for (flightNumber, info) in flightReport {
    print("Flight \(flightNumber): Status - \(info.status), Gate - \(info.gate ?? "Not assigned")")
}

print("\n--- Passenger Report ---")
let passengerReport = airportSystem.generatePassengerReport()
for (passportNumber, info) in passengerReport {
    print("Passenger \(passportNumber): Boarding Pass - \(info.boardingPass ?? "Not issued"), Notified - \(info.notified)")
}