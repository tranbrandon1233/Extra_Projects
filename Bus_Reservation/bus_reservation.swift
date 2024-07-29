import Foundation

// Enum to represent different seat types
enum SeatType {
    case regular, window, aisle
}

// Struct to represent a seat
struct Seat {
    let id: Int
    let type: SeatType
    var isAvailable: Bool
    var isSelected: Bool
    var isReserved: Bool
}

// Class to manage seat selection
class SeatSelectionManager {
    private var seats: [Seat]
    private let lock = NSLock()
    private var reservationTimer: Timer?
    private let reservationTimeout: TimeInterval = 300 // 5 minutes
    
    init(seatCount: Int) {
        // Initialize seats (you might want to customize this based on your bus layout)
        self.seats = (0..<seatCount).map { id in
            let type: SeatType = id % 4 == 0 || id % 4 == 3 ? .window : (id % 2 == 0 ? .aisle : .regular)
            return Seat(id: id, type: type, isAvailable: true, isSelected: false, isReserved: false)
        }
    }
    
    // Function to select seats
    func selectSeats(ids: [Int], forUser userId: String) -> (success: Bool, message: String) {
        lock.lock()
        defer { lock.unlock() }
        
        let selectedSeats = seats.filter { ids.contains($0.id) }
        
        // Check if all seats are available
        guard selectedSeats.allSatisfy({ $0.isAvailable && !$0.isSelected && !$0.isReserved }) else {
            return (false, "One or more selected seats are not available.")
        }
        
        // Select the seats
        for i in 0..<seats.count {
            if ids.contains(seats[i].id) {
                seats[i].isSelected = true
            }
        }
        
        // Start reservation timer
        startReservationTimer(forUser: userId)
        
        return (true, "Seats selected successfully.")
    }
    
    // Function to get available seats
    func getAvailableSeats() -> [Seat] {
        lock.lock()
        defer { lock.unlock() }
        return seats.filter { $0.isAvailable && !$0.isSelected && !$0.isReserved }
    }
    
    // Function to get seats by type
    func getSeatsByType(_ type: SeatType) -> [Seat] {
        lock.lock()
        defer { lock.unlock() }
        return seats.filter { $0.type == type }
    }
    
    // Function to handle group seating
    func selectGroupSeats(count: Int, preferredType: SeatType? = nil) -> (success: Bool, seats: [Seat]) {
        lock.lock()
        defer { lock.unlock() }
        
        var availableSeats = seats.filter { $0.isAvailable && !$0.isSelected && !$0.isReserved }
        if let type = preferredType {
            availableSeats = availableSeats.filter { $0.type == type }
        }
        
        // Find a contiguous block of seats
        for i in 0...availableSeats.count - count {
            let group = Array(availableSeats[i..<i+count])
            if group.map({ $0.id }).sorted() == Array(group.first!.id...group.last!.id) {
                // Select these seats
                for seat in group {
                    if let index = seats.firstIndex(where: { $0.id == seat.id }) {
                        seats[index].isSelected = true
                    }
                }
                return (true, group)
            }
        }
        
        return (false, [])
    }
    
    // Function to confirm seat selection (e.g., after payment)
    func confirmSelection(forUser userId: String) {
        lock.lock()
        defer { lock.unlock() }
        
        for i in 0..<seats.count {
            if seats[i].isSelected {
                seats[i].isSelected = false
                seats[i].isReserved = true
                seats[i].isAvailable = false
            }
        }
        
        // Stop the reservation timer
        reservationTimer?.invalidate()
    }
    
    // Function to release seats if payment is not completed
    private func releaseSeats(forUser userId: String) {
        lock.lock()
        defer { lock.unlock() }
        
        for i in 0..<seats.count {
            if seats[i].isSelected {
                seats[i].isSelected = false
            }
        }
    }
    
    // Function to start reservation timer
    private func startReservationTimer(forUser userId: String) {
        reservationTimer?.invalidate()
        reservationTimer = Timer.scheduledTimer(withTimeInterval: reservationTimeout, repeats: false) { [weak self] _ in
            self?.releaseSeats(forUser: userId)
        }
    }
}

// Usage example
let seatManager = SeatSelectionManager(seatCount: 40)

// Select seats
let result = seatManager.selectSeats(ids: [1, 2, 3], forUser: "user123")
print(result.message)

// Get available seats
let availableSeats = seatManager.getAvailableSeats()
print("Available seats: \(availableSeats.count)")

// Select group seats
let groupResult = seatManager.selectGroupSeats(count: 2, preferredType: .window)
if groupResult.success {
    print("Group seats selected: \(groupResult.seats.map { $0.id })")
} else {
    print("Unable to find contiguous group seats")
}

// Confirm selection (after payment)
seatManager.confirmSelection(forUser: "user123")