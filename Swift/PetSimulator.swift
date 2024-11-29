import Foundation

// Define a structure to represent the pet's state
struct PetState {
    var hunger: Int
    var happiness: Int
    var energy: Int
    var health: Int
}

// Define a class for the Pet
class Pet {
    var name: String
    var state: PetState
    var interactionCount: Int
    var feedCount: Int
    var playCount: Int
    var restCount: Int
    
    init(name: String) {
        self.name = name
        self.state = PetState(hunger: 50, happiness: 50, energy: 50, health: 100)
        self.interactionCount = 0
        self.feedCount = 0
        self.playCount = 0
        self.restCount = 0
    }
    
    func feed() {
        state.hunger = max(0, state.hunger - 10)
        state.health = min(100, state.health + 5)
        interactionCount += 1
        feedCount += 1
        advanceTime()
    }
    
    func play() {
        state.happiness = min(100, state.happiness + 10)
        state.energy = max(0, state.energy - 10)
        interactionCount += 1
        playCount += 1
        advanceTime()
    }
    
    func rest() {
        state.energy = min(100, state.energy + 20)
        state.hunger = min(100, state.hunger + 5)
        interactionCount += 1
        restCount += 1
        advanceTime()
    }
    
    func advanceTime() {
        state.hunger = min(100, state.hunger + 5)
        state.happiness = max(0, state.happiness - 5)
        state.energy = max(0, state.energy - 5)
        
        // Random event: illness
        if Int.random(in: 1...10) == 1 {
            print("\(name) got sick!")
            state.health = max(0, state.health - 20)
        }
    }
    
    func displayStatus() {
        print("\n\(name)'s Status:")
        print("Hunger: \(state.hunger)")
        print("Happiness: \(state.happiness)")
        print("Energy: \(state.energy)")
        print("Health: \(state.health)")
    }
    
    func overallWellBeing() -> Int {
        return (state.hunger + state.happiness + state.energy + state.health) / 4
    }
}

// Main program
func main() {
    print("Welcome to the Virtual Pet Care System!")
    print("Please enter a name for your pet:")
    let petName = readLine() ?? "Pet"
    let pet = Pet(name: petName)
    
    var isRunning = true
    
    while isRunning {
        pet.displayStatus()
        print("\nWhat would you like to do?")
        print("1. Feed")
        print("2. Play")
        print("3. Rest")
        print("4. Exit")
        
        if let choice = readLine(), let action = Int(choice) {
            switch action {
            case 1:
                pet.feed()
                print("\(pet.name) has been fed.")
            case 2:
                pet.play()
                print("\(pet.name) is playing.")
            case 3:
                pet.rest()
                print("\(pet.name) is resting.")
            case 4:
                isRunning = false
            default:
                print("Invalid choice. Please try again.")
            }
        }
    }
    
    // Display summary
    print("\nExiting the Virtual Pet Care System.")
    print("Summary of your pet's care:")
    print("Total interactions: \(pet.interactionCount)")
    print("Feed actions: \(pet.feedCount)")
    print("Play actions: \(pet.playCount)")
    print("Rest actions: \(pet.restCount)")
    print("Overall well-being score: \(pet.overallWellBeing())")
}

// Run the main program
main()