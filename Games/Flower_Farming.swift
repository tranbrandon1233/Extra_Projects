import Foundation

/// Represents a flower type in the game
enum FlowerType: String, CaseIterable {
    case empty = "🟫"
    case sunflower = "🌻"
    case daffodil = "🌼"
    case rose = "🌹"

    /// Returns the growth time for each flower type in seconds
    var growthTime: TimeInterval {
        switch self {
        case .sunflower: return 3
        case .daffodil: return 10
        case .rose: return 15
        default: return 0
        }
    }
}

/// Represents a plot in the farm grid
class Plot {
    var flower: FlowerType = .empty
    var plantedTime: Date?
    var harvestNotified: Bool = false
    
    /// Checks if the flower in the plot is ready to harvest
    var isReadyToHarvest: Bool {
        guard let plantedTime = plantedTime, flower != .empty else { return false }
        return Date().timeIntervalSince(plantedTime) >= flower.growthTime
    }
    
    /// Updates the plot and checks if a harvest notification should be sent
    /// - Returns: True if the flower is ready to harvest and hasn't been notified yet, false otherwise
    func update() -> Bool {
        if isReadyToHarvest && !harvestNotified {
            harvestNotified = true
            return true
        }
        return false
    }
}

class FarmSimulator {
    let gridSize: Int
    var grid: [[Plot]]
    var inventory: [FlowerType: Int] = [:]
    
    /// Initializes a new FarmSimulator with the specified grid size
    /// - Parameter size: The size of the farm grid (size x size)
    init(size: Int) {
        self.gridSize = size
        self.grid = (0..<size).map { _ in (0..<size).map { _ in Plot() } }
        FlowerType.allCases.forEach { inventory[$0] = 0 }
    }
    
    /// Plants a flower in the specified plot
    /// - Parameters:
    ///   - flower: The type of flower to plant
    ///   - row: The row index of the plot
    ///   - col: The column index of the plot
    func plantFlower(_ flower: FlowerType, at row: Int, col: Int) {
        guard row >= 0, row < gridSize, col >= 0, col < gridSize else { 
            print("Invalid plot selected.")
            return 
        }
        let plot = grid[row][col]
        if plot.flower == .empty {
            plot.flower = flower
            plot.plantedTime = Date()
            plot.harvestNotified = false
        }
    }
    
    /// Harvests a flower from the specified plot if it's ready
    /// - Parameters:
    ///   - row: The row index of the plot
    ///   - col: The column index of the plot
    func harvestFlower(at row: Int, col: Int) {
        guard row >= 0, row < gridSize, col >= 0, col < gridSize else { return }
        let plot = grid[row][col]
        if plot.isReadyToHarvest {
            print("Harvested \(plot.flower) at (\(row),\(col)).")
            inventory[plot.flower, default: 0] += 1
            plot.flower = .empty
            plot.plantedTime = nil
            plot.harvestNotified = false
        } else {
            print("This flower is not ready to be harvested yet.")
        }
    }
    
    /// Updates all plots in the grid and prints harvest notifications
    func updateGrid() {
        for row in 0..<gridSize {
            for col in 0..<gridSize {
                let plot = grid[row][col]
                if plot.update() {
                    print("\(plot.flower) at (\(row),\(col)) is ready to be harvested.")
                }
            }
        }
    }
    
    /// Displays the current state of the farm grid
    func displayGrid() {
        for row in grid {
            let rowString = row.map { $0.flower.rawValue }.joined(separator: " ")
            print(rowString)
        }
    }
    
    /// Displays the current inventory
    func displayInventory() {
        print("Inventory:")
        for (flower, count) in inventory where flower != .empty  {
            print("\(flower.rawValue): \(count)")
        }
    }
}

class GameManager {
    var isRunning = true
    var farm: FarmSimulator!
    private let updateQueue = DispatchQueue(label: "com.farmsimulator.updateQueue")
    
    /// Starts the game, initializes the farm, and handles user input
    func start() {
        print("Welcome to the Farming Simulator!")
        print("Enter the size of the farm grid:")
        guard let input = readLine(), let gridSize = Int(input), gridSize > 0 else {
            print("Invalid input. Please enter a positive number.")
            return
        }
        
        farm = FarmSimulator(size: gridSize)
        
        print("Commands: plant [sunflower/daffodil/rose] [row] [col], harvest [row] [col], inventory, quit")
        
        // Start periodic updates using GCD
        startPeriodicUpdates()
        
        while isRunning {
            farm.displayGrid()
            print("Enter a command:")
            guard let input = readLine()?.lowercased().split(separator: " ") else { continue }
            
            switch input[0] {
            case "plant":
                handlePlantCommand(input)
            case "harvest":
                handleHarvestCommand(input)
            case "inventory":
                farm.displayInventory()
            case "quit":
                isRunning = false
            default:
                print("Invalid command")
            }
        }
        
        print("Thanks for playing!")
    }
    
    /// Starts a background task to periodically update the farm grid
    private func startPeriodicUpdates() {
        updateQueue.async {
            while self.isRunning {
                self.farm.updateGrid()
                Thread.sleep(forTimeInterval: 1.0)
            }
        }
    }
    
    /// Handles the plant command
    /// - Parameter input: The user input split into components
    private func handlePlantCommand(_ input: [Substring]) {
        if input.count == 4,
           let flowerType = getFlowerType(from: String(input[1])),
           let row = Int(input[2]),
           let col = Int(input[3]) {
            farm.plantFlower(flowerType, at: row, col: col)
        } else {
            print("Invalid plant command. Usage: plant [sunflower/daffodil/rose] [row] [col]")
        }
    }
    
    /// Converts a string to a FlowerType
    /// - Parameter flowerString: The string representation of the flower
    /// - Returns: The corresponding FlowerType, or nil if invalid
    private func getFlowerType(from flowerString: String) -> FlowerType? {
        switch flowerString {
        case "sunflower":
            return .sunflower
        case "daffodil":
            return .daffodil
        case "rose":
            return .rose
        default:
            return nil
        }
    }
    
    /// Handles the harvest command
    /// - Parameter input: The user input split into components
    private func handleHarvestCommand(_ input: [Substring]) {
        if input.count == 3, let row = Int(input[1]), let col = Int(input[2]) {
            farm.harvestFlower(at: row, col: col)
        } else {
            print("Invalid harvest command. Usage: harvest [row] [col]")
        }
    }
}

// Start the game
let gameManager = GameManager()
gameManager.start()