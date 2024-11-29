import Foundation

enum UserInput: Error {
    case invalidNumber
    case outOfRange
    case emptyInput
    
    var message: String {
        switch self {
        case .invalidNumber:
            return "Please enter a number between 1-5 or 'c' for size comparison."
        case .outOfRange:
            return "Selection out of range. Please choose a number from the menu (1-5)."
        case .emptyInput:
            return "No input detected. Please make a selection."
        }
    }
}

enum SaturnCategory: String, CaseIterable {
    case basicFacts = "Basic Facts"
    case atmosphere = "Atmosphere"
    case rings = "Ring System"
    case moons = "Moons"
    case structure = "Structure"
    
    var description: String {
        switch self {
        case .basicFacts:
            return """
            🪐 Saturn Basic Facts (Source: NASA)
            - Equatorial diameter: 74,897 miles (120,500 kilometers)
            - Distance from Sun: 886 million miles (1.4 billion kilometers)
            - One day: 10.7 hours
            - One year: 29.4 Earth years (10,756 Earth days)
            - Axis tilt: 26.73 degrees
            """
        case .atmosphere:
            return """
            🌪 Atmosphere (Source: NASA)
            - Shows faint stripes, jet streams, and storms
            - Winds reach 1,600 feet per second in equatorial region
            - Notable hexagon-shaped jet stream at north pole
            - Hexagon spans 20,000 miles with 200-mile-per-hour winds
            - Appears in shades of yellow, brown, and gray
            """
        case .rings:
            return """
            💫 Ring System (Source: NASA)
            - Extends up to 175,000 miles from planet
            - Vertical height: about 30 feet in main rings
            - Made of ice and rock coated with dust
            - Main rings: A, B, and C
            - Cassini Division: 2,920 miles wide gap between Rings A and B
            """
        case .moons:
            return """
            🛸 Moons (Source: NASA)
            - Current confirmed count: 146 moons (as of June 8, 2023)
            - Notable moons include Titan and Enceladus
            - Titan has methane lakes
            - Enceladus has water jets
            - Each moon contributes to Saturn system's story
            """
        case .structure:
            return """
            ⚡️ Structure (Source: NASA)
            - Primarily composed of hydrogen and helium
            - Dense core contains metals like iron and nickel
            - Surrounded by liquid metallic hydrogen
            - Only planet less dense than water
            - Magnetic field 578 times as powerful as Earth's
            """
        }
    }
}

class SaturnExplorer {
    private let saturnArt = """
        
          ╭────────────╮
     ╭────┤  SATURN   ├────╮
    ╭┤    ╰────────────╯    ├╮
    ││     ╭──────────╮     ││
    ╰┤     │          │     ├╯
     ╰─────┤          ├─────╯
           │          │
           ╰──────────╯
    
    """
    
    private let sizeComparison = """
    NASA Size Comparison:
    Earth: ○
    Saturn: ⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️
    (Saturn is 9 times wider than Earth - if Earth were the size of a nickel, 
     Saturn would be about as big as a volleyball)
    """
    
    private var hasSeenTutorial = false
    private var consecutiveErrors = 0
    private let maxConsecutiveErrors = 3
    
    private func showTutorial() {
        print("""
        
        📚 Quick Tutorial:
        • Use number keys (1-5) to explore different categories
        • Press 'c' to see size comparison
        • Press 'h' for help
        • Press 'q' to quit
        
        Example: Press '1' to learn basic facts about Saturn
        
        Let's begin your exploration! 🚀
        """)
        hasSeenTutorial = true
    }
    
    private func showHelp() {
        print("""
        
        ℹ️ Help Menu:
        • Numbers 1-5: Select a category to explore
        • 'c': View Saturn-Earth size comparison
        • 'h': Show this help menu
        • 'q': Quit the explorer
        
        Tips:
        - Each category contains verified NASA facts
        - Take your time to explore each section
        - Invalid inputs will be guided to correct options
        """)
    }
    
    func start() {
        print("\n\(saturnArt)")
        print("Welcome to the NASA Saturn Facts Explorer! 🪐")
        print("All facts sourced directly from NASA's Saturn information page.")
        
        if !hasSeenTutorial {
            showTutorial()
        }
        
        while true {
            do {
                try displayMenuAndHandleInput()
            } catch {
                handleError(error)
            }
        }
    }
    
    private func displayMenuAndHandleInput() throws {
        displayMenu()
        
        guard let input = readLine()?.lowercased(), !input.isEmpty else {
            throw UserInput.emptyInput
        }
        
        switch input {
        case "q":
            print("\nThank you for exploring NASA's Saturn facts! 👋")
            exit(0)
        case "h":
            showHelp()
            consecutiveErrors = 0
        case "c":
            print("\n" + sizeComparison)
            consecutiveErrors = 0
        case let str where Int(str) != nil:
            try handleNumberedChoice(str)
            consecutiveErrors = 0
        default:
            throw UserInput.invalidNumber
        }
    }
    
    private func handleNumberedChoice(_ choice: String) throws {
        guard let number = Int(choice) else {
            throw UserInput.invalidNumber
        }
        
        guard (1...SaturnCategory.allCases.count).contains(number) else {
            throw UserInput.outOfRange
        }
        
        let category = SaturnCategory.allCases[number - 1]
        print("\n" + category.description)
    }
    
    private func displayMenu() {
        print("\nSelect a category to explore:")
        print("---------------------------")
        for (index, category) in SaturnCategory.allCases.enumerated() {
            print("\(index + 1). \(category.rawValue)")
        }
        print("""
        
        Additional options:
        'c' - View Size Comparison
        'h' - Help Menu
        'q' - Quit Explorer
        """)
    }
    
    private func handleError(_ error: Error) {
        consecutiveErrors += 1
        
        if let inputError = error as? UserInput {
            print("\n❌ \(inputError.message)")
        } else {
            print("\n❌ An unexpected error occurred. Please try again.")
        }
        
        if consecutiveErrors >= maxConsecutiveErrors {
            print("""
            
            ℹ️ It seems you might need some help!
            Here are some valid input examples:
            • Type '1' to learn about Saturn's basic facts
            • Type 'c' to see the size comparison
            • Type 'h' for the help menu
            """)
            consecutiveErrors = 0
        }
    }
}

// Start the explorer
let explorer = SaturnExplorer()
explorer.start()