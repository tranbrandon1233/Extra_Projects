protocol Questing {
    var name: String { get }
    var description: String { get }
    var requiredItems: [Item] { get }
    var reward: Int { get }
    var isCompleted: Bool { get set }
}

protocol Inventoriable {
    var items: [Item] { get set }
    mutating func addItem(_ item: Item)
    mutating func removeItem(_ item: Item)
    func hasRequiredItems(for quest: Questing) -> Bool
}

protocol Walletable {
    var balance: Int { get set }
    mutating func addCoins(_ amount: Int)
}

struct Item: Equatable {
    let name: String
    let quantity: Int
}

struct Quest: Questing {
    let name: String
    let description: String
    let requiredItems: [Item]
    let reward: Int
    var isCompleted: Bool = false
}

struct Inventory: Inventoriable {
    var items: [Item] = []
    
    mutating func addItem(_ item: Item) {
        if let index = items.firstIndex(where: { $0.name == item.name }) {
            items[index] = Item(name: item.name, quantity: items[index].quantity + item.quantity)
        } else {
            items.append(item)
        }
    }
    
    mutating func removeItem(_ item: Item) {
        if let index = items.firstIndex(where: { $0.name == item.name }) {
            let currentQuantity = items[index].quantity
            if currentQuantity > item.quantity {
                items[index] = Item(name: item.name, quantity: currentQuantity - item.quantity)
            } else {
                items.remove(at: index)
            }
        }
    }
    
    func hasRequiredItems(for quest: Questing) -> Bool {
        for requiredItem in quest.requiredItems {
            guard let playerItem = items.first(where: { $0.name == requiredItem.name }),
                  playerItem.quantity >= requiredItem.quantity else {
                return false
            }
        }
        return true
    }
}

struct Wallet: Walletable {
    var balance: Int = 0
    
    mutating func addCoins(_ amount: Int) {
        balance += amount
    }
}

struct Player {
    var inventory: Inventory
    var wallet: Wallet
}

class QuestManager {
    var quests: [Quest] = []
    var player: Player
    
    init(player: Player) {
        self.player = player
    }
    
    func acceptQuest(_ quest: Quest) {
        quests.append(quest)
    }
    
    func submitQuest(_ quest: Quest) {
        guard player.inventory.hasRequiredItems(for: quest) else {
            print("You don't have the required items for this quest.")
            return
        }
        
        for item in quest.requiredItems {
            player.inventory.removeItem(item)
        }
        
        player.wallet.addCoins(quest.reward)
        
        if let index = quests.firstIndex(where: { $0.name == quest.name }) {
            quests[index].isCompleted = true
        }
        
        print("Quest completed! You earned \(quest.reward) gold coins.")
    }
    
    func viewQuests() {
        if quests.count == 0{
        print("No quests available.")
            return
        }
        for (index, quest) in quests.enumerated() {
            print("\(index + 1). \(quest.name) - \(quest.isCompleted ? "Completed" : "In Progress")")
        }
    }
    
    func addItemToPlayerInventory(_ item: Item) {
        player.inventory.addItem(item)
    }
}

// Implement key press functionality
func handleKeyPress(_ key: Character) {
    switch key {
    case "i":
        print("Inventory:")
        for item in questManager.player.inventory.items {
            print("\(item.name): \(item.quantity)")
        }
    case "w":
        print("Wallet balance: \(questManager.player.wallet.balance) gold coins")
    case "q":
        print("Quests:")
        questManager.viewQuests()
    default:
        print("Invalid key press")
    }
}

// Usage
let initialPlayer = Player(inventory: Inventory(), wallet: Wallet())
let questManager = QuestManager(player: initialPlayer)

// Simulate key presses
handleKeyPress("i")
handleKeyPress("w")
handleKeyPress("q")

// Add some items to the player's inventory
questManager.addItemToPlayerInventory(Item(name: "Sword", quantity: 1))
questManager.addItemToPlayerInventory(Item(name: "Health Potion", quantity: 5))

// Check inventory
handleKeyPress("i")

// Create a quest
let slayDragonsQuest = Quest(name: "Slay Dragons", description: "Defeat 3 dragons", requiredItems: [Item(name: "Dragon Scales", quantity: 3)], reward: 100)

// Accept the quest
questManager.acceptQuest(slayDragonsQuest)

// View quests
questManager.viewQuests()

// Submit the quest before receiving the items
questManager.submitQuest(slayDragonsQuest)

questManager.addItemToPlayerInventory(Item(name: "Dragon Scales", quantity: 3))
handleKeyPress("i")

// Submit the quest
questManager.submitQuest(slayDragonsQuest)

// Simulate key presses
handleKeyPress("i")
handleKeyPress("w")
handleKeyPress("q")