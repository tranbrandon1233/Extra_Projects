import Foundation

// Item struct
struct Item {
    let name: String
    let price: Int
}

// Inventory struct
struct Inventory {
    var items: [Item]
    let capacity: Int
    
    var isFull: Bool {
        return items.count >= capacity
    }
}

// Player class
class Player {
    var gold: Int
    var inventory: Inventory
    
    init(gold: Int, inventoryCapacity: Int) {
        self.gold = gold
        self.inventory = Inventory(items: [], capacity: inventoryCapacity)
    }
}

// Shop class
class Shop {
    var items: [Item]
    
    init(items: [Item]) {
        self.items = items
    }
    
    func displayItems() {
        for (index, item) in items.enumerated() {
            print("\(index + 1). \(item.name) - \(item.price) gold")
        }
    }
}

// Purchase Manager class
class PurchaseManager {
    func purchaseItem(_ item: Item, for player: Player) -> Bool {
        // Ensure the player has enough gold for the item
        guard player.gold >= item.price else {
            print("Not enough gold to purchase \(item.name)")
            return false
        }
        // Ensure the inventory is not full
        guard !player.inventory.isFull else {
            print("Inventory is full. Cannot purchase \(item.name)")
            return false
        }
        
        // Purchase the item and deduct the gold balance
        player.gold -= item.price
        player.inventory.items.append(item)
        print("Successfully purchased \(item.name)")
        return true
    }
}

// Usage example
let player = Player(gold: 100, inventoryCapacity: 5)
let shop = Shop(items: [
    Item(name: "Sword", price: 50),
    Item(name: "Armor", price: 75),
    Item(name: "Health Potion", price: 25),
    Item(name: "Mana Potion", price: 25)
])
let purchaseManager = PurchaseManager()

print("Player Gold: \(player.gold)")
print("Shop Items:")
shop.displayItems()


let success = purchaseManager.purchaseItem(shop.items[0], for: player)
let success2 = purchaseManager.purchaseItem(shop.items[2], for: player)
if success, success2 {
    print("Player Gold after purchase: \(player.gold)")
    print("Player Inventory:\n \(player.inventory.items.map { $0.name }.joined(separator: "\n "))")
}
else{
    print("Sorry, the purchases were not successful.")
}
