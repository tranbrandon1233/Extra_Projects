import Foundation

// Item struct to represent inventory items
struct Item {
    let name: String
    var quantity: Int
}

// InventoryManager class to handle inventory operations
class InventoryManager {
    private var inventory: [Item] = []
    
    // Add a new item or update existing item quantity
    func addItem(name: String, quantity: Int) {
        if let index = inventory.firstIndex(where: { $0.name.lowercased() == name.lowercased() }) {
            inventory[index].quantity += quantity
        } else {
            inventory.append(Item(name: name, quantity: quantity))
        }
    }
    
    // Use a certain quantity of an item
    func useItem(name: String, quantity: Int) -> Bool {
        if let index = inventory.firstIndex(where: { $0.name.lowercased() == name.lowercased() }) {
            if inventory[index].quantity >= quantity {
                inventory[index].quantity -= quantity
                return true
            }
        }
        return false
    }
    
    // Remove an item entirely from the inventory
    func removeItem(name: String) -> Bool {
        if let index = inventory.firstIndex(where: { $0.name.lowercased() == name.lowercased() }) {
            inventory.remove(at: index)
            return true
        }
        return false
    }
    
    // Display current inventory
    func displayInventory() {
        print("Current Inventory:")
        for item in inventory {
            print("- \(item.name): \(item.quantity)")
        }
        print()
    }
}

// Main program
func main() {
    let manager = InventoryManager()
    
    while true {
        print("Inventory Management System")
        print("1. Add Item")
        print("2. Use Item")
        print("3. Remove Item")
        print("4. Display Inventory")
        print("5. Exit")
        print("Enter your choice: ", terminator: "")
        
        guard let choice = readLine(), let option = Int(choice) else {
            print("Invalid input. Please try again.\n")
            continue
        }
        
        switch option {
        case 1:
            print("Enter item name: ", terminator: "")
            guard let name = readLine(), !name.isEmpty else {
                print("Invalid name. Please try again.\n")
                continue
            }
            print("Enter quantity: ", terminator: "")
            guard let quantityStr = readLine(), let quantity = Int(quantityStr), quantity > 0 else {
                print("Invalid quantity. Please try again.\n")
                continue
            }
            manager.addItem(name: name, quantity: quantity)
            print("Item added successfully.\n")
            
        case 2:
            print("Enter item name: ", terminator: "")
            guard let name = readLine(), !name.isEmpty else {
                print("Invalid name. Please try again.\n")
                continue
            }
            print("Enter quantity to use: ", terminator: "")
            guard let quantityStr = readLine(), let quantity = Int(quantityStr), quantity > 0 else {
                print("Invalid quantity. Please try again.\n")
                continue
            }
            if manager.useItem(name: name, quantity: quantity) {
                print("Item used successfully.\n")
            } else {
                print("Failed to use item. Check if it exists and has sufficient quantity.\n")
            }
            
        case 3:
            print("Enter item name to remove: ", terminator: "")
            guard let name = readLine(), !name.isEmpty else {
                print("Invalid name. Please try again.\n")
                continue
            }
            if manager.removeItem(name: name) {
                print("Item removed successfully.\n")
            } else {
                print("Failed to remove item. Check if it exists in the inventory.\n")
            }
            
        case 4:
            manager.displayInventory()
            
        case 5:
            print("Exiting the program. Goodbye!")
            return
            
        default:
            print("Invalid option. Please try again.\n")
        }
    }
}

// Run the program
main()