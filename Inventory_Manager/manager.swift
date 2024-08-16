import Foundation

// Define the Inventory Item model
struct InventoryItem {
    var id: String
    var name: String
    var quantity: Int
    var threshold: Int
    var expirationDate: Date?
}

// Define the User Role enumeration
enum UserRole {
    case admin, manager, stockClerk
}

// Define the Transaction model
struct InventoryTransaction {
    var id: String
    var type: TransactionType
    var quantity: Int
    var date: Date
}

// Define the Transaction Type enumeration with `return` renamed to `itemReturn`
enum TransactionType {
    case purchase, sale, itemReturn, adjustment
}

// Inventory Manager class
class InventoryManager {
    private var items: [String: InventoryItem] = [:]
    private var transactions: [InventoryTransaction] = []
    
    func addItem(item: InventoryItem) {
        items[item.id] = item
    }
    
    func addTransaction(transaction: InventoryTransaction) {
        transactions.append(transaction)
        processTransaction(transaction)
    }
    
    func processTransaction(_ transaction: InventoryTransaction) {
        guard var item = items[transaction.id] else { return }
        
        switch transaction.type {
        case .purchase:
            item.quantity += transaction.quantity
        case .sale:
            item.quantity -= transaction.quantity
        case .itemReturn:
            item.quantity += transaction.quantity
        case .adjustment:
            item.quantity = transaction.quantity
        }
        
        items[item.id] = item
        checkStockLevels(item)
    }
    
    func generateReport() -> String {
        // Generate and return a report
        var report = "Inventory Report:\n"
        for item in items.values {
            report += "\(item.name): \(item.quantity) units\n"
        }
        return report
    }
    
    func getItems() -> [InventoryItem] {
        return Array(items.values)
    }
    
    func checkStockLevels(_ item: InventoryItem) {
        if item.quantity < item.threshold {
            AlertManager().sendAlert(for: item)
        }
    }
}

// User Manager class
class UserManager {
    private var users: [String: UserRole] = [:]
    
    func addUser(userId: String, role: UserRole) {
        users[userId] = role
    }
    
    func getUserRole(userId: String) -> UserRole? {
        return users[userId]
    }
}

// Alert Manager class
class AlertManager {
    func sendAlert(for item: InventoryItem) {
        print("Alert: Low stock for \(item.name)! Current quantity: \(item.quantity)")
    }
}

// Reorder Manager class
class ReorderManager {
    func checkAndReorder(item: InventoryItem) {
        if item.quantity < item.threshold {
            // Trigger reorder
            print("Reordering item: \(item.name)")
        }
    }
}

// Report Generator class
class ReportGenerator {
    func createDetailedReport(items: [InventoryItem]) -> String {
        var report = "Detailed Inventory Report:\n"
        for item in items {
            report += "\(item.name): \(item.quantity) units, Threshold: \(item.threshold)\n"
        }
        return report
    }
}

// Supply Chain Integrator class
class SupplyChainIntegrator {
    func synchronizeWithExternalSystem() {
        // Example implementation for synchronization
        print("Synchronizing inventory with supply chain system.")
    }
}

// Example Usage

// Create an instance of InventoryManager
let inventoryManager = InventoryManager()

// Add inventory items
inventoryManager.addItem(item: InventoryItem(id: "1", name: "Widget A", quantity: 50, threshold: 20, expirationDate: nil))
inventoryManager.addItem(item: InventoryItem(id: "2", name: "Widget B", quantity: 30, threshold: 15, expirationDate: nil))

// Add transactions
inventoryManager.addTransaction(transaction: InventoryTransaction(id: "1", type: .sale, quantity: 10, date: Date()))
inventoryManager.addTransaction(transaction: InventoryTransaction(id: "2", type: .itemReturn, quantity: 5, date: Date()))

// Generate and print report
print(inventoryManager.generateReport())

// Example user management
let userManager = UserManager()
userManager.addUser(userId: "user1", role: .admin)
print("User Role: \(String(describing: userManager.getUserRole(userId: "user1")))")

// Example alerts and reordering
let alertManager = AlertManager()
let reorderManager = ReorderManager()
for item in inventoryManager.getItems() {
    alertManager.sendAlert(for: item)
    reorderManager.checkAndReorder(item: item)
}

// Example detailed report generation
let reportGenerator = ReportGenerator()
print(reportGenerator.createDetailedReport(items: inventoryManager.getItems()))

// Example supply chain synchronization
let supplyChainIntegrator = SupplyChainIntegrator()
supplyChainIntegrator.synchronizeWithExternalSystem()
