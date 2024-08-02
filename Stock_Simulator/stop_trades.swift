import Foundation

class StockSimulator {
    var currentPrice: Double
    var initialPrice: Double
    var holdings: Int = 0
    var totalProfit: Double = 0.0
    var setSellPrice: Double
    var setBuyPrice: Double
    
    init(initialPrice: Double) {
        self.currentPrice = initialPrice
        self.initialPrice = initialPrice
        let percentage = Double.random(in: 1...5)
        self.setSellPrice = initialPrice*(1-percentage/100)
        self.setBuyPrice = initialPrice*(1+percentage/100)
    }
    
    func setStopBuy(percentage: Double) {
        print("Stop buy set at \(percentage)% above $\(String(format: "%.2f", currentPrice))")
        setBuyPrice = currentPrice*(1+percentage/100)
    }
    
    func setStopSell(percentage: Double) {
        print("Stop sell set at \(percentage)% below $\(String(format: "%.2f", currentPrice))")
        setSellPrice = currentPrice*(1-percentage/100)
    }
    
    func updatePrice() {
        let changePercentage = Double.random(in: -5...5)
        let change = currentPrice * changePercentage / 100
        currentPrice += change
        currentPrice = max(0.01, currentPrice) // Ensure price doesn't go below 0.01
        
        checkStopOrders()
    }
    
    func checkStopOrders() {

            if currentPrice >= setBuyPrice {
                buy()
            }
        
        
        if holdings > 0 {
            if currentPrice <= setSellPrice {
                sell()
            }
        }
    }
    
    func buy() {
        holdings += 1
        let cost = currentPrice
        print("Bought 1 share at $\(String(format: "%.2f", cost))")
        initialPrice = currentPrice // Reset initial price for next stop order
        setBuyPrice = 999999
    }
    
    func sell() {
        if holdings > 0 {
            holdings -= 1
            let profit = currentPrice - initialPrice
            totalProfit += profit
            print("Sold 1 share at $\(String(format: "%.2f", currentPrice))")
            print("Profit on this trade: $\(String(format: "%.2f", profit))")
            print("Total profit: $\(String(format: "%.2f", totalProfit))")
            initialPrice = currentPrice // Reset initial price for next stop order
            setSellPrice = -1
        } else {
            print("No shares to sell")
        }
    }
    
    func displayStatus() {
        print("Current price: $\(String(format: "%.2f", currentPrice))")
        print("Holdings: \(holdings) shares")
        print("Total profit: $\(String(format: "%.2f", totalProfit))")
    }
}

// Usage example
let simulator = StockSimulator(initialPrice: 100.0)

for i in 1...100 {
    simulator.updatePrice()
    
    if i % 10 == 0 { // Set new stop orders every 10 iterations
        simulator.setStopBuy(percentage: Double.random(in: 1...5))
        simulator.setStopSell(percentage: Double.random(in: 1...5))
    }
    
    simulator.displayStatus()
    print("---")
    Thread.sleep(forTimeInterval: 0.5) // Simulate passage of time
}