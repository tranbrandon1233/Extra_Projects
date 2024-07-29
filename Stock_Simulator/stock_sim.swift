import Foundation

/// `StockMarket` simulates a stock market where users can buy and sell shares.
class StockMarket {
    var availableShares: Int  // Number of shares available in the market
    var pricePerShare: Double  // Price of each share

    /// Initializes a new stock market.
    ///
    /// - Parameters:
    ///   - initialShares: The initial number of shares in the market.
    ///   - initialPrice: The initial price per share.
    init(initialShares: Int, initialPrice: Double) {
        self.availableShares = initialShares
        self.pricePerShare = initialPrice
    }

    /// Buys a certain quantity of shares.
    ///
    /// - Parameter quantity: The number of shares to buy.
    func buyShares(quantity: Int) {
        if quantity <= availableShares {
            availableShares -= quantity
            adjustPrice()
            print("You bought \(quantity) shares at $\(pricePerShare) each.")
        } else {
            print("Not enough shares available. Only \(availableShares) shares left.")
        }
    }

    /// Sells a certain quantity of shares.
    ///
    /// - Parameter quantity: The number of shares to sell.
    func sellShares(quantity: Int) {
        availableShares += quantity
        adjustPrice()
        print("You sold \(quantity) shares at $\(pricePerShare) each.")
    }

    /// Adjusts the price of the shares based on the number of available shares.
    private func adjustPrice() {
        pricePerShare = 10.0 + (1000000.0 - Double(availableShares)) / 100000.0
    }
    
    /// Displays the current status of the stock market.
    func displayStatus() {
        print("Available shares: \(availableShares)")
        print("Price per share: $\(pricePerShare)")
    }
}

// Create the stock market
let stockMarket = StockMarket(initialShares: 1000000, initialPrice: 10.0)

// Start an infinite loop for the stock market simulation
while true {
    // Print a new line and the title of the simulation
    print("\nStock Market Simulation")
    // Display the current status of the stock market
    stockMarket.displayStatus()
    // Ask the user for their next action
    print("Enter 'b' to buy shares, 's' to sell shares, or 'q' to quit:")
    // Read the user's input
    if let input = readLine() {
        // If the user wants to quit, break the loop
        if input == "q" {
            break
        } 
        // If the user wants to buy shares
        else if input == "b" {
            // Ask the user for the quantity of shares to buy
            print("Enter the number of shares to buy:")
            // Read the user's input for the quantity
            if let quantityStr = readLine(), let quantity = Int(quantityStr) {
                // Buy the entered quantity of shares
                stockMarket.buyShares(quantity: quantity)
            } else {
                // If the input is not a valid number, print an error message
                print("Invalid input. Please enter a valid number.")
            }
        } 
        // If the user wants to sell shares
        else if input == "s" {
            // Ask the user for the quantity of shares to sell
            print("Enter the number of shares to sell:")
            // Read the user's input for the quantity
            if let quantityStr = readLine(), let quantity = Int(quantityStr) {
                // Sell the entered quantity of shares
                stockMarket.sellShares(quantity: quantity)
            } else {
                // If the input is not a valid number, print an error message
                print("Invalid input. Please enter a valid number.")
            }
        } else {
            // If the input is not 'b', 's', or 'q', print an error message
            print("Invalid input. Please enter 'b', 's', or 'q'.")
        }
    }
}

