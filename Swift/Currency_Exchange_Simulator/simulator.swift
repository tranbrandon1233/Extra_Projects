import Foundation

/// Represents a currency with a name and exchange rate.
struct Currency {
    let name: String
    var rate: Double
}

/// Simulates a foreign exchange (forex) market.
class ForexSimulator {
    var currencies: [Currency]
    var alerts: [(Currency, Currency, Double, (Currency, Currency, Double) -> Void)]
    var timer: Timer?

    /// Initializes the ForexSimulator with default currencies and empty alerts.
    init() {
        currencies = [
            Currency(name: "USD", rate: 1.0), // Base currency
            Currency(name: "EUR", rate: 0.85),
            Currency(name: "GBP", rate: 0.73),
            Currency(name: "RUB", rate: 73.5)
        ]
        alerts = []
    }

    /// Starts the simulation by scheduling a timer to update rates every second.
    func startSimulation() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateRates()
        }
        RunLoop.current.run() // Keeps the run loop running to allow timer to fire.
    }

    /// Updates the exchange rates of all currencies except the base currency (USD).
    func updateRates() {
        DispatchQueue.concurrentPerform(iterations: currencies.count - 1) { i in
            let index = i + 1 // Skip USD (index 0)
            let change = Double.random(in: -0.05...0.05) // Random change between -5% and +5%
            currencies[index].rate *= (1 + change) // Apply the change to the rate
        }
        printRates() // Print updated rates
        checkAlerts() // Check if any alerts need to be triggered
    }

    /// Prints the current exchange rates of all currencies.
    func printRates() {
        for currency in currencies {
            print("\(currency.name): \(currency.rate)")
        }
        print("--------------------")
    }

    /// Sets an alert for a specific exchange rate threshold between two currencies.
    /// - Parameters:
    ///   - from: The name of the base currency.
    ///   - to: The name of the target currency.
    ///   - threshold: The exchange rate threshold to trigger the alert.
    ///   - action: The action to perform when the alert is triggered.
    func setAlert(from: String, to: String, threshold: Double, action: @escaping (Currency, Currency, Double) -> Void) {
        guard let fromCurrency = currencies.first(where: { $0.name == from }),
              let toCurrency = currencies.first(where: { $0.name == to }) else {
            print("Invalid currency names")
            return
        }
        alerts.append((fromCurrency, toCurrency, threshold, action)) // Add the alert to the list
    }

    /// Checks if any alerts need to be triggered based on the current exchange rates.
    func checkAlerts() {
        alerts = alerts.filter { (fromCurrency, toCurrency, threshold, action) in
            guard let fromCurrency = currencies.first(where: { $0.name == fromCurrency.name }),
                  let toCurrency = currencies.first(where: { $0.name == toCurrency.name }) else {
                print("Invalid currency names")
                return false
            }
            let rate = toCurrency.rate / fromCurrency.rate // Calculate the exchange rate
            if rate >= threshold {
                action(fromCurrency, toCurrency, rate) // Trigger the alert action
                return false // Remove this alert
            }
            return true // Keep this alert
        }
    }
}

// Usage
let simulator = ForexSimulator()

simulator.setAlert(from: "GBP", to: "USD", threshold: 1.15) { from, to, rate in
    print("Alert: 1 \(from.name) is now worth \(rate) \(to.name)")
}

simulator.setAlert(from: "USD", to: "RUB", threshold: 80) { from, to, rate in
    print("Alert: 1 \(from.name) is now worth \(rate) \(to.name)")
}

simulator.startSimulation()
