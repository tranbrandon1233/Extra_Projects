import Foundation

// Define custom error types
enum BankingError: Error {
    case insufficientFunds(amountNeeded: Double)
    case accountNotFound
}

// Base class for a bank account
class BankAccount {
    let accountNumber: String
    var balance: Double
    
    init(accountNumber: String, initialBalance: Double) {
        self.accountNumber = accountNumber
        self.balance = initialBalance
    }
    
    func deposit(amount: Double) {
        balance += amount
        print("Deposited \(amount). New balance is \(balance).")
    }
    
    func withdraw(amount: Double) throws {
        guard amount <= balance else {
            throw BankingError.insufficientFunds(amountNeeded: amount - balance)
        }
        balance -= amount
        print("Withdrew \(amount). New balance is \(balance).")
    }
}

// Subclass for a savings account
class SavingsAccount: BankAccount {
    let interestRate: Double
    
    init(accountNumber: String, initialBalance: Double, interestRate: Double) {
        self.interestRate = interestRate
        super.init(accountNumber: accountNumber, initialBalance: initialBalance)
    }
    
    func applyInterest() {
        let interest = balance * interestRate
        deposit(amount: interest)
        print("Applied interest of \(interest). New balance is \(balance).")
    }
}

// Bank class to manage multiple accounts
class Bank {
    private var accounts: [String: BankAccount] = [:]
    
    func addAccount(account: BankAccount) {
        accounts[account.accountNumber] = account
        print("Account \(account.accountNumber) added.")
    }
    
    func findAccount(accountNumber: String) throws -> BankAccount {
        guard let account = accounts[accountNumber] else {
            throw BankingError.accountNotFound
        }
        return account
    }
}

// Example usage
let bank = Bank()
let savingsAccount = SavingsAccount(accountNumber: "123456", initialBalance: 1000.0, interestRate: 0.05)

bank.addAccount(account: savingsAccount)

do {
    let account = try bank.findAccount(accountNumber: "123456")
    try account.withdraw(amount: 200.0)
    account.deposit(amount: 500.0)
    if let savings = account as? SavingsAccount {
        savings.applyInterest()
    }
} catch BankingError.insufficientFunds(let amountNeeded) {
    print("Insufficient funds. You need \(amountNeeded) more.")
} catch BankingError.accountNotFound {
    print("Account not found.")
} catch {
    print("An unexpected error occurred: \(error).")
}