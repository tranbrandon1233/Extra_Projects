"use strict";
class Transaction {
    constructor(date, type, amount) {
        this.date = date;
        this.type = type;
        this.amount = amount;
    }
    getDate() {
        return this.date;
    }
    getType() {
        return this.type;
    }
    getAmount() {
        return this.amount;
    }
}
class Account {
    constructor(accountNumber, initialBalance = 0) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
        this.transactions = [];
    }
    deposit(amount) {
        this.balance += amount;
        this.transactions.push(new Transaction(new Date(), "deposit", amount));
    }
    withdraw(amount) {
        if (this.balance >= amount) {
            this.balance -= amount;
            this.transactions.push(new Transaction(new Date(), "withdrawal", amount));
        }
        else {
            throw new Error("Insufficient funds");
        }
    }
    transfer(amount, toAccount) {
        if (this.balance >= amount) {
            this.balance -= amount;
            toAccount.balance += amount;
            this.transactions.push(new Transaction(new Date(), "transfer", amount));
            toAccount.transactions.push(new Transaction(new Date(), "transfer", amount));
        }
        else {
            throw new Error("Insufficient funds");
        }
    }
    getBalance() {
        return this.balance;
    }
    getTransactions(start, end, type) {
        let filteredTransactions = this.transactions;
        if (start) {
            filteredTransactions = filteredTransactions.filter(transaction => transaction.getDate() >= start);
        }
        if (end) {
            filteredTransactions = filteredTransactions.filter(transaction => transaction.getDate() <= end);
        }
        if (type) {
            filteredTransactions = filteredTransactions.filter(transaction => transaction.getType() === type);
        }
        return filteredTransactions;
    }
}
class Bank {
    constructor() {
        this.accounts = {};
    }
    createAccount(accountNumber, initialBalance = 0) {
        this.accounts[accountNumber] = new Account(accountNumber, initialBalance);
    }
    getAccount(accountNumber) {
        if (accountNumber in this.accounts) {
            return this.accounts[accountNumber];
        }
        else {
            throw new Error("Account not found");
        }
    }
    saveToFile(filename) {
        const data = JSON.stringify(this.accounts);
        const fs = require('fs');
        fs.writeFileSync(filename, data);
    }
    loadFromFile(filename) {
        try {
            const fs = require('fs');
            const data = fs.readFileSync(filename, 'utf8');
            const loadedAccounts = JSON.parse(data);
            for (const accountNumber in loadedAccounts) {
                const accountNum = parseInt(accountNumber);
                this.accounts[accountNum] = loadedAccounts[accountNum];
            }
        }
        catch (e) {
            console.log(`Error loading from file: ${e}`);
        }
    }
}
// Example usage
const bank = new Bank();
bank.createAccount(12345, 1000);
const account = bank.getAccount(12345);
account.deposit(500);
account.withdraw(200);
account.transfer(300, bank.getAccount(12345)); // Note: In a real-world scenario, you would transfer to a different account
console.log(account.getBalance());
for (const transaction of account.getTransactions()) {
    console.log(`Date: ${transaction.getDate()}, Type: ${transaction.getType()}, Amount: ${transaction.getAmount()}`);
}
bank.saveToFile('bank.json');
bank.loadFromFile('bank.json');
