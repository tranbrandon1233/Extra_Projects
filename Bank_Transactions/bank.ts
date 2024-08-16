class Transaction {
    private date: Date;
    private type: string;
    private amount: number;

    constructor(date: Date, type: string, amount: number) {
        this.date = date;
        this.type = type;
        this.amount = amount;
    }

    public getDate(): Date {
        return this.date;
    }

    public getType(): string {
        return this.type;
    }

    public getAmount(): number {
        return this.amount;
    }
}

class Account {
    private accountNumber: number;
    private balance: number;
    private transactions: Transaction[];

    constructor(accountNumber: number, initialBalance: number = 0) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
        this.transactions = [];
    }

    public deposit(amount: number): void {
        this.balance += amount;
        this.transactions.push(new Transaction(new Date(), "deposit", amount));
    }

    public withdraw(amount: number): void {
        if (this.balance >= amount) {
            this.balance -= amount;
            this.transactions.push(new Transaction(new Date(), "withdrawal", amount));
        } else {
            throw new Error("Insufficient funds");
        }
    }

    public transfer(amount: number, toAccount: Account): void {
        if (this.balance >= amount) {
            this.balance -= amount;
            toAccount.balance += amount;
            this.transactions.push(new Transaction(new Date(), "transfer", amount));
            toAccount.transactions.push(new Transaction(new Date(), "transfer", amount));
        } else {
            throw new Error("Insufficient funds");
        }
    }

    public getBalance(): number {
        return this.balance;
    }

    public getTransactions(start?: Date, end?: Date, type?: string): Transaction[] {
        let filteredTransactions: Transaction[] = this.transactions;
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
    private accounts: { [key: number]: Account };

    constructor() {
        this.accounts = {};
    }

    public createAccount(accountNumber: number, initialBalance: number = 0): void {
        this.accounts[accountNumber] = new Account(accountNumber, initialBalance);
    }

    public getAccount(accountNumber: number): Account {
        if (accountNumber in this.accounts) {
            return this.accounts[accountNumber];
        } else {
            throw new Error("Account not found");
        }
    }

    public saveToFile(filename: string): void {
        const data = JSON.stringify(this.accounts);
        const fs = require('fs');
        fs.writeFileSync(filename, data);
    }

    public loadFromFile(filename: string): void {
        try {
            const fs = require('fs');
            const data = fs.readFileSync(filename, 'utf8');
            const loadedAccounts = JSON.parse(data);
            for (const accountNumber in loadedAccounts) {
                const accountNum : number = parseInt(accountNumber);
                this.accounts[accountNum] = loadedAccounts[accountNum];
            }
        } catch (e) {
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