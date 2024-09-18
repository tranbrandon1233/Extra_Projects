
/**
 * Hello world!
 */
public class ThreadSafeBankAccount {
    private double balance;
    private final Object lock = new Object();

    public ThreadSafeBankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    public void deposit(double amount) {
        synchronized(lock) {
            if (amount < 0) {
                throw new IllegalArgumentException("Deposit amount cannot be negative.");
            }
            balance += amount;
        }
    }

    public void withdraw(double amount) throws InsufficientFundsException {
        synchronized(lock) {
            if (amount < 0) {
                throw new IllegalArgumentException("Withdrawal amount cannot be negative.");
            }
            if (amount > balance) {
                throw new InsufficientFundsException("Insufficient funds.");
            }
            balance -= amount;
        }
    }

    public double getBalance() {
        synchronized(lock) {
            return balance;
        }
    }
}

class InsufficientFundsException extends Exception {
    public InsufficientFundsException(String message) {
        super(message);
    }
}

