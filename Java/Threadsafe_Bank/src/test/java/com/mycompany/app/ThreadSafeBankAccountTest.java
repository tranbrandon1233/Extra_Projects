import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

public class ThreadSafeBankAccountTest {

    @Test
    void testDepositPositiveAmount() {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(100);
        account.deposit(50);
        assertEquals(150, account.getBalance());
    }

    @Test
    void testDepositNegativeAmount() {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(100);
        assertThrows(IllegalArgumentException.class, () -> account.deposit(-50));
    }

    @Test
    void testWithdrawPositiveAmount() throws InsufficientFundsException {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(100);
        account.withdraw(50);
        assertEquals(50, account.getBalance());
    }

    @Test
    void testWithdrawNegativeAmount() {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(100);
        assertThrows(IllegalArgumentException.class, () -> account.withdraw(-50));
    }

    @Test
    void testWithdrawInsufficientFunds() {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(100);
        assertThrows(InsufficientFundsException.class, () -> account.withdraw(150));
    }

    @Test
    void testConcurrentDeposits() throws InterruptedException {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(0);
        int numThreads = 100;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            executor.execute(() -> account.deposit(1));
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.SECONDS);
        assertEquals(numThreads, account.getBalance());
    }

    @Test
    void testConcurrentWithdrawals() throws InterruptedException {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(1000);
        int numThreads = 100;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            executor.execute(() -> {
                try {
                    account.withdraw(1);
                } catch (InsufficientFundsException e) {
                    // This should not happen in this test case
                    fail("InsufficientFundsException should not be thrown");
                }
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.SECONDS);
        assertEquals(1000 - numThreads, account.getBalance());
    }

    @Test
    void testConcurrentDepositsAndWithdrawals() throws InterruptedException {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(1000);
        int numThreads = 100;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            if (i % 2 == 0) {
                executor.execute(() -> account.deposit(1));
            } else {
                executor.execute(() -> {
                    try {
                        account.withdraw(1);
                    } catch (InsufficientFundsException e) {
                        // This should not happen in this test case
                        fail("InsufficientFundsException should not be thrown");
                    }
                });
            }
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.SECONDS);
        assertEquals(1000, account.getBalance());
    }

    @Test
    void testBoundaryDeposit() {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(0);
        account.deposit(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, account.getBalance());
    }

    @Test
    void testBoundaryWithdrawal() throws InsufficientFundsException {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(Double.MAX_VALUE);
        account.withdraw(Double.MAX_VALUE);
        assertEquals(0, account.getBalance());
    }

    @Test
    void testZeroDeposit() {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(100);
        account.deposit(0);
        assertEquals(100, account.getBalance());
    }

    @Test
    void testZeroWithdrawal() throws InsufficientFundsException {
        ThreadSafeBankAccount account = new ThreadSafeBankAccount(100);
        account.withdraw(0);
        assertEquals(100, account.getBalance());
    }
}