package main

import (
	"fmt"
	"math/rand"
	"time"
)

type StockExchange struct {
	stockPrice  float64
	sharesOwned int
	cash        float64
}

func (se *StockExchange) fluctuatePrice() {
	change := rand.Float64()*2 - 1 // Random change between -1 and 1
	se.stockPrice += change
	if se.stockPrice < 1 {
		se.stockPrice = 1 // Minimum stock price is 1
	}
}

func (se *StockExchange) buyShares(amount int) {
	cost := float64(amount) * se.stockPrice
	if cost > se.cash {
		fmt.Println("Not enough cash to buy shares!")
		return
	}
	se.cash -= cost
	se.sharesOwned += amount
	fmt.Printf("Bought %d shares at $%.2f each. Cash left: $%.2f\n", amount, se.stockPrice, se.cash)
}

func (se *StockExchange) sellShares(amount int) {
	if se.sharesOwned < 1 {
		fmt.Println("No shares to sell.")
	} else if amount > se.sharesOwned {
		revenue := float64(se.sharesOwned) * se.stockPrice
		se.cash += revenue
		se.sharesOwned = 0
		fmt.Printf("Sold %d shares at $%.2f each. Cash now: $%.2f\n", se.sharesOwned, se.stockPrice, se.cash)
	} else {
		revenue := float64(amount) * se.stockPrice
		se.cash += revenue
		se.sharesOwned -= amount
		fmt.Printf("Sold %d shares at $%.2f each. Cash now: $%.2f\n", amount, se.stockPrice, se.cash)
	}
}

func (se *StockExchange) displayStatus() {
	fmt.Printf("Stock Price: $%.2f, Shares Owned: %d, Cash: $%.2f\n", se.stockPrice, se.sharesOwned, se.cash)
}

func (se *StockExchange) calculateProfit(initialCash float64) {
	netWorth := se.cash + float64(se.sharesOwned)*se.stockPrice
	profit := netWorth - initialCash
	fmt.Printf("Net Worth: $%.2f, Profit/Loss: $%.2f\n", netWorth, profit)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	initialCash := 1000.0

	// Initialize the stock exchange
	se := StockExchange{
		stockPrice:  rand.Float64()*50 + 10, // Random initial price between 10 and 60
		cash: initialCash,
		sharesOwned: 0,
	}

	// Simulate stock exchange for 10 iterations
	for i := 1; i <= 10; i++ {
		fmt.Printf("\n--- Day %d ---\n", i)
		se.fluctuatePrice()
		se.displayStatus()

		action := rand.Intn(3) // Randomly decide to buy, sell, or hold
		switch action {
		case 0: // Buy shares
			se.buyShares(rand.Intn(5) + 1) // Randomly buy 1-5 shares
		case 1: // Sell shares
			se.sellShares(rand.Intn(5) + 1) // Randomly sell 1-5 shares
		default: // Hold
			fmt.Println("Holding shares today.")
		}
	}

	fmt.Println("\n--- Final Status ---")
	se.displayStatus()
	se.calculateProfit(initialCash)
}
