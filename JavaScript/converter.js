function processOrders(orders) {
    const conversionRates = {
        "USD": 1,
        "EUR": 1.2,
        "GBP": 1.3
    };

    let totalCostUSD = 0;

    orders.forEach(order => {
        if (!conversionRates.hasOwnProperty(order.currency)) {
            throw new Error(`Unsupported currency: ${order.currency}`);
        }

        let orderTotal = 0;
        order.items.forEach(item => {
            orderTotal += item.quantity * item.pricePerItem;
        });

        // Convert the order total to USD
        const orderTotalUSD = orderTotal * conversionRates[order.currency];
        totalCostUSD += orderTotalUSD;
    });

    // Determine discount rate
    let discount = 0;
    if (totalCostUSD >= 50 && totalCostUSD <= 100) {
        discount = 0.05;
    } else if (totalCostUSD > 100) {
        discount = 0.10;
    }

    // Apply discount
    totalCostUSD = totalCostUSD * (1 - discount);

    return totalCostUSD;
}

const orders1 = [
    { currency: "USD", items: [{ name: "Apple", quantity: 3, pricePerItem: 1 }] },
    { currency: "EUR", items: [{ name: "Orange", quantity: 1, pricePerItem: 3 }] }
];

const orders2 = [
    { currency: "USD", items: [{ name: "Apple", quantity: 3, pricePerItem: 1 }] },
    { currency: "GBP", items: [{ name: "Orange", quantity: 1, pricePerItem: 3 }] }
];

const orders3 = [
    { currency: "USD", items: [{ name: "Apple", quantity: 3, pricePerItem: 1 }] },
    { currency: "RGB", items: [{ name: "Orange", quantity: 1, pricePerItem: 3 }] }
];

try {
    console.log(processOrders(orders1)); // Output should reflect converted and discounted total cost in USD
    console.log(processOrders(orders2)); // Output should reflect converted and discounted total cost in USD
    console.log(processOrders(orders3)); // Output should reflect converted and discounted total cost in USD
} catch (error) {
    console.error(error.message);
}