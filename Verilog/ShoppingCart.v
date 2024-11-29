module CartItem(
    input [15:0] price,         // Item price (16-bit)
    input [7:0] quantity,       // Item quantity (8-bit)
    output [31:0] total_price   // Total price for the item (32-bit)
);
    // Calculate the total price for the item: price * quantity
    assign total_price = price * quantity;
endmodule

module Cart (
    input clk,                   // Clock signal
    input reset,                 // Reset signal
    input [15:0] item_price,     // Price of the item
    input [7:0] item_quantity,   // Quantity of the item
    input add_item,              // Add item signal
    input remove_item,           // Remove item signal
    input update_quantity,       // Update quantity signal
    input [7:0] new_quantity,    // New quantity for the item
    output [31:0] total_price    // Total price of all items in the cart
);
    // Internal register to store total price
    reg [31:0] cart_total;

    // Storage for items (assuming a maximum of 5 items)
    reg [15:0] item_prices[4:0];       // Prices of the items
    reg [7:0] item_quantities[4:0];    // Quantities of the items

    integer i;

    // Add item function
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Reset cart on reset signal
            cart_total <= 0;
            for (i = 0; i < 5; i = i + 1) begin
                item_prices[i] <= 0;
                item_quantities[i] <= 0;
            end
        end else begin
             if (add_item) begin
                for (i = 0; i < 5; i = i + 1) begin
                    if (item_prices[i] == 0) begin
                        item_prices[i] <= item_price;
                        item_quantities[i] <= item_quantity;
                        i = 5; // Exit loop once added
                    end
                end
            end
            if (remove_item) begin
                // Find and remove item (simplified approach)
                for (i = 0; i < 5; i = i + 1) begin
                    if (item_prices[i] == item_price) begin
                        item_prices[i] <= 0;
                        item_quantities[i] <= 0;
                        i = 5; // Exit loop once the item is removed
                    end
                end
            end
            if (update_quantity) begin
                // Update the quantity of an item
                for (i = 0; i < 5; i = i + 1) begin
                    if (item_prices[i] == item_price) begin
                        item_quantities[i] <= new_quantity;
                        i = 5; // Exit loop once the quantity is updated
                    end
                end
            end
        end
    end

    // Calculate the total price of all items
    always @(*) begin
        cart_total = 0;
        for (i = 0; i < 5; i = i + 1) begin
            cart_total = cart_total + (item_prices[i] * item_quantities[i]);
        end
    end

    assign total_price = cart_total;

endmodule

module testbench;
    reg clk;
    reg reset;
    reg [15:0] item_price;
    reg [7:0] item_quantity;
    reg add_item;
    reg remove_item;
    reg update_quantity;
    reg [7:0] new_quantity;
    wire [31:0] total_price;

    // Instantiate the Cart module
    Cart cart(
        .clk(clk),
        .reset(reset),
        .item_price(item_price),
        .item_quantity(item_quantity),
        .add_item(add_item),
        .remove_item(remove_item),
        .update_quantity(update_quantity),
        .new_quantity(new_quantity),
        .total_price(total_price)
    );

    // Clock generation
    always begin
        #5 clk = ~clk;  // Clock period of 10 time units
    end

    initial begin
        // Initialize signals
        clk = 0;
        reset = 0;
        add_item = 0;
        remove_item = 0;
        update_quantity = 0;
        item_price = 16'd100;   // Price of an item
        item_quantity = 8'd1;   // Quantity of the item
        new_quantity = 8'd2;    // New quantity for update

        // Apply reset
        reset = 1;
        #10;
        reset = 0;

        // Add item to cart
        add_item = 1;
        #10;
        add_item = 0;

        // Add another item to cart
        item_price = 16'd50;  // Change item price
        item_quantity = 8'd3; // Change item quantity
        add_item = 1;
        #10;
        add_item = 0;

        // Update quantity of the first item
        update_quantity = 1;
        #10;
        update_quantity = 0;

        // Remove item from cart
        remove_item = 1;
        #10;
        remove_item = 0;

        // Final total price
        #10;
        $display("Total price of items in the cart: $%d", total_price);
    end
endmodule