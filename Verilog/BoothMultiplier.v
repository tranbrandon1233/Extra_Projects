module BoothMultiplier (
    input [7:0] multiplicand,
    input [7:0] multiplier,
    output reg [15:0] product
);
    reg [7:0] A, Q, M;
    reg Q_1;
    reg [3:0] count;
    
    always @(*) begin
        // Initialize registers
        A = 8'b0;
        Q = multiplier;
        Q_1 = 1'b0;
        M = multiplicand;
        count = 4'd8; // 8-bit multiplication
        
        while (count > 0) begin
            case ({Q[0], Q_1})
                2'b10: A = A - M; // Subtract multiplicand
                2'b01: A = A + M; // Add multiplicand
                default: ; // No operation
            endcase
            
            // Arithmetic right shift
            {A, Q, Q_1} = {A[7], A, Q, Q_1} >> 1;
            
            // Decrement the counter
            count = count - 1;
        end
        
        // Combine result
        product = {A, Q};
    end
endmodule

module BoothMultiplier_tb;
    reg [7:0] multiplicand;
    reg [7:0] multiplier;
    wire [15:0] product;

    // Instantiate the BoothMultiplier module
    BoothMultiplier uut (
        .multiplicand(multiplicand),
        .multiplier(multiplier),
        .product(product)
    );

    initial begin
        // Test cases
        multiplicand = 8'b00000101; // 5
        multiplier = 8'b00000011;   // 3
        #10;
        $display("Multiplicand: %d, Multiplier: %d, Product: %d", multiplicand, multiplier, product);

        multiplicand = 8'b00001010; // 10
        multiplier = 8'b00000011;   // 3
        #10;
        $display("Multiplicand: %d, Multiplier: %d, Product: %d", multiplicand, multiplier, product);

        multiplicand = 8'b00011001; // 25
        multiplier = 8'b00100111;   // 39
        #10;
        $display("Multiplicand: %d, Multiplier: %d, Product: %d", multiplicand, multiplier, product);

        multiplicand = 8'b01111011; // 123
        multiplier = 8'b01111101;   // 125
        #10;
        $display("Multiplicand: %d, Multiplier: %d, Product: %d", multiplicand, multiplier, product);

        // Add more test cases as needed
        $finish;
    end
endmodule
