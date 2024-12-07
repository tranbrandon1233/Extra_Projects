module divider(
    input wire clk,
    input wire reset,
    input wire start,
    input wire [31:0] dividend,
    input wire [31:0] divisor,
    output reg [31:0] quotient,
    output reg [31:0] remainder,
    output reg done
);

    reg [31:0] dividend_copy;
    reg [31:0] divisor_copy;
    reg [5:0] count;
    
    // States
    parameter IDLE = 2'b00;
    parameter DIVIDING = 2'b01;
    parameter DONE = 2'b10;
    
    reg [1:0] state;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            quotient <= 32'd0;
            remainder <= 32'd0;
            dividend_copy <= 32'd0;
            divisor_copy <= 32'd0;
            count <= 6'd0;
            done <= 1'b0;
            state <= IDLE;
        end
        else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        if (divisor == 0) begin  // Check for division by zero
                            state <= DONE;
                            done <= 1'b1;
                            quotient <= 32'hFFFFFFFF;  // Indicate error
                            remainder <= 32'hFFFFFFFF; // Indicate error
                        end
                        else begin
                            dividend_copy <= dividend;
                            divisor_copy <= divisor;
                            quotient <= 32'd0;
                            remainder <= 32'd0;
                            count <= 6'd0;
                            done <= 1'b0;
                            state <= DIVIDING;
                        end
                    end
                end
                
                DIVIDING: begin
                    if (dividend_copy >= divisor_copy) begin
                        dividend_copy <= dividend_copy - divisor_copy;
                        quotient <= quotient + 1;
                    end
                    else begin
                        remainder <= dividend_copy;
                        state <= DONE;
                        done <= 1'b1;
                    end
                end
                
                DONE: begin
                    if (!start) begin
                        state <= IDLE;
                        done <= 1'b0;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule


module divider_tb;
    reg clk;
    reg reset;
    reg start;
    reg [31:0] dividend;
    reg [31:0] divisor;
    wire [31:0] quotient;
    wire [31:0] remainder;
    wire done;
    
    // Instance of divider module
    divider dut(
        .clk(clk),
        .reset(reset),
        .start(start),
        .dividend(dividend),
        .divisor(divisor),
        .quotient(quotient),
        .remainder(remainder),
        .done(done)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        // Initialize signals
        reset = 1;
        start = 0;
        dividend = 0;
        divisor = 0;
        #10;
        
        reset = 0;
        #10;
        
        // Test Case 1: 20 ÷ 4
        dividend = 20;
        divisor = 4;
        start = 1;
        $display("Test Case 1: %d ÷ %d", dividend, divisor);
        wait(done);
        #10;
        $display("Quotient = %d, Remainder = %d\n", quotient, remainder);
        start = 0;
        #20;
        
        // Test Case 2: 17 ÷ 3
        dividend = 17;
        divisor = 3;
        start = 1;
        $display("Test Case 2: %d ÷ %d", dividend, divisor);
        wait(done);
        #10;
        $display("Quotient = %d, Remainder = %d\n", quotient, remainder);
        start = 0;
        #20;
        
        // Test Case 3: Division by zero
        dividend = 25;
        divisor = 0;
        start = 1;
        $display("Test Case 3: %d ÷ %d (Division by zero)", dividend, divisor);
        wait(done);
        #10;
        $display("Quotient = %h, Remainder = %h\n", quotient, remainder);
        start = 0;
        #20;
        
        // Test Case 4: 7 ÷ 8 (Dividend smaller than divisor)
        dividend = 7;
        divisor = 8;
        start = 1;
        $display("Test Case 4: %d ÷ %d", dividend, divisor);
        wait(done);
        #10;
        $display("Quotient = %d, Remainder = %d\n", quotient, remainder);
        start = 0;
        #20;
        
        // Test Case 5: 100 ÷ 10
        dividend = 100;
        divisor = 10;
        start = 1;
        $display("Test Case 5: %d ÷ %d", dividend, divisor);
        wait(done);
        #10;
        $display("Quotient = %d, Remainder = %d\n", quotient, remainder);
        start = 0;
        #20;
        
        $finish;
    end
endmodule