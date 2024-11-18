module leap_year_calculator_asm(
    input wire clk,              
    input wire reset,            
    input wire [15:0] year,      
    input wire [7:0] num1,       
    input wire [7:0] num2,       
    input wire [1:0] operation,  
    output reg leap_year,        
    output reg [15:0] result     
);
    
    // State definitions
    localparam IDLE = 4'd0;
    localparam CHECK_DIV_4 = 4'd1;
    localparam CHECK_DIV_100 = 4'd2;
    localparam CHECK_DIV_400 = 4'd3;
    localparam INIT_CALC = 4'd4;
    localparam PERFORM_OP = 4'd5;
    localparam DONE = 4'd6;

    // Registers for state machine
    reg [3:0] current_state;
    reg [15:0] tmp_year;
    reg [7:0] tmp_num1, tmp_num2;
    
    // Divider results for leap year calculation
    reg [15:0] div4_result;
    reg [15:0] div100_result;
    reg [15:0] div400_result;
    reg div4_remainder;
    reg div100_remainder;
    reg div400_remainder;

    // Sequential logic for state machine and calculations
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            current_state <= IDLE;
            tmp_year <= 16'd0;
            tmp_num1 <= 8'd0;
            tmp_num2 <= 8'd0;
            leap_year <= 1'b0;
            result <= 16'd0;
            div4_result <= 16'd0;
            div100_result <= 16'd0;
            div400_result <= 16'd0;
            div4_remainder <= 1'b0;
            div100_remainder <= 1'b0;
            div400_remainder <= 1'b0;
        end else begin
            case(current_state)
                IDLE: begin
                    tmp_year <= year;
                    tmp_num1 <= num1;
                    tmp_num2 <= num2;
                    // Calculate division results for leap year
                    div4_result <= year / 4;
                    div4_remainder <= (year % 4) != 0;
                    div100_result <= year / 100;
                    div100_remainder <= (year % 100) != 0;
                    div400_result <= year / 400;
                    div400_remainder <= (year % 400) != 0;
                    current_state <= CHECK_DIV_4;
                end

                CHECK_DIV_4: begin
                    if (!div4_remainder) begin  // Divisible by 4
                        current_state <= CHECK_DIV_100;
                    end else begin
                        leap_year <= 1'b0;
                        current_state <= INIT_CALC;
                    end
                end

                CHECK_DIV_100: begin
                    if (!div100_remainder) begin  // Divisible by 100
                        current_state <= CHECK_DIV_400;
                    end else begin
                        leap_year <= 1'b1;
                        current_state <= INIT_CALC;
                    end
                end

                CHECK_DIV_400: begin
                    if (!div400_remainder) begin  // Divisible by 400
                        leap_year <= 1'b1;
                    end else begin
                        leap_year <= 1'b0;
                    end
                    current_state <= INIT_CALC;
                end
                
                INIT_CALC: begin
                    current_state <= PERFORM_OP;
                end

                PERFORM_OP: begin
                    case (operation)
                        2'b00: result <= {8'b0, tmp_num1} + {8'b0, tmp_num2};    // Addition
                        2'b01: result <= (tmp_num1 >= tmp_num2) ? 
                                       (tmp_num1 - tmp_num2) : 
                                       16'hFFFF;                                  // Subtraction
                        2'b10: result <= tmp_num1 * tmp_num2;                    // Multiplication
                        2'b11: begin                                             // Division
                            if (tmp_num2 != 0) begin
                                result <= tmp_num1 / tmp_num2;
                            end else begin
                                result <= 16'hFFFF;  // Division by zero error
                            end
                        end
                    endcase
                    current_state <= DONE;
                end

                DONE: begin
                    // Stay in DONE state until reset
                    // Values are held until reset
                end

                default: current_state <= IDLE;
            endcase
        end
    end

endmodule
module leap_year_calculator_tb();
    // Test bench signals
    reg clk;
    reg reset;
    reg [15:0] year;
    reg [7:0] num1;
    reg [7:0] num2;
    reg [1:0] operation;
    wire leap_year;
    wire [15:0] result;

    // Instantiate the module under test
    leap_year_calculator_asm uut (
        .clk(clk),
        .reset(reset),
        .year(year),
        .num1(num1),
        .num2(num2),
        .operation(operation),
        .leap_year(leap_year),
        .result(result)
    );

    // Clock generation - 10ns period
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test vectors and checking
    integer errors = 0;
    
    // Task for checking leap year results
    task check_leap_year;
        input [15:0] test_year;
        input expected_leap;
        begin
            @(negedge clk);
            year = test_year;
            reset = 1;
            #10 reset = 0;
            
            // Wait for several clock cycles for calculation to complete
            repeat(10) @(posedge clk);
            
            if (leap_year !== expected_leap) begin
                $display("ERROR: Year %d - Expected leap_year = %b, Got = %b", 
                    test_year, expected_leap, leap_year);
                errors = errors + 1;
            end else begin
                $display("PASS: Year %d correctly identified as %s", 
                    test_year, expected_leap ? "leap year" : "non-leap year");
            end
            #20; // Additional wait between tests
        end
    endtask

    // Task for checking arithmetic operations
    task check_arithmetic;
        input [7:0] a;
        input [7:0] b;
        input [1:0] op;
        input [15:0] expected_result;
        begin
            @(negedge clk);
            num1 = a;
            num2 = b;
            operation = op;
            reset = 1;
            #10 reset = 0;
            
            // Wait for several clock cycles for calculation to complete
            repeat(10) @(posedge clk);
            
            if (result !== expected_result) begin
                $display("ERROR: %d %s %d - Expected result = %d, Got = %d",
                    a, 
                    op == 2'b00 ? "+" :
                    op == 2'b01 ? "-" :
                    op == 2'b10 ? "*" : "/",
                    b, expected_result, result);
                errors = errors + 1;
            end else begin
                $display("PASS: Arithmetic operation %d %s %d = %d",
                    a,
                    op == 2'b00 ? "+" :
                    op == 2'b01 ? "-" :
                    op == 2'b10 ? "*" : "/",
                    b, result);
            end
            #20; // Additional wait between tests
        end
    endtask

    // Main test procedure
    initial begin
        // Initialize signals
        reset = 1;
        year = 0;
        num1 = 0;
        num2 = 0;
        operation = 0;
        
        // Wait 100 ns for global reset
        #100;
        reset = 0;
        
        $display("\nStarting Leap Year Tests...");
        check_leap_year(2000, 1);  // Divisible by 400 - leap year
        check_leap_year(2100, 0);  // Divisible by 100 but not 400 - not leap year
        check_leap_year(2004, 1);  // Divisible by 4 but not 100 - leap year
        check_leap_year(2001, 0);  // Not divisible by 4 - not leap year
        check_leap_year(2400, 1);  // Divisible by 400 - leap year
        check_leap_year(1900, 0);  // Divisible by 100 but not 400 - not leap year
        check_leap_year(2024, 1);  // Divisible by 4 but not 100 - leap year
        
        $display("\nStarting Arithmetic Tests...");
        // Addition tests
        check_arithmetic(10, 20, 2'b00, 30);    // Basic addition
        check_arithmetic(255, 1, 2'b00, 256);   // Addition with overflow
        check_arithmetic(0, 0, 2'b00, 0);       // Addition with zeros
        
        // Subtraction tests
        check_arithmetic(20, 10, 2'b01, 10);    // Basic subtraction
        check_arithmetic(0, 1, 2'b01, 16'hFFFF);// Subtraction with underflow
        check_arithmetic(50, 50, 2'b01, 0);     // Subtraction to zero
        
        // Multiplication tests
        check_arithmetic(10, 20, 2'b10, 200);   // Basic multiplication
        check_arithmetic(0, 50, 2'b10, 0);      // Multiplication with zero
        check_arithmetic(255, 2, 2'b10, 510);   // Multiplication near max
        
        // Division tests
        check_arithmetic(20, 5, 2'b11, 4);      // Basic division
        check_arithmetic(0, 5, 2'b11, 0);       // Division of zero
        check_arithmetic(50, 0, 2'b11, 16'hFFFF); // Division by zero
        check_arithmetic(255, 2, 2'b11, 127);   // Division with max input
        
        // Report results
        #100;
        $display("\nTest Summary:");
        $display("Total errors: %d", errors);
        if (errors == 0)
            $display("All tests passed successfully!");
        else
            $display("Some tests failed. Please check the error messages above.");
        
        #100;
        $finish;
    end

endmodule