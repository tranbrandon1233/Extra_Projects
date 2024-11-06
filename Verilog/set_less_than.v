`timescale 1ns/1ps

module slt (output result, input [31:0] A, input [31:0] B);
    wire [31:0] AddSubResult;
    wire V, C32, x_res, notB;

    // Assuming adder is a module that performs A + B + Cin
    adder add1 (AddSubResult, C32, A, ~B, 1'b1); // ~B + 1 is two's complement of B

    // Assuming detect_overflow is a module that detects overflow
    detect_overflow vf1 (V, A[31], ~B[31], AddSubResult[31]);

    // XOR to determine the result based on overflow
    xor xor1 (x_res, V, AddSubResult[31]);

    // Result is 1 if A < B, otherwise 0
    assign result = x_res;
endmodule

// 32-bit adder module
module adder(
    output [31:0] sum,
    output cout,
    input [31:0] a,
    input [31:0] b,
    input cin
);
    assign {cout, sum} = a + b + cin;
endmodule

// Overflow detection module
module detect_overflow(
    output overflow,
    input a_sign,
    input b_sign,
    input sum_sign
);
    // Overflow occurs when:
    // 1. Adding two positives gives a negative (pos + pos = neg)
    // 2. Adding two negatives gives a positive (neg + neg = pos)
    assign overflow = (a_sign == b_sign) && (a_sign != sum_sign);
endmodule


// Testbench
module slt_tb();
    // Signals
    reg [31:0] A, B;
    wire result;
    integer errors;
    reg expected;
    
    // Instantiate the SLT module
    slt uut (
        .result(result),
        .A(A),
        .B(B)
    );
    
    // Helper function to check if A is less than B (signed comparison)
    function expected_result;
        input [31:0] a, b;
        begin
            expected_result = ($signed(a) < $signed(b));
        end
    endfunction
    
    initial begin
        errors = 0;
        
        // Test Case Category 1: Basic Positive Numbers
        // ------------------------------------------
        $display("Test Category 1: Basic Positive Numbers");
        
        // TC1.1: Simple positive numbers
        A = 32'd5; B = 32'd10;
        #10;
        expected = expected_result(A, B);
        check_result("TC1.1");
        
        // TC1.2: Equal positive numbers
        A = 32'd15; B = 32'd15;
        #10;
        expected = expected_result(A, B);
        check_result("TC1.2");
        
        // TC1.3: A > B with positive numbers
        A = 32'd100; B = 32'd50;
        #10;
        expected = expected_result(A, B);
        check_result("TC1.3");
        
        // Test Case Category 2: Basic Negative Numbers
        // ------------------------------------------
        $display("\nTest Category 2: Basic Negative Numbers");
        
        // TC2.1: Simple negative numbers
        A = -32'd5; B = -32'd3;
        #10;
        expected = expected_result(A, B);
        check_result("TC2.1");
        
        // TC2.2: Equal negative numbers
        A = -32'd15; B = -32'd15;
        #10;
        expected = expected_result(A, B);
        check_result("TC2.2");
        
        // TC2.3: Larger negative numbers
        A = -32'd1000; B = -32'd500;
        #10;
        expected = expected_result(A, B);
        check_result("TC2.3");
        
        // Test Case Category 3: Mixed Signs
        // --------------------------------
        $display("\nTest Category 3: Mixed Signs");
        
        // TC3.1: Negative vs Positive
        A = -32'd5; B = 32'd5;
        #10;
        expected = expected_result(A, B);
        check_result("TC3.1");
        
        // TC3.2: Positive vs Negative
        A = 32'd5; B = -32'd5;
        #10;
        expected = expected_result(A, B);
        check_result("TC3.2");
        
        // TC3.3: Zero vs Negative
        A = 32'd0; B = -32'd5;
        #10;
        expected = expected_result(A, B);
        check_result("TC3.3");
        
        // TC3.4: Negative vs Zero
        A = -32'd5; B = 32'd0;
        #10;
        expected = expected_result(A, B);
        check_result("TC3.4");
        
        // Test Case Category 4: Edge Cases
        // -------------------------------
        $display("\nTest Category 4: Edge Cases");
        
        // TC4.1: Maximum positive vs Maximum positive-1
        A = 32'h7FFFFFFF; B = 32'h7FFFFFFE;
        #10;
        expected = expected_result(A, B);
        check_result("TC4.1");
        
        // TC4.2: Minimum negative vs Maximum positive
        A = 32'h80000000; B = 32'h7FFFFFFF;
        #10;
        expected = expected_result(A, B);
        check_result("TC4.2");
        
        // TC4.3: Maximum positive vs Minimum negative
        A = 32'h7FFFFFFF; B = 32'h80000000;
        #10;
        expected = expected_result(A, B);
        check_result("TC4.3");
        
        // TC4.4: Minimum negative vs Minimum negative+1
        A = 32'h80000000; B = 32'h80000001;
        #10;
        expected = expected_result(A, B);
        check_result("TC4.4");
        
        // Test Case Category 5: Corner Cases with Overflow
        // ---------------------------------------------
        $display("\nTest Category 5: Corner Cases with Overflow");
        
        // TC5.1: Numbers that cause overflow in subtraction
        A = 32'h7FFFFFFF; B = 32'h80000000;
        #10;
        expected = expected_result(A, B);
        check_result("TC5.1");
        
        // TC5.2: Another overflow case
        A = 32'h80000000; B = 32'h7FFFFFFF;
        #10;
        expected = expected_result(A, B);
        check_result("TC5.2");
        
        // Report results
        $display("\nTest Summary:");
        $display("Total Errors: %0d", errors);
        
        if (errors == 0)
            $display("All tests PASSED!");
        else
            $display("Some tests FAILED!");
            
        $finish;
    end
    
    // Task to check result and report
    task check_result;
        input [8*8-1:0] test_name;
        begin
            if (result !== expected) begin
                $display("Error in %s: A=%0d, B=%0d, Expected=%0b, Got=%0b", 
                        test_name, $signed(A), $signed(B), expected, result);
                errors = errors + 1;
            end else begin
                $display("PASS: %s: A=%0d, B=%0d, Result=%0b", 
                        test_name, $signed(A), $signed(B), result);
            end
        end
    endtask

endmodule