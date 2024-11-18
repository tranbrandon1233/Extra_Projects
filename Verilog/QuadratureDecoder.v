module quadrature_decoder (
    input wire clk,       // System clock
    input wire reset,     // Reset signal
    input wire A,         // Encoder signal A
    input wire B,         // Encoder signal B
    output reg signed [15:0] position,  // Output position count
    output reg direction               // Output direction (1 for CW, 0 for CCW)
);

    reg A_delayed, B_delayed; // Delayed signals for edge detection

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            position <= 16'd0;
            direction <= 1'b0;
            A_delayed <= 1'b0;
            B_delayed <= 1'b0;
        end else begin
            // Delay the signals for edge detection
            A_delayed <= A;
            B_delayed <= B;

            // Detect rising edge of signal A
            if (A && !A_delayed) begin
                if (B) begin
                    // Clockwise (CW) rotation
                    direction <= 1'b1;
                    position <= position + 1;
                end else begin
                    // Counter-clockwise (CCW) rotation
                    direction <= 1'b0;
                    position <= position - 1;
                end
            end
            // Detect rising edge of signal B
            else if (B && !B_delayed) begin
                if (A) begin
                    // Counter-clockwise (CCW) rotation
                    direction <= 1'b0;
                    position <= position - 1;
                end else begin
                    // Clockwise (CW) rotation
                    direction <= 1'b1;
                    position <= position + 1;
                end
            end
        end
    end

endmodule

`timescale 1ns/1ps

module quadrature_decoder_tb;

    // Testbench signals
    reg clk;
    reg reset;
    reg A;
    reg B;
    wire signed [15:0] position;
    wire direction;

    // Instantiate the quadrature decoder
    quadrature_decoder uut (
        .clk(clk),
        .reset(reset),
        .A(A),
        .B(B),
        .position(position),
        .direction(direction)
    );

    // Clock generation
    always begin
        clk = 0;
        #5;
        clk = 1;
        #5;
    end

    // Test stimulus
    initial begin
        // Initialize waveform dump
        $dumpfile("quadrature_decoder_tb.vcd");
        $dumpvars(0, quadrature_decoder_tb);

        // Initialize signals
        A = 0;
        B = 0;
        reset = 1;

        // Wait for 2 clock cycles and release reset
        #20;
        reset = 0;
        #10;

        // Test Case 1: Clockwise rotation (A leads B)
        // A: 0->1, B: 0
        A = 1;
        #10;
        B = 1;
        #10;
        A = 0;
        #10;
        B = 0;
        #10;

        // Test Case 2: Counter-clockwise rotation (B leads A)
        B = 1;
        #10;
        A = 1;
        #10;
        B = 0;
        #10;
        A = 0;
        #10;

        // Test Case 3: Multiple clockwise rotations
        repeat(3) begin
            A = 1;
            #10;
            B = 1;
            #10;
            A = 0;
            #10;
            B = 0;
            #10;
        end

        // Test Case 4: Test reset during operation
        A = 1;
        #10;
        reset = 1;
        #20;
        reset = 0;
        #10;

        // Test Case 5: Rapid changes
        repeat(5) begin
            // Fast clockwise rotation
            A = 1; B = 0;
            #5;
            B = 1;
            #5;
            A = 0;
            #5;
            B = 0;
            #5;
        end

        // Test Case 6: Invalid transition test
        A = 1;
        B = 1;
        #10;
        A = 0;
        B = 0;
        #10;

        // End simulation
        #100;
        $display("Simulation completed");
        $finish;
    end

    // Monitor changes
    always @(position or direction) begin
        $display("Time=%0t Position=%d Direction=%b", $time, position, direction);
    end

    // Verification
    reg [15:0] expected_position;
    reg expected_direction;
    
    initial begin
        expected_position = 0;
        expected_direction = 0;
        
        // Wait for reset to complete
        #30;
        
        // Monitor for errors
        forever @(position) begin
            if (reset) begin
                expected_position = 0;
                if (position !== 0) begin
                    $display("Error at time %0t: Reset failed. Position should be 0 but is %d", 
                            $time, position);
                end
            end
        end
    end

endmodule