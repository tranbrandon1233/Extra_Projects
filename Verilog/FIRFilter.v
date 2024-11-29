`timescale 1ns/1ps

module fir_filter (
    input clk,
    input reset,
    input signed [15:0] sample_in,
    output reg signed [31:0] filter_out
);
    parameter N = 4;
    reg signed [15:0] coeffs [0:N-1];
    reg signed [15:0] shift_reg [0:N-1];
    integer i;

    initial begin
        coeffs[0] = 16'sd1;
        coeffs[1] = 16'sd2;
        coeffs[2] = 16'sd3;
        coeffs[3] = 16'sd4;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            filter_out <= 32'sd0;
            for (i = 0; i < N; i = i + 1) begin
                shift_reg[i] <= 16'sd0;
            end
        end else begin
            shift_reg[0] <= sample_in;
            for (i = 1; i < N; i = i + 1) begin
                shift_reg[i] <= shift_reg[i-1];
            end
            filter_out <= 32'sd0;
            for (i = 0; i < N; i = i + 1) begin
                filter_out <= filter_out + shift_reg[i] * coeffs[i];
            end
        end
    end
endmodule

module fir_filter_tb();
    // Test bench signals
    reg clk;
    reg reset;
    reg signed [15:0] sample_in;
    wire signed [31:0] filter_out;
    
    // Instantiate the FIR filter
    fir_filter uut (
        .clk(clk),
        .reset(reset),
        .sample_in(sample_in),
        .filter_out(filter_out)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz clock
    end
    
    // Test stimulus
    initial begin
        // Initialize waveform dumping
        $dumpfile("fir_filter_tb.vcd");
        $dumpvars(0, fir_filter_tb);
        
        // Initialize signals
        reset = 0;
        sample_in = 0;
        
        // Test case 1: Reset behavior
        #10 reset = 1;
        #20 reset = 0;
        
        // Verify reset state
        if (filter_out !== 32'sd0) begin
            $display("Error: Reset failed! Expected 0, got %d", filter_out);
        end
        
        // Test case 2: Single impulse response
        // Should see the coefficients appear in sequence
        #10 sample_in = 16'sd1000;
        #10 sample_in = 16'sd0;
        
        // Wait for all coefficients to propagate
        #40;
        
        // Test case 3: Step response
        sample_in = 16'sd500;
        #50;  // Wait for steady state
        
        // Test case 4: Negative values
        sample_in = -16'sd750;
        #50;
        
        // Test case 5: Maximum value test
        sample_in = 16'sh7FFF;  // Maximum positive 16-bit value
        #50;
        
        // Test case 6: Minimum value test
        sample_in = 16'sh8000;  // Minimum negative 16-bit value
        #50;
        
        // Test case 7: Quick alternating values
        sample_in = 16'sd1000;
        #10 sample_in = -16'sd1000;
        #10 sample_in = 16'sd1000;
        #10 sample_in = -16'sd1000;
        #40;
        
        // End simulation
        #100 $display("Simulation completed");
        $finish;
    end
    
    // Monitor changes
    initial begin
        $monitor("Time=%0t reset=%b sample_in=%d filter_out=%d",
                 $time, reset, sample_in, filter_out);
    end
    
    // Additional checking logic
    reg [31:0] expected_output;
    reg [3:0] test_case;
    
    always @(posedge clk) begin
        if (!reset) begin
            // Verify impulse response for first test case
            if (test_case == 1) begin
                case ($time)
                    // Check outputs at specific times after impulse
                    70: expected_output = 32'sd1000;    // First coefficient
                    80: expected_output = 32'sd2000;    // Second coefficient
                    90: expected_output = 32'sd3000;    // Third coefficient
                    100: expected_output = 32'sd4000;   // Fourth coefficient
                    default: expected_output = 32'dx;   // Don't care
                endcase
                
                if (expected_output !== 32'dx && filter_out !== expected_output) begin
                    $display("Error at time %0t: Expected %d, got %d",
                            $time, expected_output, filter_out);
                end
            end
        end
    end
endmodule