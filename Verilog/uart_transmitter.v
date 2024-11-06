module uart_transmitter (

    input clk,

    input reset,

    input [7:0] data_in,

    input tx_start,

    output reg tx_busy,

    output reg tx_data

);

    parameter BAUD_RATE = 19200;

    parameter CLK_FREQ = 50000000;

    parameter CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    reg [15:0] clk_count;

    reg [3:0] bit_count;

    reg [8:0] tx_data_reg;

    always @(posedge clk or posedge reset) begin

        if (reset) begin

            clk_count <= 0;

            bit_count <= 0;

            tx_busy <= 0;

            tx_data <= 1; // Idle state (high)

        end

        else begin

            if (tx_start) begin

                tx_busy <= 1;

                tx_data_reg <= {data_in, 1'b1}; // Data bits, stop bit (1)

                tx_data <= 1'b0; // Start bit (0)

                clk_count <= 0;

                bit_count <= 0;

            end

            else if (tx_busy) begin

                if (clk_count == CLKS_PER_BIT - 1) begin

                    clk_count <= 0;

                    if (bit_count == 9) begin

                        tx_busy <= 0;

                        tx_data <= 1; // Idle state (high)

                    end

                    else begin

                        tx_data <= tx_data_reg[0];

                        tx_data_reg <= {1'b1, tx_data_reg[8:1]}; // Shift right by 1 bit

                        bit_count <= bit_count + 1;

                    end

                end

                else begin

                    clk_count <= clk_count + 1;

                end

            end

        end

    end

endmodule
`timescale 1ns/1ps

module uart_transmitter_tb;

    // Parameters
    parameter CLK_FREQ = 50000000;
    parameter BAUD_RATE = 19200;
    parameter CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    parameter CLK_PERIOD = 20; // 50MHz clock -> 20ns period

    // Test bench signals
    reg clk;
    reg reset;
    reg [7:0] data_in;
    reg tx_start;
    wire tx_busy;
    wire tx_data;

    // Instantiate UART transmitter
    uart_transmitter #(
        .BAUD_RATE(BAUD_RATE),
        .CLK_FREQ(CLK_FREQ)
    ) uut (
        .clk(clk),
        .reset(reset),
        .data_in(data_in),
        .tx_start(tx_start),
        .tx_busy(tx_busy),
        .tx_data(tx_data)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test cases
    initial begin
        // Initialize waveform dump
        $dumpfile("uart_tx_test.vcd");
        $dumpvars(0, uart_transmitter_tb);

        // Initialize signals
        reset = 0;
        data_in = 8'h00;
        tx_start = 0;

        // Test Case 1: Reset
        $display("Test Case 1: Reset");
        #100 reset = 1;
        #100 reset = 0;
        
        // Verify reset conditions
        if (tx_data !== 1'b1) $error("Reset: tx_data should be 1");
        if (tx_busy !== 1'b0) $error("Reset: tx_busy should be 0");

        // Test Case 2: Single Byte Transmission (0x55)
        $display("Test Case 2: Single Byte Transmission (0x55)");
        #100;
        data_in = 8'h55;  // Alternating 1s and 0s
        tx_start = 1;
        #(CLK_PERIOD);
        tx_start = 0;
        
        // Wait for transmission to complete
        wait(!tx_busy);
        #(CLK_PERIOD * 10);

        // Test Case 3: Back-to-back Transmission
        $display("Test Case 3: Back-to-back Transmission");
        data_in = 8'hAA;
        tx_start = 1;
        #(CLK_PERIOD);
        tx_start = 0;
        
        // Try to start another transmission while busy
        #(CLK_PERIOD * 10);
        data_in = 8'h33;
        tx_start = 1;
        #(CLK_PERIOD);
        tx_start = 0;

        // Wait for transmission to complete
        wait(!tx_busy);
        #(CLK_PERIOD * 10);

        // Test Case 4: Maximum Value
        $display("Test Case 4: Maximum Value");
        data_in = 8'hFF;
        tx_start = 1;
        #(CLK_PERIOD);
        tx_start = 0;
        
        // Wait for transmission to complete
        wait(!tx_busy);
        #(CLK_PERIOD * 10);

        // Test Case 5: Minimum Value
        $display("Test Case 5: Minimum Value");
        data_in = 8'h00;
        tx_start = 1;
        #(CLK_PERIOD);
        tx_start = 0;
        
        // Wait for transmission to complete
        wait(!tx_busy);
        #(CLK_PERIOD * 10);

        // Test completion
        $display("All test cases completed");
        #1000 $finish;
    end

    // Monitor for transmission timing
    real start_time;
    real end_time;
    real bit_time;
    
    always @(negedge tx_data) begin
        if (tx_busy) begin
            start_time = $realtime;
            $display("Transmission started at %t", start_time);
        end
    end

    always @(posedge tx_data) begin
        if (!tx_busy) begin
            end_time = $realtime;
            bit_time = (end_time - start_time) / 10; // 10 bits total (start + 8 data + stop)
            $display("Transmission ended at %t", end_time);
            $display("Average bit time: %t ns", bit_time);
        end
    end

    // Task to verify correct baud rate timing
    task check_baud_rate;
        real expected_bit_time;
        begin
            expected_bit_time = 1000000000.0 / BAUD_RATE; // Convert to ns
            if (bit_time < expected_bit_time * 0.95 || bit_time > expected_bit_time * 1.05)
                $error("Baud rate error: Expected %f ns/bit, got %f ns/bit", expected_bit_time, bit_time);
        end
    endtask

endmodule