`timescale 1ns / 1ps

module AdvancedPriorityEncoderWithInterrupt (
    input [15:0] request,
    input [15:0] mask,
    input [1:0] priority_mode,  // Priority modes: 00 = MSB, 01 = LSB, 10 = Round-Robin, 11 = Reverse
    input clk,
    input reset_n,             // Active-low asynchronous reset
    output reg [3:0] priority,
    output reg interrupt
);

    reg [3:0] last_priority;    // To store last priority for Round-Robin mode
    reg interrupt_pending;      // Internal register to simulate delayed interrupt
    integer i;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            priority <= 4'b0000;
            interrupt <= 1'b0;
            last_priority <= 4'b0000;
            interrupt_pending <= 1'b0;
        end else begin
            interrupt <= interrupt_pending;
        end
    end

    always @(*) begin
        priority = 4'b0000;
        interrupt_pending = 1'b0;

        case (priority_mode)
            2'b00: begin // MSB Priority
                for (i = 15; i >= 0; i = i - 1) begin
                    if (request[i] && mask[i]) begin
                        priority = i;
                        interrupt_pending = 1'b1;
                        i = -1;
                    end
                end
            end
            2'b01: begin // LSB Priority
                for (i = 0; i < 16; i = i + 1) begin
                    if (request[i] && mask[i]) begin
                        priority = i;
                        interrupt_pending = 1'b1;
                        i = 16;
                    end
                end
            end
            2'b10: begin // Round-Robin Priority
                for (i = last_priority + 1; i < last_priority + 17; i = i + 1) begin
                    if (request[i % 16] && mask[i % 16]) begin
                        priority = i % 16;
                        interrupt_pending = 1'b1;
                        i = last_priority + 17;
                    end
                end
                if (interrupt_pending) begin
                    last_priority = priority;
                end
            end
            2'b11: begin // Reverse Priority
                for (i = 0; i < 16; i = i + 1) begin
                    if (request[15 - i] && mask[15 - i]) begin
                        priority = 15 - i;
                        interrupt_pending = 1'b1;
                        i=16;
                    end
                end
            end
            default: begin
                priority = 4'b0000;
                interrupt_pending = 1'b0;
            end
        endcase
    end
endmodule

module tb_AdvancedPriorityEncoderWithInterrupt;

    // Inputs
    reg [15:0] request;
    reg [15:0] mask;
    reg [1:0] priority_mode;
    reg clk;
    reg reset_n;

    // Outputs
    wire [3:0] priority;
    wire interrupt;

    // Instantiate the module
    AdvancedPriorityEncoderWithInterrupt uut (
        .request(request),
        .mask(mask),
        .priority_mode(priority_mode),
        .clk(clk),
        .reset_n(reset_n),
        .priority(priority),
        .interrupt(interrupt)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    // Test procedure
    initial begin
        // Initialize inputs
        request = 16'b0;
        mask = 16'b0;
        priority_mode = 2'b00;
        reset_n = 1'b0;

        // Apply reset
        #10 reset_n = 1'b1;

        // Test 1: MSB-first priority mode
        priority_mode = 2'b00;
        request = 16'b1000_0000_0000_0001;
        mask = 16'b1111_1111_1111_1111;
        #10;
        $display("MSB-first: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 2: LSB-first priority mode
        priority_mode = 2'b01;
        request = 16'b1000_0000_0000_0001;
        mask = 16'b1111_1111_1111_1111;
        #10;
        $display("LSB-first: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 3: Round-Robin priority mode
        priority_mode = 2'b10;
        request = 16'b0000_0000_0000_1111;
        mask = 16'b1111_1111_1111_1111;
        #10;
        $display("Round-Robin: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 4: Reverse priority mode
        priority_mode = 2'b11;
        request = 16'b1000_0000_0000_0001;
        mask = 16'b1111_1111_1111_1111;
        #10;
        $display("Reverse: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 5: No active requests
        request = 16'b0000_0000_0000_0000;
        mask = 16'b1111_1111_1111_1111;
        #10;
        $display("No active requests: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 6: Fully masked out requests
        request = 16'b1111_1111_1111_1111;
        mask = 16'b0000_0000_0000_0000;
        #10;
        $display("Fully masked out: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 7: Conflicting scenarios
        request = 16'b1010_1010_1010_1010;
        mask = 16'b0101_0101_0101_0101;
        #10;
        $display("Conflicting scenarios: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 8: Round-Robin statefulness
        priority_mode = 2'b10;
        request = 16'b0000_0000_0000_1111;
        mask = 16'b1111_1111_1111_1111;
        #10;
        request = 16'b1111_0000_0000_0000;
        #10;
        $display("Round-Robin statefulness: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 9: Asynchronous reset
        reset_n = 1'b0;
        #10;
        reset_n = 1'b1;
        request = 16'b1000_0000_0000_0001;
        mask = 16'b1111_1111_1111_1111;
        #10;
        $display("After reset: Priority = %b, Interrupt = %b", priority, interrupt);

        // Test 10: Stress test with random values
        repeat (10) begin
            request = $random;
            mask = $random;
            priority_mode = $random % 4;
            #10;
            $display("Random test: Priority = %b, Interrupt = %b", priority, interrupt);
        end

        // Test 11: All bits set high
        request = 16'b1111_1111_1111_1111;
        mask = 16'b1111_1111_1111_1111;
        priority_mode = 2'b00;
        #10;
        $display("All bits high: Priority = %b, Interrupt = %b", priority, interrupt);

        // Finish simulation
        $finish;
    end

endmodule