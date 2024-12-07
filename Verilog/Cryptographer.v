module crypto_accelerator (
    input wire clk,
    input wire rst,
    input wire [7:0] data_in,
    input wire [7:0] key,
    input wire encrypt,    // 1 for encrypt, 0 for decrypt
    input wire start,
    output reg [7:0] data_out,
    output reg done
);

    // States
    localparam IDLE = 2'b00;
    localparam PROCESS = 2'b01;
    localparam COMPLETE = 2'b10;

    reg [1:0] current_state, next_state;
    reg [7:0] result;  // Internal register for computation

    // State register
    always @(posedge clk or posedge rst) begin
        if (rst)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end

    // Next state logic
    always @(*) begin
        case (current_state)
            IDLE: 
                next_state = start ? PROCESS : IDLE;
            PROCESS: 
                next_state = COMPLETE;
            COMPLETE: 
                next_state = IDLE;
            default: 
                next_state = IDLE;
        endcase
    end

    // Datapath
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 8'b0;
            result <= 8'b0;   // Reset internal computation
            done <= 1'b0;
        end else if (data_in == 8'b0) begin
            data_out <= 8'b0;
            result <= 8'b0; 
            done <= 1'b1;
        end else begin
            case (current_state)
                IDLE: begin
                    data_out <= 8'b0;  // Reset output
                    result <= 8'b0;   // Reset internal computation
                    done <= 1'b0;
                end
                PROCESS: begin
                    result <= data_in ^ key;  // Perform XOR operation
                end
                COMPLETE: begin
                    data_out <= result;      // Finalize data_out
                    done <= 1'b1;            // Signal completion
                end
            endcase
        end
    end

endmodule

module crypto_accelerator_tb;
    reg clk;
    reg rst;
    reg [7:0] data_in;
    reg [7:0] key;
    reg encrypt;
    reg start;
    wire [7:0] data_out;
    wire done;
    integer i;

    // Instantiate the crypto_accelerator
    crypto_accelerator uut (
        .clk(clk),
        .rst(rst),
        .data_in(data_in),
        .key(key),
        .encrypt(encrypt),
        .start(start),
        .data_out(data_out),
        .done(done)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    // Custom assert task
    task assert(input condition, input [255:0] message);
        begin
            if (!condition) begin
                $display("ASSERTION FAILED: %s", message);
                $stop; // Halt simulation on failure
            end
        end
    endtask

    // Test stimulus
    initial begin
        // Initialize signals
        rst = 1;
        data_in = 8'h00;
        key = 8'h00;
        encrypt = 0;
        start = 0;
        #10;

        // Release reset
        rst = 0;
        #10;

        // Test Case 1: Encryption
        $display("Test Case 1: Encryption");
        data_in = 8'hA5;
        key = 8'h3C;
        encrypt = 1;
        start = 1;
        #10;
        start = 0;
        wait(done);
        assert(data_out == (data_in ^ key), "Encryption failed for Test Case 1");
        $display("Input: %h, Key: %h, Encrypted: %h", data_in, key, data_out);
        #10;

        // Test Case 2: Decryption
        $display("\nTest Case 2: Decryption");
        data_in = data_out;  // Use encrypted data from previous test
        key = 8'h3C;        // Same key
        encrypt = 0;
        start = 1;
        #10;
        start = 0;
        wait(done);
        assert(data_out == 8'hA5, "Decryption failed for Test Case 2");
        $display("Input: %h, Key: %h, Decrypted: %h", data_in, key, data_out);
        #10;

        // Test Case 3: Different key
        $display("\nTest Case 3: Different key");
        data_in = 8'hFF;
        key = 8'h55;
        encrypt = 1;
        start = 1;
        #10;
        start = 0;
        wait(done);
        assert(data_out == (data_in ^ key), "Encryption failed for Test Case 3");
        $display("Input: %h, Key: %h, Encrypted: %h", data_in, key, data_out);
        #10;

        // Test Case 4: Reset functionality
        $display("\nTest Case 4: Reset functionality");
        data_in = 8'hAB;
        key = 8'hCD;
        encrypt = 1;
        start = 1;
        #10;
        rst = 1;  // Assert reset mid-operation
        #10;
        rst = 0;  // Release reset
        #10;
        assert(data_out == 8'b0 && done == 1'b0, "Reset failed to clear outputs for Test Case 4");
        $display("Reset properly cleared outputs.");
        #10;

        // Test Case 5: Zero input and key
        $display("\nTest Case 5: Zero input and key");
        data_in = 8'h00;
        key = 8'h00;
        encrypt = 1;
        start = 1;
        #10;
        start = 0;
        wait(done);
        assert(data_out == 8'h00, "Encryption failed for zero input and key in Test Case 5.");
        $display("Input: %h, Key: %h, Encrypted: %h", data_in, key, data_out);
        #10;

        // Test Case 6: Maximum input and key
        $display("\nTest Case 6: Maximum input and key");
        data_in = 8'hFF;
        key = 8'hFF;
        encrypt = 1;
        start = 1;
        #10;
        start = 0;
        wait(done);
        assert(data_out == 8'h00, "Encryption failed for max input and key in Test Case 6");
        $display("Input: %h, Key: %h, Encrypted: %h", data_in, key, data_out);
        #10;

        // Test Case 7: Rapid toggle between encryption and decryption
        $display("\nTest Case 7: Rapid toggle between encryption and decryption");
        data_in = 8'h6C;
        key = 8'h3A;
        for (i = 0; i < 5; i = i + 1) begin
            encrypt = (i % 2) ? 1 : 0;  // Alternate between encryption and decryption
            start = 1;
            #10;
            start = 0;
            wait(done);
            $display("Iteration %0d: Encrypt=%0b, Data Out=%h", i, encrypt, data_out);
        end
        $display("All tests completed successfully.");
        $finish;
    end
endmodule
