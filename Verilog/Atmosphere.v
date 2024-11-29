

module sensor_interface_parallel_64 (
    input wire clk,
    input wire rst,
    input wire [2047:0] raw_pixel_data_in,
    output reg [2047:0] pixel_data_out,
    output reg data_valid
);
    integer i;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pixel_data_out <= 2048'b0;
            data_valid <= 0;
        end else begin
            pixel_data_out <= raw_pixel_data_in;
            data_valid <= 1;
        end
    end
endmodule

module atmospheric_correction_parallel_64 (
    input wire clk,
    input wire [2047:0] raw_pixel_data,
    input wire [31:0] correction_factor,
    output reg [2047:0] corrected_pixel_data
);
    integer i;
    reg [31:0] temp_pixel;
    always @(posedge clk) begin
        for (i = 0; i < 64; i = i + 1) begin
            temp_pixel = raw_pixel_data[(i * 32) +: 32];
            corrected_pixel_data[(i * 32) +: 32] <= temp_pixel * correction_factor;
        end
    end
endmodule

module output_interface_parallel_64 (
    input wire clk,
    input wire rst,
    input wire [2047:0] corrected_pixel_data,
    output reg [2047:0] memory_data_out,
    output reg [63:0] mem_write_enable
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            memory_data_out <= 2048'b0;
            mem_write_enable <= 64'b0;
        end else begin
            memory_data_out <= corrected_pixel_data;
            mem_write_enable <= 64'hFFFFFFFFFFFFFFFF;
        end
    end
endmodule

module fpga_accelerator_parallel_64 (
    input wire clk,
    input wire rst,
    input wire [2047:0] raw_pixel_data_in,
    input wire [31:0] correction_factor,
    output wire [2047:0] memory_data_out,
    output wire [63:0] mem_write_enable
);
    wire [2047:0] pixel_data;
    wire data_valid;
    wire [2047:0] corrected_data;

    sensor_interface_parallel_64 sensor_if (
        .clk(clk),
        .rst(rst),
        .raw_pixel_data_in(raw_pixel_data_in),
        .pixel_data_out(pixel_data),
        .data_valid(data_valid)
    );

    atmospheric_correction_parallel_64 correction_unit (
        .clk(clk),
        .raw_pixel_data(pixel_data),
        .correction_factor(correction_factor),
        .corrected_pixel_data(corrected_data)
    );

    output_interface_parallel_64 output_if (
        .clk(clk),
        .rst(rst),
        .corrected_pixel_data(corrected_data),
        .memory_data_out(memory_data_out),
        .mem_write_enable(mem_write_enable)
    );
endmodule

module tb_fpga_accelerator_parallel_64;

    reg clk;
    reg rst;
    reg [2047:0] raw_pixel_data_in;       // Flattened input for 64 raw pixels
    reg [31:0] correction_factor;         // Correction factor
    wire [2047:0] memory_data_out;        // Flattened output for 64 corrected pixels
    wire [63:0] mem_write_enable;         // Memory write enable signals

    // Instantiate the FPGA accelerator
    fpga_accelerator_parallel_64 accelerator (
        .clk(clk),
        .rst(rst),
        .raw_pixel_data_in(raw_pixel_data_in),
        .correction_factor(correction_factor),
        .memory_data_out(memory_data_out),
        .mem_write_enable(mem_write_enable)
    );

    // Clock generation: 10ns clock period
    always #5 clk = ~clk;

    // Task to initialize all pixel data
    task initialize_pixels(input [31:0] value);
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1) begin
                raw_pixel_data_in[(i * 32) +: 32] = value;
            end
        end
    endtask

    // Test Normal Operation
    task test_normal_operation;
        begin
            $display("\nTest: Normal operation with correction factor = 1");
            initialize_pixels(32'h12345678);    // Initialize all pixels to the same value
            correction_factor = 32'h00000001;   // Correction factor of 1
            #20;                                // Wait for signals to propagate
            $display("Expected: memory_data_out[0:63] = 0x12345678, mem_write_enable = 1");
            $display("Actual  : memory_data_out[0] = %h, mem_write_enable[0] = %b", memory_data_out[0+:32], mem_write_enable[0]);
        end
    endtask

    // Test All Zeros
    task test_all_zeros;
        begin
            $display("\nTest: All pixel inputs and correction factor set to zero");
            initialize_pixels(32'h00000000);    // All pixels are zero
            correction_factor = 32'h00000000;   // Correction factor of 0
            #20;
            $display("Expected: memory_data_out[0:63] = 0x00000000, mem_write_enable = 1");
            $display("Actual  : memory_data_out[0] = %h, mem_write_enable[0] = %b", memory_data_out[0+:32], mem_write_enable[0]);
        end
    endtask

    // Test Maximum Values
    task test_maximum_values;
        begin
            $display("\nTest: Maximum input and correction factor values");
            initialize_pixels(32'hFFFFFFFF);    // All pixels are maximum (all ones)
            correction_factor = 32'hFFFFFFFF;   // Maximum correction factor
            #20;
            $display("Expected: memory_data_out[0:63] = 0xFFFFFFFE (wrap-around), mem_write_enable = 1");
            $display("Actual  : memory_data_out[0] = %h, mem_write_enable[0] = %b", memory_data_out[0+:32], mem_write_enable[0]);
        end
    endtask

    // Test Mixed Values
    task test_mixed_values;
        begin
            $display("\nTest: Mixed pixel values with correction factor = 2");
            raw_pixel_data_in[0 +: 32] = 32'h00000001;
            raw_pixel_data_in[32 +: 32] = 32'h00000002;
            raw_pixel_data_in[64 +: 32] = 32'h00000003;
            raw_pixel_data_in[96 +: 32] = 32'h00000004;
            initialize_pixels(32'h00000005);    // Set remaining pixels to 5
            correction_factor = 32'h00000002;   // Correction factor of 2
            #20;
            $display("Expected: Doubled values for each pixel");
            $display("Actual  : memory_data_out[0] = %h, memory_data_out[1] = %h, memory_data_out[2] = %h, memory_data_out[3] = %h",
                      memory_data_out[0+:32], memory_data_out[32+:32], memory_data_out[64+:32], memory_data_out[96+:32]);
        end
    endtask

    // Test Reset Behavior
    task test_reset_behavior;
        begin
            $display("\nTest: System reset functionality");
            initialize_pixels(32'h12345678);    // Initialize all pixels to arbitrary values
            correction_factor = 32'h00000001;   // Correction factor of 1
            rst = 1;                            // Assert reset
            #10;
            rst = 0;                            // Deassert reset
            #20;
            $display("Expected: All outputs = 0 after reset, mem_write_enable = 0");
            $display("Actual  : memory_data_out[0] = %h, mem_write_enable[0] = %b", memory_data_out[0+:32], mem_write_enable[0]);
        end
    endtask

    // Test Zero Correction
    task test_zero_correction;
        begin
            $display("\nTest: Zero correction factor");
            initialize_pixels(32'hFFFFFFFF);    // Initialize all pixels to maximum
            correction_factor = 32'h00000000;   // Correction factor of 0
            #20;
            $display("Expected: All outputs = 0, mem_write_enable = 1");
            $display("Actual  : memory_data_out[0] = %h, mem_write_enable[0] = %b", memory_data_out[0+:32], mem_write_enable[0]);
        end
    endtask

    // Test Variable Correction Factor
    task test_variable_correction;
        begin
            $display("\nTest: Variable correction factor (Maximum)");
            initialize_pixels(32'h12345678);    // All pixels initialized to arbitrary value
            correction_factor = 32'hFFFFFFFF;   // Maximum correction factor
            #20;
            $display("Expected: Wrap-around behavior due to overflow");
            $display("Actual  : memory_data_out[0] = %h, mem_write_enable[0] = %b", memory_data_out[0+:32], mem_write_enable[0]);
        end
    endtask

    // Test Write Enable Timing
    task test_write_enable_timing;
        begin
            $display("\nTest: Write enable timing");
            initialize_pixels(32'h11111111);    // Initialize all pixels with arbitrary value
            correction_factor = 32'h00000001;   // Correction factor of 1
            #20;
            $display("Expected: mem_write_enable = 1 when data is valid");
            $display("Actual  : mem_write_enable[0] = %b", mem_write_enable[0]);
        end
    endtask

    // Main testbench logic
    initial begin
        clk = 0;            // Initialize clock
        rst = 1;            // Start with reset asserted

        // Deassert reset after some time
        #10 rst = 0;

        // Run all test cases
        test_normal_operation();
        test_all_zeros();
        test_maximum_values();
        test_mixed_values();
        test_reset_behavior();
        test_zero_correction();
        test_variable_correction();
        test_write_enable_timing();

        // End the simulation
        #50;
        $finish;
    end

endmodule