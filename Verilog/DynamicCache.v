module DynamicCache (
    input wire clk,                      // Clock signal
    input wire reset,                    // Reset signal
    input wire [15:0] address,           // 16-bit memory address
    input wire read_write,               // Read/Write signal (0 = Read, 1 = Write)
    input wire [31:0] write_data,        // Data to write (for write operations)
    input wire [1:0] partition_select,   // Cache partition selector (2-bit for 4 partitions)
    input wire [3:0] workload_intensity, // Workload intensity signal for DVFS (0 = light, 15 = heavy)
    output reg [31:0] read_data,         // Data output (for read operations)
    output reg hit                       // Cache hit flag
);
    // Cache parameters
    parameter CACHE_LINES = 16;          // Total number of cache lines
    parameter PARTITIONS = 4;            // Number of partitions
    parameter LINE_SIZE = 32;            // Size of each line in bits

    // Partition storage
    reg [31:0] cache_data[PARTITIONS-1:0][CACHE_LINES/PARTITIONS-1:0];
    reg [11:0] cache_tags[PARTITIONS-1:0][CACHE_LINES/PARTITIONS-1:0];
    reg valid[PARTITIONS-1:0][CACHE_LINES/PARTITIONS-1:0];
    reg [7:0] access_count[PARTITIONS-1:0][CACHE_LINES/PARTITIONS-1:0]; // Energy-aware metric

    // Dynamic voltage and frequency (simulated)
    reg [7:0] voltage;                  // Simulated voltage level
    reg [7:0] frequency;                // Simulated clock frequency

    // Integer for partition selection
    integer selected_partition;

    // Index and tag extraction
    wire [3:0] index;                   // 4-bit index for each partition
    wire [11:0] tag;                    // Remaining 12 bits as the tag
    assign index = address[3:0];        // Lower 4 bits as index
    assign tag = address[15:4];         // Upper 12 bits as tag

    // Dynamic frequency scaling based on workload intensity
    always @(*) begin
        case (workload_intensity)
            4'b0000: begin voltage = 8'd80; frequency = 8'd50; end // Light workload
            4'b1111: begin voltage = 8'd120; frequency = 8'd100; end // Heavy workload
            default: begin voltage = 8'd100; frequency = 8'd75; end // Moderate
        endcase
    end

    // Initialize cache on reset
    integer i, j;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < PARTITIONS; i = i + 1) begin
                for (j = 0; j < CACHE_LINES / PARTITIONS; j = j + 1) begin
                    cache_data[i][j] <= 0;
                    cache_tags[i][j] <= 0;
                    valid[i][j] <= 0;
                    access_count[i][j] <= 0;
                end
            end
            hit <= 0;
            read_data <= 32'b0;
        end
    end

    // Cache read/write operation
    always @(posedge clk) begin
        if (!reset) begin
            selected_partition = partition_select; // Explicitly set partition
            if (valid[selected_partition][index] && (cache_tags[selected_partition][index] == tag)) begin
                // Cache hit
                hit <= 1;
                access_count[selected_partition][index] <= access_count[selected_partition][index] + 1;
                if (read_write == 0) begin
                    // Read operation
                    read_data <= cache_data[selected_partition][index];
                end else begin
                    // Write operation
                    cache_data[selected_partition][index] <= write_data;
                end
            end else begin
                // Cache miss
                hit <= 0;
                if (read_write == 0) begin
                    // Simulated memory fetch for read miss
                    read_data <= 32'hDEADBEEF;
                end else begin
                    // Directly write to memory for write miss
                    cache_data[selected_partition][index] <= write_data;
                end
                cache_tags[selected_partition][index] <= tag;
                valid[selected_partition][index] <= 1;
                access_count[selected_partition][index] <= 1; // Reset access count on new entry
            end
        end
    end

endmodule


module Fully_Revised_Cache_Testbench;
    // Testbench signals
    reg clk;
    reg reset;
    reg [15:0] address;
    reg read_write; // 0 = Read, 1 = Write
    reg [31:0] write_data;
    reg [1:0] partition_select; // Partition selector (2 bits for 4 partitions)
    reg [3:0] workload_intensity; // Workload intensity (0 = light, 15 = heavy)
    wire [31:0] read_data;
    wire hit;

    // Instantiate the Cache module
    DynamicCache uut (
        .clk(clk),
        .reset(reset),
        .address(address),
        .read_write(read_write),
        .write_data(write_data),
        .partition_select(partition_select),
        .workload_intensity(workload_intensity),
        .read_data(read_data),
        .hit(hit)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Test sequence
    initial begin
        // Initialize signals
        clk = 0;
        reset = 1;
        address = 0;
        read_write = 0;
        write_data = 0;
        partition_select = 2'b00; // Start with partition 0
        workload_intensity = 4'b0000; // Start with light workload

        // Wait for reset to complete
        #10 reset = 0;

        // Test case 1: Accessing different partitions
        partition_select = 2'b00;
        address = 16'h0001;
        write_data = 32'hA5A5A5A5;
        read_write = 1; // Write operation
        #10;

        partition_select = 2'b01;
        address = 16'h0011;
        write_data = 32'h5A5A5A5A;
        read_write = 1; // Write operation
        #10;

        partition_select = 2'b10;
        address = 16'h0021;
        read_write = 0; // Read operation
        #10;

        // Test case 2: Workload Intensity and DVFS
        workload_intensity = 4'b1111; // Heavy workload
        address = 16'h0031;
        read_write = 0; // Read operation
        #10;

        workload_intensity = 4'b0000; // Light workload
        address = 16'h0041;
        read_write = 0; // Read operation
        #10;

        // Test case 3: Energy-Aware Replacement Policy
        partition_select = 2'b11; // Partition 3
        repeat (5) begin
            address = address + 1;
            write_data = 32'hDEADBEEF;
            read_write = 1; // Write operations
            #10;
        end

        partition_select = 2'b11;
        address = 16'h0051;
        read_write = 0; // Read operation to validate retention of frequently accessed lines
        #10;

        // End of simulation
        $finish;
    end

    // Monitor signals
    initial begin
        $monitor("Time=%0t | Partition=%b | Address=%h | RW=%b | Write Data=%h | Read Data=%h | Hit=%b | Workload=%b",
                 $time, partition_select, address, read_write, write_data, read_data, hit, workload_intensity);
    end

endmodule