module priorityencoder_dynamic #(parameter WIDTH = 8) (
    input wire en,                     // Enable signal
    input wire [WIDTH-1:0] i,          // Input signals (dynamically sized)
    output reg [clog2(WIDTH)-1:0] y,   // Output: Encoded priority
    output reg error,                  // Error flag for no active input
    output reg priority_detected       // High priority flag
);

    // Calculate the log2 of the width for indexing
    function integer clog2(input integer value);
        integer i;
        begin
            i = value - 1;
            for (clog2 = 0; i > 0; clog2 = clog2 + 1)
                i = i >> 1;
        end
    endfunction

    always @(en, i) begin
        if (en == 1) begin
            // Reset error and priority flags
            error = 0;
            priority_detected = 0;
            // Priority Encoder Logic
            if (i[WIDTH-1] == 1) begin
                y = WIDTH-1;                  // Highest priority
                priority_detected = 1;
            end else if (i[WIDTH-2] == 1) begin
                y = WIDTH-2;
                priority_detected = 1;
            end else if (i[WIDTH-3] == 1) begin
                y = WIDTH-3;
                priority_detected = 1;
            end else if (i[WIDTH-4] == 1) begin
                y = WIDTH-4;
                priority_detected = 1;
            end else if (i[WIDTH-5] == 1) begin
                y = WIDTH-5;
                priority_detected = 1;
            end else if (i[WIDTH-6] == 1) begin
                y = WIDTH-6;
                priority_detected = 1;
            end else if (i[WIDTH-7] == 1) begin
                y = WIDTH-7;
                priority_detected = 1;
            end else if (i[WIDTH-8] == 1) begin
                y = WIDTH-8;
                priority_detected = 1;
            end else begin
                y = {clog2(WIDTH){1'b0}};  // Default to zero if no input is active
                error = 1;                  // Error: No active input
            end
        end else begin
            y = {clog2(WIDTH){1'bz}};   // High impedance state
            error = 1;                   // No valid output when disabled
            priority_detected = 0;
        end
    end

endmodule

module tb_priorityencoder_dynamic;

    reg [7:0] i;                      // 8-bit input signal
    reg en;                            // Enable signal
    wire [2:0] y;                     // Priority encoder output (3-bit)
    wire error;                        // Error flag
    wire priority_detected;            // Priority detected flag

    // Instantiate the dynamic priority encoder with 8 inputs
    priorityencoder_dynamic #(8) dut (
        .en(en),
        .i(i),
        .y(y),
        .error(error),
        .priority_detected(priority_detected)
    );

    initial begin
        // Monitor outputs
        $monitor("en=%b i=%b y=%b error=%b priority_detected=%b", en, i, y, error, priority_detected);

        // Test sequence with different inputs
        en = 1; i = 8'b10000000; #5;  // Highest priority
        en = 1; i = 8'b01000000; #5;
        en = 1; i = 8'b00100000; #5;
        en = 1; i = 8'b00010000; #5;
        en = 1; i = 8'b00001000; #5;
        en = 1; i = 8'b00000100; #5;
        en = 1; i = 8'b00000010; #5;
        en = 1; i = 8'b00000001; #5;  // Lowest priority
        en = 0; i = 8'b00000001; #5;  // Disabled, high impedance
        en = 1; i = 8'b00000000; #5;  // No active input, error flag should be 1

        $finish;
    end

endmodule
