module ProjectileMotion(
    input wire [15:0] initialVelocity,      // Initial velocity in m/s
    input wire [15:0] launchAngle,          // Launch angle in degrees
    input wire [1:0] selectOutput,          // Select between timeOfFlight, maxHeight, range (00 = Time, 01 = Height, 10 = Range, 11 = Vertical Velocity)
    input wire [15:0] customGravity,        // Custom gravity input for different environments (scaled)
    input wire [15:0] angleScaling,         // Scaling factor for angle precision (scaled)
    input wire clk,                         // Clock signal for controlling output
    input wire reset,                       // Reset signal
    output reg [15:0] result                // Final result based on selected output
);

// Parameters for gravity and Pi approximation
parameter PI = 16'd314;                    // Approximate value of Pi (3.14 scaled by 100)
parameter MAX_ANGLE = 16'd90;             // Maximum launch angle

// Internal signals
reg [15:0] angleRadians;
reg [15:0] sinAngle;
reg [15:0] cosAngle;
reg [15:0] sin2Angle;
reg [15:0] timeOfFlight, maxHeight, range;
reg [15:0] verticalVelocity, horizontalVelocity, timeToMaxHeight;

// Sine and Cosine Lookup Table (LUT) for angles 0 to 90 degrees
reg [15:0] sine_LUT [0:90];
reg [15:0] cosine_LUT [0:90];

initial begin
    // Sine values for every 5 degrees (scaled by 1000)
    sine_LUT[0] = 16'd0;    // sin(0)
    sine_LUT[5] = 16'd87;   // sin(5)
    sine_LUT[10] = 16'd174; // sin(10)
    sine_LUT[15] = 16'd259; // sin(15)
    sine_LUT[20] = 16'd342; // sin(20)
    sine_LUT[25] = 16'd423; // sin(25)
    sine_LUT[30] = 16'd500; // sin(30)
    sine_LUT[35] = 16'd574; // sin(35)
    sine_LUT[40] = 16'd643; // sin(40)
    sine_LUT[45] = 16'd707; // sin(45)
    sine_LUT[50] = 16'd766; // sin(50)
    sine_LUT[55] = 16'd819; // sin(55)
    sine_LUT[60] = 16'd866; // sin(60)
    sine_LUT[65] = 16'd906; // sin(65)
    sine_LUT[70] = 16'd940; // sin(70)
    sine_LUT[75] = 16'd966; // sin(75)
    sine_LUT[80] = 16'd985; // sin(80)
    sine_LUT[85] = 16'd996; // sin(85)
    sine_LUT[90] = 16'd1000; // sin(90)

    // Cosine values for every 5 degrees (scaled by 1000)
    cosine_LUT[0] = 16'd1000; // cos(0)
    cosine_LUT[5] = 16'd996;  // cos(5)
    cosine_LUT[10] = 16'd985; // cos(10)
    cosine_LUT[15] = 16'd966; // cos(15)
    cosine_LUT[20] = 16'd940; // cos(20)
    cosine_LUT[25] = 16'd906; // cos(25)
    cosine_LUT[30] = 16'd866; // cos(30)
    cosine_LUT[35] = 16'd819; // cos(35)
    cosine_LUT[40] = 16'd766; // cos(40)
    cosine_LUT[45] = 16'd707; // cos(45)
    cosine_LUT[50] = 16'd643; // cos(50)
    cosine_LUT[55] = 16'd574; // cos(55)
    cosine_LUT[60] = 16'd500; // cos(60)
    cosine_LUT[65] = 16'd423; // cos(65)
    cosine_LUT[70] = 16'd342; // cos(70)
    cosine_LUT[75] = 16'd259; // cos(75)
    cosine_LUT[80] = 16'd174; // cos(80)
    cosine_LUT[85] = 16'd87;  // cos(85)
    cosine_LUT[90] = 16'd0;   // cos(90)
end

// Function to convert angle to radians
always @(*) begin
    angleRadians = (launchAngle * PI) / 16'd180;
end

// Function to calculate sine and cosine for a given angle using the LUT
always @(*) begin
    if (launchAngle <= 90) begin
        sinAngle = sine_LUT[launchAngle];  // Direct access to LUT for sine
        cosAngle = cosine_LUT[launchAngle]; // Direct access to LUT for cosine
    end else begin
        sinAngle = 16'd0;  // For angles > 90, sin(angle) is assumed as 0 (simplified)
        cosAngle = 16'd0;  // For angles > 90, cos(angle) is assumed as 0 (simplified)
    end
end

// Calculate projectile motion parameters
always @(posedge clk or posedge reset) begin
    if (reset) begin
        timeOfFlight <= 16'd0;
        maxHeight <= 16'd0;
        range <= 16'd0;
        verticalVelocity <= 16'd0;
        horizontalVelocity <= 16'd0;
        timeToMaxHeight <= 16'd0;
    end
    else begin
        // Calculate time of flight: (2 * initialVelocity * sin(angleRadians)) / gravity
        timeOfFlight <= (2 * initialVelocity * sinAngle) / customGravity;

        // Calculate maximum height: (initialVelocity^2 * sin(angleRadians)^2) / (2 * gravity)
        maxHeight <= ((initialVelocity * sinAngle) ** 2) / (2 * customGravity);

        // Calculate range: (initialVelocity^2 * sin(2 * angleRadians)) / gravity
        range <= ((initialVelocity ** 2) * (2 * sinAngle * cosAngle)) / customGravity;

        // Calculate vertical velocity: (initialVelocity * sin(angleRadians))
        verticalVelocity <= initialVelocity * sinAngle;

        // Calculate horizontal velocity: (initialVelocity * cos(angleRadians))
        horizontalVelocity <= initialVelocity * cosAngle;

        // Calculate time to reach maximum height: (initialVelocity * sin(angleRadians)) / gravity
        timeToMaxHeight <= (initialVelocity * sinAngle) / customGravity;
    end
end

// Output the selected result based on the selectOutput signal
always @(posedge clk) begin
    case (selectOutput)
        2'b00: result <= timeOfFlight;        // Time of flight
        2'b01: result <= maxHeight;           // Maximum height
        2'b10: result <= range;               // Range
        2'b11: result <= verticalVelocity;    // Vertical velocity
        default: result <= 16'd0;             // Default to 0 if no valid output
    endcase
end

endmodule

module ProjectileMotion_tb;

    // Test bench signals
    reg [15:0] initialVelocity;
    reg [15:0] launchAngle;
    reg [1:0] selectOutput;
    reg [15:0] customGravity;
    reg [15:0] angleScaling;
    reg clk;
    reg reset;
    wire [15:0] result;

    // Instantiate the ProjectileMotion module
    ProjectileMotion pm (
        .initialVelocity(initialVelocity),
        .launchAngle(launchAngle),
        .selectOutput(selectOutput),
        .customGravity(customGravity),
        .angleScaling(angleScaling),
        .clk(clk),
        .reset(reset),
        .result(result)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test stimulus
    initial begin
        // Initialize waveform dump
        $dumpfile("projectile_motion.vcd");
        $dumpvars(0, ProjectileMotion_tb);

        // Initialize inputs
        reset = 1;
        initialVelocity = 16'd0;
        launchAngle = 16'd0;
        selectOutput = 2'b00;
        customGravity = 16'd981; // 9.81 m/s² scaled by 100
        angleScaling = 16'd100;  // Scale factor of 100

        // Wait for 2 clock cycles and release reset
        #20;
        reset = 0;

        // Test Case 1: 45-degree launch
        initialVelocity = 16'd1000; // 10 m/s scaled by 100
        launchAngle = 16'd45;
        
        // Test all outputs
        selectOutput = 2'b00; // Time of flight
        #20;
        $display("Test Case 1 (45 degrees, 10 m/s):");
        $display("Time of Flight: %d", result);
        
        selectOutput = 2'b01; // Maximum height
        #20;
        $display("Maximum Height: %d", result);
        
        selectOutput = 2'b10; // Range
        #20;
        $display("Range: %d", result);
        
        selectOutput = 2'b11; // Vertical velocity
        #20;
        $display("Vertical Velocity: %d", result);

        // Test Case 2: 30-degree launch
        initialVelocity = 16'd2000; // 20 m/s scaled by 100
        launchAngle = 16'd30;
        
        // Test all outputs
        selectOutput = 2'b00;
        #20;
        $display("\nTest Case 2 (30 degrees, 20 m/s):");
        $display("Time of Flight: %d", result);
        
        selectOutput = 2'b01;
        #20;
        $display("Maximum Height: %d", result);
        
        selectOutput = 2'b10;
        #20;
        $display("Range: %d", result);
        
        selectOutput = 2'b11;
        #20;
        $display("Vertical Velocity: %d", result);

        // Add more test cases as needed

        #100;
        $finish;
    end

endmodule