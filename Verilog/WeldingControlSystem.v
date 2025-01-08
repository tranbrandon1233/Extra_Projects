module wpcs (  
    input clk,  
    input reset,  
    input start,  
    input [15:0] temperature,  
    input [15:0] pressure,  
    input [15:0] voltage,  
    output reg alarm,  
    output reg [1:0] current_state,  
    output reg [15:0] last_temperature, 
    output reg [15:0] last_pressure,  
    output reg [15:0] last_voltage,  
    output reg [31:0] fail_count // New feature: Failure counter  
);  

    parameter IDLE = 2'b00;  
    parameter MONITOR = 2'b01;  
    parameter FAILURE = 2'b10;  
    parameter RESET = 2'b11;  

    parameter TEMP_LIMIT = 16'd200;  
    parameter PRESS_LIMIT = 16'd150;
    parameter VOLT_LIMIT = 16'd300;

    reg [1:0] state, next_state;  

    always @(posedge clk or posedge reset) begin  
        if (reset) begin  
            state <= IDLE;  
            alarm <= 1'b0;  
            last_temperature <= 16'd0;
            last_pressure <= 16'd0;  
            last_voltage <= 16'd0;  
            fail_count <= 32'd0; // Initialize the failure counter  
        end else begin  
            state <= next_state;  
        end  
    end  

    always @(*) begin  
        next_state = state;  
        alarm = 1'b0;  

        case (state)  
            IDLE: begin  
                if (start) begin  
                    next_state = MONITOR;  
                end  
            end  

            MONITOR: begin  
                last_temperature = temperature;  
                last_pressure = pressure;  
                last_voltage = voltage;  

                if (temperature > TEMP_LIMIT || pressure > PRESS_LIMIT || voltage > VOLT_LIMIT) begin
                    alarm = 1'b1;  
                    next_state = FAILURE;  
                end
            end  

            FAILURE: begin  
                if (reset) begin  
                    next_state = IDLE;  
                    alarm = 1'b0;  
                end else if (start) begin  
                    next_state = MONITOR;  
                end
            end  

            RESET: begin  
                next_state = IDLE;  
            end  

            default: begin  
                next_state = IDLE;  
            end  
        endcase  
    end  

    always @(posedge clk or posedge reset) begin  
        if (reset) begin  
            current_state <= IDLE;  
        end else begin  
            current_state <= state;  
        end  
    end  

    // Increment the fail_count whenever entering MONITOR state  
    always @(posedge clk or posedge reset) begin  
        if (reset) begin  
            fail_count <= 32'd0;  
        end else if (state == MONITOR && next_state == FAILURE) begin  
            fail_count <= fail_count + 1;  
        end  
    end  

endmodule


`timescale 1ns / 1ps

module wpcs_tb;

    // Inputs
    reg clk;
    reg reset;
    reg start;
    reg [15:0] temperature;
    reg [15:0] pressure;
    reg [15:0] voltage;

    // Outputs
    wire alarm;
    wire [1:0] current_state;
    wire [15:0] last_temperature;
    wire [15:0] last_pressure;
    wire [15:0] last_voltage;
    wire [31:0] fail_count;

    // Instantiate the Unit Under Test (UUT)
    wpcs uut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .temperature(temperature),
        .pressure(pressure),
        .voltage(voltage),
        .alarm(alarm),
        .current_state(current_state),
        .last_temperature(last_temperature),
        .last_pressure(last_pressure),
        .last_voltage(last_voltage),
        .fail_count(fail_count)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    // Test sequence
    initial begin
        // Initialize inputs
        reset = 1;
        start = 0;
        temperature = 16'd0;
        pressure = 16'd0;
        voltage = 16'd0;

        // Wait for global reset
        #20;
        reset = 0;

        // Test 1: Transition from IDLE to MONITOR
        start = 1;
        #10;
        start = 0;
        temperature = 16'd100; // Normal temperature
        pressure = 16'd100;    // Normal pressure
        voltage = 16'd100;     // Normal voltage

        #50; // Let it stay in MONITOR for a while

        // Test 2: Trigger FAILURE and then RESET
        temperature = 16'd250; // Over temperature to trigger alarm
        #20;

        reset = 1; // Reset the system
        #10;
        reset = 0;

        // Test 3: Re-enter MONITOR multiple times
        start = 1;
        #10;
        start = 0;
        #50;

        start = 1;
        #10;
        start = 0;
        #50;

        start = 1;
        #10;
        start = 0;
        #50;

        // End simulation
        $stop;
    end

    // Monitor outputs
    initial begin
        $monitor("Time=%0t | Reset=%b | Start=%b | Temp=%d | Pressure=%d | Voltage=%d | Alarm=%b | State=%b | fail_count=%d",
            $time, reset, start, temperature, pressure, voltage, alarm, current_state, fail_count);
    end

endmodule
