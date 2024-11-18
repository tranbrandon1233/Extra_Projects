module CarParkingSystem(
    input clk,
    input reset,
    input car_enter,
    input car_exit,
    output reg [7:0] car_count
);

always @(posedge clk or posedge reset) begin
    if (reset)
        car_count <= 8'd0;
    else if (car_enter && car_count == 8'd255)
        car_count <= 8'd255; // Stop incrementing at 255
    else if (car_enter && car_count < 8'd255)
        car_count <= car_count + 1;
    else if (car_exit && car_count > 8'd0)
        car_count <= car_count - 1;
end

endmodule

`timescale 1ns/1ps

module CarParkingSystem_tb();
    // Test bench signals
    reg clk;
    reg reset;
    reg car_enter;
    reg car_exit;
    wire [7:0] car_count;
    
    // Instantiate the CarParkingSystem
    CarParkingSystem dut (
        .clk(clk),
        .reset(reset),
        .car_enter(car_enter),
        .car_exit(car_exit),
        .car_count(car_count)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        // Initialize waveform dump
        $dumpfile("car_parking.vcd");
        $dumpvars(0, CarParkingSystem_tb);
        
        // Initialize inputs
        reset = 0;
        car_enter = 0;
        car_exit = 0;
        
        // Test 1: Reset check
        #10 reset = 1;
        #10 reset = 0;
        
        // Test 2: Normal increment
        repeat(5) begin
            #10 car_enter = 1;
            #10 car_enter = 0;
        end
        
        // Test 3: Normal decrement
        repeat(2) begin
            #10 car_exit = 1;
            #10 car_exit = 0;
        end
        
        // Test 4: Boundary condition - approach 255
        repeat(252) begin
            #10 car_enter = 1;
            #10 car_enter = 0;
        end
        
        // Test 5: Try to exceed 255
        repeat(5) begin
            #10 car_enter = 1;
            #10 car_enter = 0;
        end
        
        // Test 6: Decrement from 255
        repeat(5) begin
            #10 car_exit = 1;
            #10 car_exit = 0;
        end
        
        // Test 7: Try to decrement below 0
        #10 reset = 1;
        #10 reset = 0;
        repeat(5) begin
            #10 car_exit = 1;
            #10 car_exit = 0;
        end
        
        // Test 8: Simultaneous enter and exit (enter should take precedence)
        #10 car_enter = 1;
        car_exit = 1;
        #10 car_enter = 0;
        car_exit = 0;
        
        // End simulation
        #100 $finish;
    end
    
    // Monitor changes
    initial begin
        $monitor("Time=%0t reset=%b car_enter=%b car_exit=%b car_count=%d",
                 $time, reset, car_enter, car_exit, car_count);
    end
    
    // Assertions
    always @(posedge clk) begin
        // Assert car_count never exceeds 255
        if (car_count > 255) begin
            $display("Error: car_count exceeded 255!");
            $finish;
        end
        
        // Assert car_count never goes below 0 (although this is impossible with unsigned number)
        if (car_count < 0) begin
            $display("Error: car_count went below 0!");
            $finish;
        end
    end
    
endmodule