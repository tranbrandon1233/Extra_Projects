module microwave_fsm (
    input clk,
    input rst, // Reset signal
    input start, // Start input signal
    input stop, // Stop input signal
    input door_open, // Signal to detect whether or not the door is open
    input door_close, // Signal to detect whether or not the door is closed
    input [3:0] time_set, // Time in minutes
    output reg cooking, // Signal to control microwave
    output reg [3:0] remaining_time // Remaining time in minutes
);

    // State encoding using parameters
    parameter IDLE = 3'b000,
              SET_TIME = 3'b001,
              COOKING = 3'b010,
              PAUSED = 3'b011,
              DONE = 3'b100;

    reg [2:0] current_state, next_state;

    // State transition
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= IDLE;
            remaining_time <= 0;
            cooking <= 0;
        end else begin
            current_state <= next_state;
        end
    end

    // Next state logic
    always @(*) begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                cooking = 0;
                if (time_set > 0)
                    next_state = SET_TIME;
            end
            SET_TIME: begin
                cooking = 0;
                if (start)
                    next_state = COOKING;
            end
            COOKING: begin
                cooking = 1;
                if (remaining_time == 0)
                    next_state = DONE;
                else if (stop || door_open)
                    next_state = PAUSED;
            end
            PAUSED: begin
                cooking = 0;
                if (door_close && start)
                    next_state = COOKING;
            end
            DONE: begin
                cooking = 0;
                next_state = IDLE;
            end
        endcase
    end

    // Countdown logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            remaining_time <= 0;
        end else if (current_state == SET_TIME) begin
            remaining_time <= time_set;
        end else if (current_state == COOKING && remaining_time > 0) begin
            remaining_time <= remaining_time - 1;
        end
    end
endmodule


`timescale 1ns / 1ps

module microwave_fsm_tb;
    reg clk, rst, start, stop, door_open, door_close;
    reg [3:0] time_set;
    wire cooking;
    wire [3:0] remaining_time;

    microwave_fsm uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .stop(stop),
        .door_open(door_open),
        .door_close(door_close),
        .time_set(time_set),
        .cooking(cooking),
        .remaining_time(remaining_time)
    );

    // Clock generation
    always #5 clk = ~clk;

    initial begin
        // Initialize inputs
        clk = 0; rst = 0; start = 0; stop = 0; 
        door_open = 0; door_close = 0; time_set = 0;

        // Reset the system
        $display("Starting Microwave FSM Testbench");
        rst = 1; #10;
        rst = 0;

        // Set time and start cooking
        $display("Setting time to 5 minutes");
        time_set = 4'd5; #10;

        $display("Starting cooking");
        start = 1; #10;
        start = 0;

        // Simulate cooking for 3 cycles
        repeat (3) begin
            #10;
            $display("Time remaining: %d, Cooking: %b", remaining_time, cooking);
        end

        // Pause cooking by opening the door
        $display("Pausing cooking (door open)");
        door_open = 1; #10;
        $display("Time remaining: %d, Cooking: %b", remaining_time, cooking);
        door_open = 0;

        // Resume cooking
        $display("Resuming cooking");
        door_close = 1; start = 1; #10;
        door_close = 0; start = 0;

        // Finish cooking
        repeat (2) begin
            #10;
            $display("Time remaining: %d, Cooking: %b", remaining_time, cooking);
        end

        // Cooking done
        #10;
        $display("Cooking done. Remaining Time: %d, Cooking: %b", remaining_time, cooking);

        $finish;
    end
endmodule