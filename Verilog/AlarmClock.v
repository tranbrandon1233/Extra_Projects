// Digital Clock with Alarm Module
module digital_clock (
    input wire clk_10hz,          // 10Hz clock input
    input wire rst,               // Active-high reset
    input wire set_time,          // Set current time enable
    input wire set_alarm,         // Set alarm time enable
    input wire alarm_en,          // Alarm enable
    input wire stop_alarm,        // Stop alarm manually
    input wire [5:0] in_seconds,  // Input seconds (0-59)
    input wire [5:0] in_minutes,  // Input minutes (0-59)
    input wire [4:0] in_hours,    // Input hours (0-23)
    output reg [5:0] seconds,     // Current seconds
    output reg [5:0] minutes,     // Current minutes
    output reg [4:0] hours,       // Current hours
    output reg alarm              // Alarm output signal
);

    // Internal registers for alarm time
    reg [5:0] alarm_seconds;
    reg [5:0] alarm_minutes;
    reg [4:0] alarm_hours;
    
    // Counter for generating 1-second tick from 10Hz clock
    reg [3:0] counter;
    
    // Generate 1-second tick from 10Hz clock
    wire tick = (counter == 4'd9);
    
    always @(posedge clk_10hz or posedge rst) begin
        if (rst) begin
            counter <= 4'd0;
        end else begin
            counter <= (counter == 4'd9) ? 4'd0 : counter + 1;
        end
    end
    
    // Clock logic
    always @(posedge clk_10hz or posedge rst) begin
        if (rst) begin
            seconds <= 6'd0;
            minutes <= 6'd0;
            hours <= 5'd0;
            alarm <= 1'b0;
        end else begin
            if (set_time) begin
                seconds <= in_seconds;
                minutes <= in_minutes;
                hours <= in_hours;
            end else if (tick) begin
                seconds <= (seconds == 6'd59) ? 6'd0 : seconds + 1;
                if (seconds == 6'd59) begin
                    minutes <= (minutes == 6'd59) ? 6'd0 : minutes + 1;
                    if (minutes == 6'd59) begin
                        hours <= (hours == 5'd23) ? 5'd0 : hours + 1;
                    end
                end
            end
        end
    end
    
    // Alarm logic
    always @(posedge clk_10hz or posedge rst) begin
        if (rst) begin
            alarm_seconds <= 6'd0;
            alarm_minutes <= 6'd0;
            alarm_hours <= 5'd0;
        end else if (set_alarm) begin
            alarm_seconds <= in_seconds;
            alarm_minutes <= in_minutes;
            alarm_hours <= in_hours;
        end
    end
    
    // Alarm trigger logic
    always @(posedge clk_10hz or posedge rst) begin
        if (rst) begin
            alarm <= 1'b0;
        end else if (stop_alarm) begin
            alarm <= 1'b0;
        end else if (alarm_en && 
                    seconds == alarm_seconds && 
                    minutes == alarm_minutes && 
                    hours == alarm_hours) begin
            alarm <= 1'b1;
        end
    end
    
endmodule

// Testbench
module digital_clock_tb;
    reg clk_10hz;
    reg rst;
    reg set_time;
    reg set_alarm;
    reg alarm_en;
    reg stop_alarm;
    reg [5:0] in_seconds;
    reg [5:0] in_minutes;
    reg [4:0] in_hours;
    
    wire [5:0] seconds;
    wire [5:0] minutes;
    wire [4:0] hours;
    wire alarm;
    
    // Instantiate the digital clock
    digital_clock dut (
        .clk_10hz(clk_10hz),
        .rst(rst),
        .set_time(set_time),
        .set_alarm(set_alarm),
        .alarm_en(alarm_en),
        .stop_alarm(stop_alarm),
        .in_seconds(in_seconds),
        .in_minutes(in_minutes),
        .in_hours(in_hours),
        .seconds(seconds),
        .minutes(minutes),
        .hours(hours),
        .alarm(alarm)
    );
    
    // Clock generation
    initial begin
        clk_10hz = 0;
        forever #50 clk_10hz = ~clk_10hz;
    end
    
    // Test stimulus
    initial begin
        // Initialize all inputs
        rst = 1;
        set_time = 0;
        set_alarm = 0;
        alarm_en = 0;
        stop_alarm = 0;
        in_seconds = 0;
        in_minutes = 0;
        in_hours = 0;
        
        // Test Case 1: Reset Check
        #100;
        rst = 0;
        $display("Test Case 1 - Reset Check:");
        $display("Time: %02d:%02d:%02d, Alarm: %b", hours, minutes, seconds, alarm);
        
        // Test Case 2: Set Time
        #100;
        set_time = 1;
        in_hours = 5'd23;
        in_minutes = 6'd59;
        in_seconds = 6'd55;
        #100;
        set_time = 0;
        $display("\nTest Case 2 - Set Time (23:59:55):");
        $display("Time: %02d:%02d:%02d, Alarm: %b", hours, minutes, seconds, alarm);
        
        // Test Case 3: Set Alarm
        set_alarm = 1;
        in_hours = 0;
        in_minutes = 0;
        in_seconds = 0;
        #100;
        set_alarm = 0;
        alarm_en = 1;
        $display("\nTest Case 3 - Set Alarm (00:00:00):");
        $display("Time: %02d:%02d:%02d, Alarm: %b", hours, minutes, seconds, alarm);
        
        // Test Case 4: Wait for alarm trigger
        #5000;
        $display("\nTest Case 4 - Alarm Trigger Check:");
        $display("Time: %02d:%02d:%02d, Alarm: %b", hours, minutes, seconds, alarm);
        
        // Test Case 5: Stop Alarm
        stop_alarm = 1;
        #100;
        stop_alarm = 0;
        $display("\nTest Case 5 - Stop Alarm:");
        $display("Time: %02d:%02d:%02d, Alarm: %b", hours, minutes, seconds, alarm);
        
        #1000;
        $finish;
    end
endmodule