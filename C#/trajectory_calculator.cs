using System;

public class BulletSimulator
{
   public static void SimulateBallisticCurve(double initialSpeed, double initialAngleDegrees)
    {
        // Get initial values
        const double GRAVITATIONAL_ACCELERATION = 9.81; // m/s^2
        double initialAngleRadians = Math.PI * initialAngleDegrees / 180.0;
        double initialVelocityX = initialSpeed * Math.Cos(initialAngleRadians);
        double initialVelocityY = initialSpeed * Math.Sin(initialAngleRadians);
        // Print header of the table
        Console.WriteLine("Time\tDistance\tHeight");
        Console.WriteLine("------------------------------");
        
        // Calculate the distance and height
        double time = 0;
        double distance = initialVelocityX * time;
        double height = (initialVelocityY * time) - (0.5) * GRAVITATIONAL_ACCELERATION * Math.Pow(time, 2);
       
        // Loop until the bullet hits the ground
        while (height >= 0)
        {
            // Write the data to the console
            Console.WriteLine($"{time:0.00}\t{distance:0.00}\t{height:0.00}");
            // Calculate the distance and height again
            distance = initialVelocityX * time;
            height = (initialVelocityY * time) - (0.5) * GRAVITATIONAL_ACCELERATION * Math.Pow(time, 2);
            time += 0.01;
        }
    }

    public static void Main()
    {
        // Example usage
         double initialSpeed = 100; // Input (m/s)
         double initialAngle = 45; // Input (degrees)

         SimulateBallisticCurve(initialSpeed, initialAngle);
    }
 }