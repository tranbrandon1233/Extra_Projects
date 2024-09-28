using System;
using System.Linq;

class Program
{
    /// <summary>
    /// Returns the minimum number of boxes required to store all apples from different packs.
    /// Apples from the same pack can be split across multiple boxes.
    /// </summary>
    /// <param name="apples">An array where each element represents the number of apples in each pack.</param>
    /// <param name="capacity">An array where each element represents the capacity of each box.</param>
    /// <returns>The minimum number of boxes required to store all apples, or -1 if it's not possible.</returns>
    public static int MinimumBoxes(int[] apples, int[] capacity)
    {
        // Check if the apples array is empty
        if(apples.Length == 0){
            return 0;
        }
        
        // Calculate the total number of apples
        int totalApples = apples.Sum();
        
        // Sort the capacity array in descending order to use the largest boxes first
        Array.Sort(capacity);
        Array.Reverse(capacity);

        // Variable to keep track of the current sum of capacities and number of boxes used
        int currentCapacity = 0;
        int boxesUsed = 0;

        // Iterate over the sorted capacity array and accumulate the capacity
        foreach (int box in capacity)
        {
            currentCapacity += box;
            boxesUsed++;

            // If current capacity is greater than or equal to the total apples, return the box count
            if (currentCapacity >= totalApples)
            {
                return boxesUsed;
            }
        }

        // If all boxes are used and it's still not enough, return -1
        return -1;
    }

    // Test cases to validate the solution
    public static void Main(string[] args)
    {
        // Example test case 1
        int[] apples1 = { 1, 3, 2 };
        int[] capacity1 = { 4, 3, 1, 5, 2 };
        Console.WriteLine(MinimumBoxes(apples1, capacity1)); // Expected Output: 2

        // Example test case 2
        int[] apples2 = { 5, 5, 5 };
        int[] capacity2 = { 2, 4, 2, 7 };
        Console.WriteLine(MinimumBoxes(apples2, capacity2)); // Expected Output: 4

        // Edge case: when apples array is empty
        int[] apples3 = { };
        int[] capacity3 = { 3, 2, 1 };
        Console.WriteLine(MinimumBoxes(apples3, capacity3)); // Expected Output: 0 (no apples to distribute)

        // Edge case: when capacity array is empty
        int[] apples4 = { 5, 5, 5 };
        int[] capacity4 = { };
        Console.WriteLine(MinimumBoxes(apples4, capacity4)); // Expected Output: -1 (no boxes available)

        // Edge case: when the capacity is less than total apples
        int[] apples5 = { 5, 5, 5 };
        int[] capacity5 = { 3, 3, 2 };
        Console.WriteLine(MinimumBoxes(apples5, capacity5)); // Expected Output: -1 (not enough capacity)
    }
}