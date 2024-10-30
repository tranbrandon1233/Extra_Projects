// Online C# Editor for free
// Write, Edit and Run your C# code using C# Online Compiler

using System;

using System.Collections.Generic;
using System.Linq;


public static class StatisticsExtensions
{
  /// <summary>
  /// Calculates the variance and standard deviation of a list of doubles.
  /// </summary>
  /// <param name="values">The list of doubles.</param>
  /// <param name="variance">The calculated variance.</param>
  /// <param name="standardDeviation">The calculated standard deviation.</param>
  public static void CalculateVarianceAndStandardDeviation(this List<double> values, out double variance, out double standardDeviation)
      {
        if (values == null || values.Count == 0)
        {
          throw new ArgumentException("The list of values cannot be null or empty.");
        }
    
        // Calculate the mean
        double mean = values.Average();
    
        // Calculate the sum of squared differences from the mean
        double sumOfSquaredDifferences = values.Sum(value => Math.Pow(value - mean, 2));
    
        // Calculate the variance
        variance = sumOfSquaredDifferences / values.Count;
    
        // Calculate the standard deviation
        standardDeviation = Math.Sqrt(variance);
      }

    public static void Main(string[] args)
    {
       List<double> numbers = new List<double> { 1.5, 2.8, 3.1, 4.6, 5.2 };
        
        double variance, standardDeviation;
        numbers.CalculateVarianceAndStandardDeviation(out variance, out standardDeviation);
        
        Console.WriteLine($"Variance: {variance}");
        Console.WriteLine($"Standard Deviation: {standardDeviation}");
    }
}