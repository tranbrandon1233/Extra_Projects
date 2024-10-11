import random
import math

def stochastic_hill_climbing(objective_function, bounds, iterations, step_size):
  """
  Stochastic Hill Climbing algorithm for optimization.

  Args:
    objective_function: The function to be minimized.
    bounds: A list of tuples representing the lower and upper bounds for each dimension.
    iterations: The number of iterations to run the algorithm.
    step_size: The size of the step to take in each dimension.

  Returns:
    A tuple containing the best solution found and its corresponding objective function value.
  """

  # Initialize a random solution within the bounds
  current_solution = random.randint(bounds[0], bounds[1])
  current_value = objective_function(current_solution)

  for _ in range(iterations):
    # Generate a random neighbor by adding a small random step to the current solution
    neighbor = current_solution + step_size*random.randint(-1, 1) 

    # Ensure the neighbor is within the bounds
    neighbor = min(max(neighbor, bounds[0]), bounds[1])

    # Evaluate the neighbor's objective function value
    neighbor_value = objective_function(neighbor)

    # If the neighbor is better, accept it as the new current solution
    if neighbor_value < current_value:
      current_solution = neighbor
      current_value = neighbor_value

  return current_solution, current_value


# Example usage:
def sphere_function(x):
  """A simple sphere function to minimize."""
  return x**2 + 2*math.sin(3*x)

# Define the bounds for each dimension
bounds = [-3, 0]

# Run the stochastic hill climbing algorithm
best_solution, best_value = stochastic_hill_climbing(sphere_function, bounds, iterations=10000, step_size=0.01)

print("Best solution found:", best_solution)
print("Best objective function value:", best_value) 