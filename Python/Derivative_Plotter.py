import numpy as np
import matplotlib.pyplot as plt

# Function to plot the polynomial and its derivatives
def plot_polynomial_and_derivatives(coefficients):
    x = np.linspace(-10, 10, 400)  # Generate x values
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Color cycle for different derivatives
    
    plt.figure(figsize=(8, 6))
    
    poly = np.poly1d(coefficients)  # Create the polynomial from coefficients
    derivative_order = 0

    while len(poly) > 0:  # Continue until the derivative is 0
        y = poly(x)  # Calculate the polynomial values for x
        plt.plot(x, y, label=f'Derivative order {derivative_order}', color=colors[derivative_order % len(colors)])
        poly = np.polyder(poly)  # Differentiate the polynomial
        derivative_order += 1

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial and its Derivatives')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
coefficients = [43,7,12,4,213]  # Represents the polynomial x^2 - 3x + 2
plot_polynomial_and_derivatives(coefficients)
