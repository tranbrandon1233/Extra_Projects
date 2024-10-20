def my_sqrt(number, precision=0.0001):
  """Computes the square root of a number using an iterative approach.

  Args:
    number: The number for which to calculate the square root.
    precision: The desired precision of the result.

  Returns:
    The approximate square root of the number.
  """

  if number < 0:
    raise ValueError("Cannot compute the square root of a negative number.")

  low = 0
  high = number
  guess = (low + high) / 2

  while abs(guess * guess - number) > precision:
    if guess * guess < number:
      low = guess
    else:
      high = guess
    guess = (low + high) / 2

  return guess

# Example usage:
number = 36
result = my_sqrt(number)
print(f"The square root of {number} is approximately {result}")