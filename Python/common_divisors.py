def common_divisors(numbers):
  """
  Finds common divisors for a list of numbers.

  Args:
    numbers: A list of integers.

  Returns:
    A list of common divisors.
  """

  if not numbers:
    return []

  divisors = set()
  # Start with divisors of the first number
  for i in range(1, abs(numbers[0]) + 1):
    if numbers[0] % i == 0:
      divisors.add(i)

  # Check if each divisor divides all other numbers
  for divisor in list(divisors):
    for num in numbers:
      if num % divisor != 0:
        divisors.remove(divisor)
        break

  return list(divisors)

# Example usage:
numbers = [12, 24, 36,48]
common_divs = common_divisors(numbers)
print(f"Common divisors of {numbers}: {common_divs}")