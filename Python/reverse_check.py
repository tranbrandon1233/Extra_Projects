def check_reverse_encompassment(target: list, competition: list) -> bool:
  """
  This function determines if the 'target' list encompasses all elements of the 'competition' list in reversed order. 

  Args:
    target: The list to be evaluated as the encompassing list.
    competition: The list to be evaluated as the enclosed list.

  Returns:
    True if the 'target' list encompasses the reversed 'competition' list, False otherwise.

  Raises:
    TypeError: If either 'target' or 'competition' is not a list.
  """

  # Check if both inputs are valid lists
  if not isinstance(target, list) or not isinstance(competition, list):
    raise TypeError("Both 'target' and 'competition' must be lists.")

  # Initialize pointers for iteration
  target_index = 0
  competition_index = len(competition) - 1

  # Iterate until either list is exhausted
  while target_index < len(target) and competition_index >= 0:
    # Compare elements at current indices in both lists
    if target[target_index] == competition[competition_index]:
      # If elements match, advance both pointers
      target_index += 1
      competition_index -= 1
    else:
      # If elements don't match, advance only the target pointer
      target_index += 1

  # The function returns True only if the entire 'competition' list has been 'consumed' (competition_index < 0)
  return competition_index < 0

print(check_reverse_encompassment([1, 2, 3, 4, 5, 6], [6, 4, 2]))  # True
print(check_reverse_encompassment([1, 2, 3], [6, 4, 2]))          # False
print(check_reverse_encompassment([1, 3, 5, 2, 4], [4, 2, 5, 1, 3])) # True
print(check_reverse_encompassment([], []))                         # True (empty lists are considered encompassing)
print(check_reverse_encompassment([1, 2, 3], []))                   # True (target encompassing empty list is always True)

try:
  check_reverse_encompassment(1, [1, 2, 3])
except TypeError as e:
  print(f"Error: {e}")  # Output: Error: Both 'target' and 'competition' must be lists.