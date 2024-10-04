def quick_sort(arr, low, high):
  """
  In-place quicksort implementation in Python.

  Args:
      arr: The array to be sorted.
      low:  The starting index of the subarray.
      high: The ending index of the subarray.
  """
  if low < high:
    # Partition the array around a pivot
    pi = partition(arr, low, high)

    # Recursively sort the subarrays before and after the pivot
    quick_sort(arr, low, pi - 1)  # Sort left side
    quick_sort(arr, pi + 1, high)  # Sort right side


def partition(arr, low, high):
  """
  Partitions an array around a pivot (using the last element).

  Args:
      arr: The array to partition.
      low: The starting index.
      high: The ending index.

  Returns:
      The index of the pivot after partitioning.
  """
  pivot = arr[high]  # Choose the last element as pivot (can be chosen differently)
  i = low - 1        # Initialize index of smaller element

  for j in range(low, high):
    # If current element is smaller than or equal to pivot
    if arr[j] <= pivot:
      # Increment i, and swap arr[i] with arr[j]
      i = i + 1
      arr[i], arr[j] = arr[j], arr[i]

  # Swap the pivot element with the greater element (arr[i])
  arr[i + 1], arr[high] = arr[high], arr[i + 1]
  return i + 1  # Return the position from where partition is done


# Example usage:
data = [12, 9, 15, 2, 11, 18, 7,-9]
quick_sort(data, 0, len(data) - 1)
print("Sorted array:", data)