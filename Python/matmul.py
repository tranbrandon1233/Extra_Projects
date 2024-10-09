
def search_matrix(matrix, target):
    # Get the number of rows and columns in the matrix
    rows = len(matrix)
    if rows == 0:
        return False
    cols = len(matrix[0])

    # Define the search space for the binary search
    low, high = 0, rows * cols - 1

    # Perform binary search
    while low <= high:
        mid = (low + high) // 2
        
        # Calculate the row and column based on the mid index
        row = mid // cols
        col = mid % cols
        
        # Check if the target is found
        if matrix[row][col] == target:
            return True
        # If the target is greater, search the right half
        elif matrix[row][col] < target:
            low = mid + 1
        # If the target is smaller, search the left half
        else:
            high = mid - 1

    # Target was not found in the matrix
    return False

# Example Usage:
# Each row of the matrix is sorted in ascending order.
# The first element of each row is greater than the last element of the previous row.
matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 60]
]

target = 16
print(search_matrix(matrix, target))  # Output: True

target = 13
print(search_matrix(matrix, target))  # Output: False
