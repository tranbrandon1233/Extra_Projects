class Matrix:
    """
    Represents a 2D matrix and provides methods for matrix operations.
    """

    def __init__(self, matrix):
        """
        Initializes a Matrix object.

        Args:
            matrix (list of lists): A 2D list of integers representing the matrix.

        Raises:
            ValueError: If the input is not a valid rectangular 2D list.
        """
        if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
            raise ValueError("Input must be a 2D list.")
        
        if not matrix:  # Check for empty matrix
            self.matrix = []
            return

        row_len = len(matrix[0])
        if not all(len(row) == row_len for row in matrix):
            raise ValueError("Matrix must be rectangular (rows must have the same length).")
        
        self.matrix = matrix

    def transpose(self):
        """
        Returns a new Matrix object that is the transpose of the current matrix.

        Returns:
            Matrix: The transposed matrix.
        """
        transposed = [[self.matrix[j][i] for j in range(len(self.matrix))] for i in range(len(self.matrix[0]))]
        return Matrix(transposed)

    def determinant(self):
        """
        Computes the determinant of a 2x2 matrix.

        Returns:
            int: The determinant of the matrix.

        Raises:
            NotImplementedError: If the matrix is not 2x2.
        """
        if len(self.matrix) != 2 or len(self.matrix[0]) != 2:
            raise NotImplementedError("Determinant calculation is only implemented for 2x2 matrices.")
        
        return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]

    def __str__(self):
        """
        Returns a string representation of the matrix.

        Returns:
            str: The matrix as a string with rows on separate lines.
        """
        return '\n'.join([' '.join(map(str, row)) for row in self.matrix])

# Example Usage
try:
    matrix = Matrix([[1, 2], [3, 4]])
    print("Original Matrix:")
    print(matrix)

    transposed_matrix = matrix.transpose()
    print("\nTransposed Matrix:")
    print(transposed_matrix)

    det = matrix.determinant()
    print("\nDeterminant:", det)
    
    print("\nMatrix as a string:")
    print(str(matrix))

    matrix_3x3 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix_3x3.determinant()  # This will raise NotImplementedError
    
    matrix_0x0 = Matrix([[]])
    matrix_0x0.determinant()  # This will raise NotImplementedError

    matrix_2x3 = Matrix([ [4, 5, 6], [7, 8, 9]])
    matrix_2x3.determinant()  # This will raise NotImplementedError
except ValueError as e:
    print(f"Error: {e}")
except NotImplementedError as e:
    print(f"Error: {e}")