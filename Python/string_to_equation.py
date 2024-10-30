import unittest
def calculate(expression):
    """
    Evaluates a mathematical expression string with support for addition, subtraction,
    multiplication, division, parentheses, and operator precedence.

    Args:
        expression: A string representing the mathematical expression.

    Returns:
        The result of the evaluated expression.

    Raises:
        ValueError: If the expression is invalid (e.g., misplaced operators or parentheses).
    """
    if not expression:
        return ""

    def find_closing_parenthesis(expr, start_index):
        """Helper function to find the matching closing parenthesis."""
        count = 1
        for i in range(start_index + 1, len(expr)):
            if expr[i] == '(':
                count += 1
            elif expr[i] == ')':
                count -= 1
                if count == 0:
                    return i
        raise ValueError("Mismatched parentheses")

    def evaluate_expression(expr):
        """Evaluates the expression considering operator precedence."""
        nums = []
        ops = []
        i = 0

        def compute():
            if not nums or not ops:
                return
            num2 = nums.pop()
            num1 = nums.pop() if nums else 0  # Handle unary minus
            op = ops.pop()
            if op == '+':
                nums.append(num1 + num2)
            elif op == '-':
                nums.append(num1 - num2)
            elif op == '*':
                nums.append(num1 * num2)
            elif op == '/':
                if num2 == 0:
                    raise ValueError("Division by zero")
                nums.append(num1 / num2)  # Use regular division, not integer division

        while i < len(expr):
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue

            # Handle numbers (including negative numbers)
            if char.isdigit() or char == '.' or (char == '-' and 
                (i == 0 or expr[i-1] in '(+-*/' or expr[i-1].isspace())):
                start = i
                if char == '-':
                    i += 1
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                try:
                    num = float(expr[start:i])
                    nums.append(num)
                except ValueError:
                    raise ValueError(f"Invalid number format: {expr[start:i]}")
                continue
            
            # Handle parentheses
            elif char == '(':
                end = find_closing_parenthesis(expr, i)
                nums.append(evaluate_expression(expr[i + 1:end]))
                i = end + 1
                continue
                
            # Handle operators
            elif char in '+-*/':
                while (ops and ((char in '+-' and ops[-1] in '+-*/') or 
                              (char in '*/' and ops[-1] in '*/'))): 
                    compute()
                ops.append(char)
                i += 1
            else:
                raise ValueError(f"Invalid character in expression: {char}")

        # Process remaining operations
        while ops:
            compute()

        if not nums:
            return 0
        return nums[0]

    # Handle simple numeric input (including negative numbers)
    try:
        return float(expression)
    except ValueError:
        return evaluate_expression(expression)

class TestCalculator(unittest.TestCase):
    def test_basic_operations(self):
        """Test basic arithmetic operations"""
        test_cases = [
            ("2+2", 4.0),
            ("3-1", 2.0),
            ("4*5", 20.0),
            ("15/3", 5.0),
            ("2.5+1.5", 4.0),
            ("3.3*2", 6.6),
        ]
        for expression, expected in test_cases:
            with self.subTest(expression=expression):
                self.assertAlmostEqual(calculate(expression), expected, places=7)

    def test_operator_precedence(self):
        """Test operator precedence (PEMDAS)"""
        test_cases = [
            ("2+3*4", 14.0),
            ("10-2*3", 4.0),
            ("20/5+3", 7.0),
            ("2+3*4-1", 13.0),
            ("10/2*5", 25.0),
            ("2*3+4*5", 26.0),
        ]
        for expression, expected in test_cases:
            with self.subTest(expression=expression):
                self.assertAlmostEqual(calculate(expression), expected, places=7)

    def test_parentheses(self):
        """Test expressions with parentheses"""
        test_cases = [
            ("(2+3)*4", 20.0),
            ("2*(3+4)", 14.0),
            ("(10-2)*(3+1)", 32.0),
            ("((2+3)*4)", 20.0),
            ("(2+3)*(4+5)", 45.0),
            ("(10/(2+3))*4", 8.0),
        ]
        for expression, expected in test_cases:
            with self.subTest(expression=expression):
                self.assertAlmostEqual(calculate(expression), expected, places=7)

    def test_nested_parentheses(self):
        """Test expressions with nested parentheses"""
        test_cases = [
            ("((2+3)*4)+1", 21.0),
            ("2*((3+4)*2)", 28.0),
            ("(2+(3*(4+1)))", 17.0),
            ("((10/2)+(3*4))", 17.0),
            ("(2*(3+(4*(5+1))))", 54.0),
        ]
        for expression, expected in test_cases:
            with self.subTest(expression=expression):
                self.assertAlmostEqual(calculate(expression), expected, places=7)

    def test_negative_numbers(self):
        """Test expressions with negative numbers"""
        test_cases = [
            ("-2+3", 1.0),
            ("2+(-3)", -1.0),
            ("-2*(-3)", 6.0),
            ("(-2)*3", -6.0),
            ("(-10)/2", -5.0),
            ("2*((-3)+1)", -4.0),
        ]
        for expression, expected in test_cases:
            with self.subTest(expression=expression):
                self.assertAlmostEqual(calculate(expression), expected, places=7)

    def test_decimal_numbers(self):
        """Test expressions with decimal numbers"""
        test_cases = [
            ("2.5+1.5", 4.0),
            ("3.3*2.0", 6.6),
            ("10.5/2.1", 5.0),
            ("2.5*(1.5+2.0)", 8.75),
            ("0.1+0.2", 0.3),
            ("3.14159*2", 6.28318),
        ]
        for expression, expected in test_cases:
            with self.subTest(expression=expression):
                self.assertAlmostEqual(calculate(expression), expected, places=7)


if __name__ == '__main__':
    unittest.main()
