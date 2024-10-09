class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def invert_tree(node):
    if node is None:
        return None

    # Invert left and right subtrees
    left = invert_tree(node.left)
    right = invert_tree(node.right)

    # Swap the left and right children
    node.left = right
    node.right = left

    return node


def print_tree(node):
  """
  Prints the binary tree in a pre-order traversal.

  Args:
    node: The root node of the binary tree.
  """

  if node:
    print(node.data, end=" ")
    print_tree(node.left)
    print_tree(node.right)

# Example usage:
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)

print("Original tree:")
# (Assuming you have a function to print the tree)
print_tree(root)

inverted_root = invert_tree(root)

print("\nInverted tree:")
print_tree(inverted_root)