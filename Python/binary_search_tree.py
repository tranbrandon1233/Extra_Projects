import random
import time

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def insert(root, data):
    if root is None:
        return Node(data)
    else:
        if data < root.data:
            root.left = insert(root.left, data)
        else:
            root.right = insert(root.right, data)
        return root

def search(root, key):
    if root is None or root.data == key:
        return root
    if key < root.data:
        return search(root.left, key)
    return search(root.right, key)

# Generate 1,000,000 unique numbers
numbers = random.sample(range(1, 3000000), 1000000) 

# Create the tree
root = None
start_time = time.time()
for num in numbers:
    root = insert(root, num)
end_time = time.time()
print(f"Tree creation time: {end_time - start_time:.4f} seconds")

# Get search key from the user
search_key = int(input("Enter a number to search: "))

# Search for the key
start_time = time.time()
result = search(root, search_key)
end_time = time.time()

if result:
    print(f"Number {search_key} found in the tree.")
else:
    print(f"Number {search_key} not found in the tree.")
print(f"Search time: {end_time - start_time:.6f} seconds")