/**
 * TreeNode class represents each node in the binary tree.
 * @param {number} val - The value of the node.
 * @param {TreeNode|null} left - The left child of the node.
 * @param {TreeNode|null} right - The right child of the node.
 */
class TreeNode {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
}


/**
 * Inserts a value into a Binary Search Tree (BST).
 * @param {TreeNode} root - The root node of the BST.
 * @param {number} val - The value to be inserted into the BST.
 * @returns {TreeNode} - The root node of the BST after insertion.
 */
function insertIntoBST(root, val) {
    if (val === -1) return root; // Skip null nodes

    if (root === null) {
        return new TreeNode(val); // Create a new node if root is null
    }

    if (val < root.val) {
        root.left = insertIntoBST(root.left, val); // Insert in the left subtree
    } else if (val > root.val) {
        root.right = insertIntoBST(root.right, val); // Insert in the right subtree
    }

    return root; // Return the root node
}

/**
 * Converts an array into a Binary Search Tree (BST).
 * @param {number[]} arr - The array of values to be converted into a BST.
 * @returns {TreeNode} - The root node of the resulting BST.
 */
function arrayToBinaryTree(arr) {
    let root = null;
    for (let val of arr) {
        root = insertIntoBST(root, val); // Insert each value into the BST
    }
    return root; // Return the root node of the BST
}


/**
 * Traverses the binary tree, adding the root and then all the nodes on the right.
 * If no right node exists, the function adds the left node before continuing on the right.
 * @param {TreeNode} root - The root of the binary tree.
 * @returns {number[]} - An array of node values following the right-first traversal rules.
 */
function rightSideView(arr) {
    arr = arrayToBinaryTree(arr)
    let result = [];

    /**
     * Helper function to traverse the tree recursively.
     * @param {TreeNode} node - The current node being processed.
     */
    function traverse(node) {
        if (!node) return;

        // Add the current node's value to the result array
        result.push(node.val);

        // Traverse right first, then left
        if (node.right) {
            traverse(node.right);
        } else if (node.left) {
            traverse(node.left);
        }
    }

    // Start traversal from the root
    traverse(arr);

    return result;
}

// Example usage:
console.log(rightSideView([3, 2, 9, 7, -1, -1, 4]));  // Output: [3, 9, 7, 4]

console.log(rightSideView([1, -1, 2, -1, 3, -1, 5, -1, 4])); // Output: [1, 2, 3, 3, 4]
console.log(rightSideView([1, -1, 3])); // Output: [1, 3]

console.log(rightSideView([3]));  // Output: [3]

console.log(rightSideView([])); // Output: []