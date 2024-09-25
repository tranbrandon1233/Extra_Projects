"use strict";
function unique(array) {
    return Array.from(new Set(array));
}
// Test cases
const numbers = [1, 2, 3, 2, 4, 1, 5];
const strings = ["apple", "banana", "cherry", "apple", "banana"];
const objects = [
    { id: 1, name: "John" },
    { id: 2, name: "Jane" },
    { id: 1, name: "John" },
];
// Test cases for numbers
console.log(unique(numbers)); // Output: [1, 2, 3, 4, 5]
// Test cases for strings
console.log(unique(strings)); // Output: ["apple", "banana", "cherry"]
// Test cases for objects
console.log(unique(objects)); // Output: [{ id: 1, name: "John" }, { id: 2, name: "Jane" }]
