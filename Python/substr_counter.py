# Python function to find the number of distinct substrings in a given string using suffix arrays
def count_distinct_substrings(input_string):

    # Convert the input string to lowercase to perform case-insensitive comparison
    input_string = input_string.lower()

    # Create an empty set to store distinct substrings
    distinct_substrings = set()

    # Iterate through each character in the input string
    for i in range(len(input_string)):
        for j in range(i,len(input_string)+1):

            # Create a substring starting from the current index
            substring = input_string[i:j]
            
            # Ensure the substring is not empty
            if len(substring) > 0:
                # Add the substring to the set if it has not been seen before
                distinct_substrings.add(substring)

    # Return the length of the set minus one, which represents the number of distinct substrings
    return len(distinct_substrings)

# Example usage with various test cases
examples = ["banana", "abc", "abcd", "aabbcc", "apple"]
for example in examples:
    result = count_distinct_substrings(example)
    print(f"Distinct substrings in '{example}': {result}")