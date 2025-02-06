def count_pairs(nums, target):
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1

    count = 0
    for num in freq:
        complement = target - num

        if complement in freq:
            if complement == num:
                count += freq[num] * (freq[num] - 1) // 2  
            elif complement > num:  
                count += freq[num] * freq[complement]

    return count

# Test cases
def test_count_pairs():
    # Test Case 1: Empty array -> no pairs exist.
    nums = []
    target = 5
    expected = 0
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 2: Single element -> no pairs can be formed.
    nums = [1]
    target = 2
    expected = 0
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 3: Two elements that form a valid pair.
    nums = [1, 1]
    target = 2
    expected = 1  # Only one pair: (0, 1)
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 4: Multiple pairs with distinct indices.
    nums = [1, 2, 3, 4, 3]
    target = 6
    # Valid pairs:
    # (1, 3) -> 2 + 4 = 6
    # (2, 4) -> 3 + 3 = 6
    expected = 2
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 5: All identical numbers where every pair sums to target.
    nums = [3, 3, 3, 3]
    target = 6
    # All unique index pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) -> 6 pairs.
    expected = 6
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 6: Negative numbers in the list.
    nums = [-1, -2, 3, 4, 0]
    target = 2
    # Valid pairs:
    # (0, 2) -> -1 + 3 = 2
    # (1, 3) -> -2 + 4 = 2
    expected = 2
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 7: Repeated values leading to multiple distinct pairs.
    nums = [5, 5, 5, 5]
    target = 10
    # Unique pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) -> 6 pairs.
    expected = 6
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 8: Ascending numbers where no two numbers sum to target.
    nums = [1, 2, 3, 4, 5]
    target = 10
    expected = 0
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    # Test Case 9: Ascending numbers with one valid pair.
    nums = [1, 2, 3, 4, 5]
    target = 9
    # Valid pair: (3, 4) -> 4 + 5 = 9
    expected = 1
    assert count_pairs(nums, target) == expected, f"Failed on {nums} with target {target}"

    print("All test cases passed.")

# Run the tests
test_count_pairs()