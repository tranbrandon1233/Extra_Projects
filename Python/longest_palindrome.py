def longest_palindrome(s: str) -> str:
    if not s:
        return ""
    max_pal = ""
    for i in range(len(s)):
        for j in range(i, len(s) + 1):
            current_substring = s[i:j]
            if current_substring == current_substring[::-1] and len(current_substring) > len(max_pal):
                max_pal = current_substring
    return max_pal


# --- Simple assertion-based tests ---
def run_simple_tests():
    # 1. Empty string should return an empty string.
    assert longest_palindrome("") == "", "Failed on empty string."

    # 2. A single character is a palindrome by itself.
    assert longest_palindrome("a") == "a", "Failed on single character."

    # 3. When no palindrome longer than one character exists, a one-letter string is returned.
    #    For "abc", any single letter is acceptable.
    result = longest_palindrome("abc")
    assert len(result) == 1 and result in "abc", "Failed on string with no longer palindrome."

    # 4. Odd-length palindrome.
    #    For "babad", acceptable answers are "bab" or "aba".
    result = longest_palindrome("babad")
    assert result in ("bab", "aba"), f"Failed on odd-length palindrome. Got: {result}"

    # 5. Even-length palindrome.
    assert longest_palindrome("cbbd") == "bb", "Failed on even-length palindrome."

    # 6. When the entire string is a palindrome.
    assert longest_palindrome("abba") == "abba", "Failed on full string palindrome."

    # 7. All characters same.
    assert longest_palindrome("aaaa") == "aaaa", "Failed on all same characters."

    # 8. A more complex case.
    assert longest_palindrome("forgeeksskeegfor") == "geeksskeeg", "Failed on complex palindrome."

    print("All simple assertion tests passed.")


# --- Unit tests using unittest ---
import unittest

class TestLongestPalindrome(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(longest_palindrome(""), "")

    def test_single_character(self):
        self.assertEqual(longest_palindrome("a"), "a")

    def test_no_longer_palindrome(self):
        result = longest_palindrome("abc")
        self.assertEqual(len(result), 1)
        self.assertIn(result, "abc")

    def test_odd_length_palindrome(self):
        result = longest_palindrome("babad")
        self.assertIn(result, ("bab", "aba"))

    def test_even_length_palindrome(self):
        self.assertEqual(longest_palindrome("cbbd"), "bb")

    def test_full_string_palindrome(self):
        self.assertEqual(longest_palindrome("abba"), "abba")

    def test_all_same_characters(self):
        self.assertEqual(longest_palindrome("aaaa"), "aaaa")

    def test_complex_case(self):
        self.assertEqual(longest_palindrome("forgeeksskeegfor"), "geeksskeeg")


if __name__ == "__main__":
    # Run simple assertion-based tests.
    run_simple_tests()

    # Run unittest test cases.
    unittest.main(exit=False)
