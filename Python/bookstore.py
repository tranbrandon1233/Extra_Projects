import unittest

class Bookstore:
    def __init__(self):
        """
        Initializes an empty inventory.
        """
        self.inventory = []

    def add_book(self, title, author):
        """
        Adds a book to the inventory if it doesn't already exist.
        """
        for book in self.inventory:
            if book['title'] == title:
                return "Book already exists in the inventory."
        self.inventory.append({"title": title, "author": author})
        return f"'{title}' by {author} added to inventory."

    def remove_book(self, title, author):
        """
        Removes a book from the inventory if it exists.
        """
        for book in self.inventory:
            if book['title'] == title and book['author'] == author:
                self.inventory.remove(book)
                return f"'{title}' by {author} removed from inventory."
        return "Book not found in inventory."

    def search_by_author(self, author):
        """
        Searches for books by a specific author.
        """
        return [book for book in self.inventory if book['author'] == author]

class TestBookstore(unittest.TestCase):
    def setUp(self):
        """
        Initializes a new Bookstore instance before each test.
        """
        self.bookstore = Bookstore()

    def test_add_book(self):
        """
        Tests adding a new book to the inventory.
        """
        result = self.bookstore.add_book("1984", "George Orwell")
        self.assertEqual(result, "'1984' by George Orwell added to inventory.")
        self.assertIn({"title": "1984", "author": "George Orwell"}, self.bookstore.inventory)

    def test_add_duplicate_book(self):
        """
        Tests that the same book cannot be added twice to the inventory.
        """
        self.bookstore.add_book("1984", "George Orwell")
        result = self.bookstore.add_book("1984", "George Orwell")
        self.assertEqual(result, "Book already exists in the inventory.")
        self.assertEqual(len(self.bookstore.inventory), 1)

    def test_remove_book(self):
        """
        Tests removing a book from the inventory.
        """
        self.bookstore.add_book("1984", "George Orwell")
        result = self.bookstore.remove_book("1984", "George Orwell")
        self.assertEqual(result, "'1984' by George Orwell removed from inventory.")
        self.assertNotIn({"title": "1984", "author": "George Orwell"}, self.bookstore.inventory)

    def test_remove_non_existing_book(self):
        """
        Tests trying to remove a book that doesn't exist in the inventory.
        """
        result = self.bookstore.remove_book("1984", "George Orwell")
        self.assertEqual(result, "Book not found in inventory.")

    def test_search_by_author(self):
        """
        Tests searching for books by a specific author.
        """
        self.bookstore.add_book("1984", "George Orwell")
        self.bookstore.add_book("Animal Farm", "George Orwell")
        self.bookstore.add_book("Brave New World", "Aldous Huxley")

        search_result = self.bookstore.search_by_author("George Orwell")
        self.assertEqual(len(search_result), 2)
        self.assertIn({"title": "1984", "author": "George Orwell"}, search_result)
        self.assertIn({"title": "Animal Farm", "author": "George Orwell"}, search_result)

    def test_search_no_books_by_author(self):
        """
        Tests searching for books by an author who doesn't have any books in the inventory.
        """
        search_result =self.bookstore.search_by_author("J.K. Rowling")
        self.assertEqual(len(search_result), 0)

if __name__ == '__main__':
    unittest.main()