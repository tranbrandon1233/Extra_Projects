import Foundation

// Define the maximum number of books a user can borrow

let maxBooksPerUser = 3

// Struct to represent a library system

struct Library {

    var inventory: [String: Int]

    var userBorrowedBooks: [String: [String]]

    /**
        Description: Function to check out a book

        Parameters: 
            - user: A `String` representing the user who is checking out the book.
            - book: A `String` representing the book that the user is checking out.
    */

    mutating func checkoutBook(user: String, book: String) {

        // Check if the book is available in the inventory

        guard let availableBooks = inventory[book], availableBooks > 0 else {

            print("Sorry, \(book) is not available.")

            return

        } 

        // Check if the user has already borrowed the maximum number of books

        var borrowedBooks = userBorrowedBooks[user] ?? []

        guard borrowedBooks.count < maxBooksPerUser else {

            print("\(user) has already borrowed the maximum number of books.")

            return

        }

        // Update inventory and user's borrowed books

        inventory[book]! -= 1

        borrowedBooks.append(book)

        userBorrowedBooks[user] = borrowedBooks

        print("\(user) checked out \(book).")

    }

}

// Example usage

var library = Library(inventory: ["Book1": 3, "Book2": 2, "Book3": 1], userBorrowedBooks: ["User1": ["Book1"], "User2": []])

library.checkoutBook(user: "User1", book: "Book2")

library.checkoutBook(user: "User3", book: "Book3")

library.checkoutBook(user: "User3", book: "Book1")

library.checkoutBook(user: "User3", book: "Book2")

library.checkoutBook(user: "User3", book: "Book2") // Should print max books error

print(library.inventory)

print(library.userBorrowedBooks)