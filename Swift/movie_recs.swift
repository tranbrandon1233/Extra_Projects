import Foundation

struct Movie {
    let title: String
    let rating: Double
    let numberOfRatings: Int
}

struct PriorityQueue<T> {
    private var heap: [T] = []
    private let comparator: (T, T) -> Bool

    init(comparator: @escaping (T, T) -> Bool) {
        self.comparator = comparator
    }

    mutating func enqueue(_ element: T) {
        heap.append(element)
        siftUp(from: heap.count - 1)
    }

    mutating func dequeue() -> T? {
        guard !heap.isEmpty else { return nil }
        heap.swapAt(0, heap.count - 1)
        let item = heap.removeLast()
        siftDown(from: 0)
        return item
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2
        while child > 0 && comparator(heap[child], heap[parent]) {
            heap.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let leftChild = 2 * parent + 1
            let rightChild = 2 * parent + 2
            var candidate = parent

            if leftChild < heap.count && comparator(heap[leftChild], heap[candidate]) {
                candidate = leftChild
            }
            if rightChild < heap.count && comparator(heap[rightChild], heap[candidate]) {
                candidate = rightChild
            }
            if candidate == parent {
                return
            }
            heap.swapAt(parent, candidate)
            parent = candidate
        }
    }

    var count: Int {
        return heap.count
    }
}
func topRatedMovies(_ movies: [Movie], n: Int) -> [Movie] {
    var pq = PriorityQueue<Movie> { (m1, m2) -> Bool in
        if m1.rating == m2.rating {
            return m1.numberOfRatings > m2.numberOfRatings
        }
        return m1.rating > m2.rating
    }

    for movie in movies {
        if pq.count < n {
            pq.enqueue(movie)
        } else if let lowestRated = pq.dequeue() {
            if movie.rating > lowestRated.rating || 
               (movie.rating == lowestRated.rating && movie.numberOfRatings > lowestRated.numberOfRatings) {
                pq.enqueue(movie)
            } else {
                pq.enqueue(lowestRated)
            }
        }
    }

    var result: [Movie] = []
    while let movie = pq.dequeue() {
        result.insert(movie, at: 0)
    }
    return result
}

// Example usage:
let movies = [
    Movie(title: "Movie A", rating: 8.5, numberOfRatings: 1000),
    Movie(title: "Movie B", rating: 9.0, numberOfRatings: 500),
    Movie(title: "Movie C", rating: 8.5, numberOfRatings: 1500),
    Movie(title: "Movie D", rating: 7.5, numberOfRatings: 2000)
]

let topMovies = topRatedMovies(movies, n: 1)
for movie in topMovies {
    print("\(movie.title): \(movie.rating) (\(movie.numberOfRatings) ratings)")
}