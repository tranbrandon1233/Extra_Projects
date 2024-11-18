import Foundation

class AirTrafficController {
    func manageRunwayAccess(_ planes: [(fuelLevel: Int, urgency: Int)], _ runwayAvailable: Bool) -> Int {
        var runwayQueue = PriorityQueue<(Int, Int)>(comparator: { (a, b) -> Bool in
            if a.1 == b.1 { return a.0 < b.0 } // Prioritize by fuel level if urgency levels are equal
            return a.1 < b.1 // Prioritize by urgency level
        })
        
        // Add planes to the priority queue based on urgency and fuel levels
        for plane in planes {
            runwayQueue.add((plane.fuelLevel, plane.urgency))
        }
        
        var elapsedTime = 0
        while !runwayQueue.isEmpty {
            if runwayAvailable {
                let (fuelLevel, urgency) = runwayQueue.poll()!
                elapsedTime += calculateRunwayTime(fuelLevel, urgency)
            } else {
                // Increment time if runway is not available
                elapsedTime += 1
            }
        }
        
        return elapsedTime
    }
    
    private func calculateRunwayTime(_ fuelLevel: Int, _ urgency: Int) -> Int {
        // Hypothetical function to calculate time for each plane
        return max(fuelLevel - urgency, 1)
    }
}

// PriorityQueue wrapper
struct PriorityQueue<Element> {
    private var heap: [Element]
    private let comparator: (Element, Element) -> Bool

    init(comparator: @escaping (Element, Element) -> Bool) {
        self.heap = []
        self.comparator = comparator
    }

    var isEmpty: Bool {
        return heap.isEmpty
    }

    mutating func add(_ element: Element) {
        heap.append(element)
        swim(heap.count - 1)
    }

    mutating func poll() -> Element? {
        if heap.isEmpty { return nil }
        if heap.count == 1 { return heap.removeFirst() }

        heap.swapAt(0, heap.count - 1)
        let element = heap.removeLast()
        sink(0)

        return element
    }

    private mutating func swim(_ index: Int) {
        var childIndex = index
        var parentIndex = (childIndex - 1) / 2

        while childIndex > 0 && comparator(heap[childIndex], heap[parentIndex]) {
            heap.swapAt(childIndex, parentIndex)
            childIndex = parentIndex
            parentIndex = (childIndex - 1) / 2
        }
    }

    private mutating func sink(_ index: Int) {
        var parentIndex = index

        while true {
            let leftChildIndex = 2 * parentIndex + 1
            let rightChildIndex = 2 * parentIndex + 2
            var candidateIndex = parentIndex

            if leftChildIndex < heap.count && comparator(heap[leftChildIndex], heap[candidateIndex]) {
                candidateIndex = leftChildIndex
            }
            if rightChildIndex < heap.count && comparator(heap[rightChildIndex], heap[candidateIndex]) {
                candidateIndex = rightChildIndex
            }

            if candidateIndex == parentIndex {
                return
            }

            heap.swapAt(parentIndex, candidateIndex)
            parentIndex = candidateIndex
        }
    }
}