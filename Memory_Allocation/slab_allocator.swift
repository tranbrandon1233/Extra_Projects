
import Foundation

class SlabAllocator {
    private var slabs: [Slab]
    private let slabSize: Int
    private let objectSize: Int
    
    init(slabSize: Int, objectSize: Int) {
        self.slabSize = slabSize
        self.objectSize = objectSize
        self.slabs = []
    }
    
    func allocate() -> Int? {
        for slab in slabs {
            if let address = slab.allocate() {
                return address
            }
        }
        
        // No free objects, create a new slab
        let newSlab = Slab(size: slabSize, objectSize: objectSize)
        slabs.append(newSlab)
        return newSlab.allocate()
    }
    
    func deallocate(address: Int) {
        for slab in slabs {
            if slab.containsAddress(address) {
                slab.deallocate(address: address)
                
                // Remove empty slabs to save memory
                slabs.removeAll { $0.isEmpty }
                return
            }
        }
    }
}

class Slab {
    private var memory: UnsafeMutableRawPointer
    private var bitmap: [Bool]
    private let objectSize: Int
    private let objectCount: Int
    
    init(size: Int, objectSize: Int) {
        self.memory = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: MemoryLayout<Int>.alignment)
        self.objectSize = objectSize
        self.objectCount = size / objectSize
        self.bitmap = Array(repeating: false, count: objectCount)
    }
    
    deinit {
        memory.deallocate()
    }
    
    func allocate() -> Int? {
        if let index = bitmap.firstIndex(of: false) {
            bitmap[index] = true
            return Int(bitPattern: memory) + index * objectSize
        }
        return nil
    }
    
    func deallocate(address: Int) {
        let index = (address - Int(bitPattern: memory)) / objectSize
        bitmap[index] = false
    }
    
    func containsAddress(_ address: Int) -> Bool {
        let start = Int(bitPattern: memory)
        let end = start + objectCount * objectSize
        return (start..<end).contains(address)
    }
    
    var isEmpty: Bool {
        return bitmap.allSatisfy { !$0 }
    }
}

// Initialize a Slab Allocator with 1MB slabs for 256-byte objects
let slabAllocator = SlabAllocator(slabSize: 1024 * 1024, objectSize: 256)

// Allocate objects
var allocatedAddresses: [Int] = []

for _ in 1...5000 {
    if let address = slabAllocator.allocate() {
        allocatedAddresses.append(address)
    }
}

print("Allocated \(allocatedAddresses.count) objects")

// Deallocate half of the objects
for i in 0..<2500 {
    slabAllocator.deallocate(address: allocatedAddresses[i])
}

print("Deallocated 2500 objects")

// Allocate more objects
var newAllocations = 0
for _ in 1...3000 {
    if slabAllocator.allocate() != nil {
        newAllocations += 1
    }
}

print("Allocated \(newAllocations) new objects")