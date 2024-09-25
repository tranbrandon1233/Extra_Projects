import Foundation

/// SlabAllocator class is responsible for managing memory allocation using slab allocation technique.
class SlabAllocator {
    private var slabs: [Slab]  // List of slabs
    private let slabSize: Int  // Size of each slab
    private let objectSize: Int  // Size of each object
    
    /// Initialize a new SlabAllocator with given slab size and object size.
    /// - Parameters:
    ///   - slabSize: The size of each slab.
    ///   - objectSize: The size of each object.
    init(slabSize: Int, objectSize: Int) {
        self.slabSize = slabSize
        self.objectSize = objectSize
        self.slabs = []
    }
    
    /// Allocate memory for an object and return its address. If no free objects, create a new slab.
    /// - Returns: The address of the allocated object, or nil if allocation failed.
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
    
    /// Deallocate memory at the given address and remove empty slabs to save memory.
    /// - Parameter address: The address of the object to deallocate.
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

/// Slab class represents a slab of memory.
class Slab {
    private var memory: UnsafeMutableRawPointer  // Pointer to the slab's memory
    private var bitmap: [Bool]  // Bitmap indicating which objects are allocated
    private let objectSize: Int  // Size of each object
    private let objectCount: Int  // Number of objects in the slab
    
    /// Initialize a new Slab with given size and object size.
    /// - Parameters:
    ///   - size: The size of the slab.
    ///   - objectSize: The size of each object.
    init(size: Int, objectSize: Int) {
        self.memory = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: MemoryLayout<Int>.alignment)
        self.objectSize = objectSize
        self.objectCount = size / objectSize
        self.bitmap = Array(repeating: false, count: objectCount)
    }
    
    /// Deallocate the slab's memory when it is destroyed.
    deinit {
        memory.deallocate()
    }
    
    /// Allocate memory for an object in the slab and return its address.
    /// - Returns: The address of the allocated object, or nil if allocation failed.
    func allocate() -> Int? {
        if let index = bitmap.firstIndex(of: false) {
            bitmap[index] = true
            return Int(bitPattern: memory) + index * objectSize
        }
        return nil
    }
    
    /// Deallocate memory for an object in the slab at the given address.
    /// - Parameter address: The address of the object to deallocate.
    func deallocate(address: Int) {
        let index = (address - Int(bitPattern: memory)) / objectSize
        bitmap[index] = false
    }
    
    /// Check if the slab contains the given address.
    /// - Parameter address: The address to check.
    /// - Returns: True if the slab contains the address, false otherwise.
    func containsAddress(_ address: Int) -> Bool {
        let start = Int(bitPattern: memory)
        let end = start + objectCount * objectSize
        return (start..<end).contains(address)
    }
    
    /// Check if the slab is empty.
    /// - Returns: True if the slab is empty, false otherwise.
    var isEmpty: Bool {
        return bitmap.allSatisfy { !$0 }
    }
}

// Initialize a Slab Allocator with 1MB slabs for 256-byte objects
let slabAllocator = SlabAllocator(slabSize: 1024 * 1024, objectSize: 256)

// Allocate objects
var allocatedAddresses: [Int] = []

// Allocate 5000 objects and store their addresses
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
