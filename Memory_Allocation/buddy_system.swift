import Foundation

/// Class representing the Buddy System memory allocator.
class BuddySystem {
    private var memorySize: Int
    private var minBlockSize: Int
    private var maxOrder: Int
    private var freeLists: [[Int]]
    
    /** Initializes the Buddy System with the given memory size and minimum block size.
     - Parameters:
       - memorySize: The total size of the memory to be managed.
       - minBlockSize: The minimum size of a block that can be allocated.
    */
    init(memorySize: Int, minBlockSize: Int) {
        self.memorySize = memorySize
        self.minBlockSize = minBlockSize
        self.maxOrder = Int(log2(Double(memorySize / minBlockSize)))
        self.freeLists = Array(repeating: [], count: maxOrder + 1)
        
        // Initialize the largest block as free.
        freeLists[maxOrder].append(0)
    }
    
    /** Allocates a block of memory of the given size.
     - Parameter size: The size of the memory block to allocate.
     - Returns: The starting address of the allocated block, or nil if allocation fails.
     */
    func allocate(size: Int) -> Int? {
        // Determine the order of the block to allocate.
        let order = max(Int(ceil(log2(Double(size) / Double(minBlockSize)))), 0)
        
        // If the requested size is too large, return nil.
        if order > maxOrder {
            return nil 
        }
        
        // Find the smallest available block that can satisfy the request.
        for i in order...maxOrder {
            if !freeLists[i].isEmpty {
                let block = freeLists[i].removeLast()
                
                // Split the block into smaller blocks until the desired size is reached.
                for j in stride(from: i - 1, through: order, by: -1) {
                    let buddy = block + (1 << j) * minBlockSize
                    freeLists[j].append(buddy)
                }
                
                return block
            }
        }
        
        // If no suitable block is found, return nil.
        return nil 
    }
    
    /** Deallocates a previously allocated block of memory.
     - Parameters:
       - block: The starting address of the block to deallocate.
       - size: The size of the block to deallocate.
    */
    func deallocate(block: Int, size: Int) {
        var order = max(Int(ceil(log2(Double(size) / Double(minBlockSize)))), 0)
        var currentBlock = block
        
        // Merge the block with its buddy if possible.
        while order <= maxOrder {
            let buddy = currentBlock ^ (1 << order) * minBlockSize
            
            if freeLists[order].contains(buddy) {
                freeLists[order].removeAll { $0 == buddy }
                currentBlock = min(currentBlock, buddy)
                order += 1
            } else {
                break
            }
        }
        
        // Add the merged block to the free list.
        freeLists[order].append(currentBlock)
    }
}

let buddySystem = BuddySystem(memorySize: 1024 * 1024, minBlockSize: 4 * 1024)

var allocatedBlocks: [(address: Int, size: Int)] = []

// Allocate a 16KB block.
if let block1 = buddySystem.allocate(size: 16 * 1024) {
    print("Allocated 16KB block at address: \(block1)")
    allocatedBlocks.append((address: block1, size: 16 * 1024))
} else {
    print("Failed to allocate 16KB block")
}

// Allocate a 64KB block.
if let block2 = buddySystem.allocate(size: 64 * 1024) {
    print("Allocated 64KB block at address: \(block2)")
    allocatedBlocks.append((address: block2, size: 64 * 1024))
} else {
    print("Failed to allocate 64KB block")
}

// Deallocate the first allocated block.
if let firstBlock = allocatedBlocks.first {
    buddySystem.deallocate(block: firstBlock.address, size: firstBlock.size)
    print("Deallocated \(firstBlock.size / 1024)KB block at address: \(firstBlock.address)")
    allocatedBlocks.removeFirst()
}

// Allocate a 512KB block.
if let largeBlock = buddySystem.allocate(size: 512 * 1024) {
    print("Allocated 512KB block at address: \(largeBlock)")
    allocatedBlocks.append((address: largeBlock, size: 512 * 1024))
} else {
    print("Failed to allocate 512KB block")
}

// Deallocate all remaining allocated blocks.
for block in allocatedBlocks {
    buddySystem.deallocate(block: block.address, size: block.size)
    print("Deallocated \(block.size / 1024)KB block at address: \(block.address)")
}