import Foundation

class BuddySystem {
    private var memorySize: Int
    private var minBlockSize: Int
    private var maxOrder: Int
    private var freeLists: [[Int]]
    
    init(memorySize: Int, minBlockSize: Int) {
        self.memorySize = memorySize
        self.minBlockSize = minBlockSize
        self.maxOrder = Int(log2(Double(memorySize / minBlockSize)))
        self.freeLists = Array(repeating: [], count: maxOrder + 1)
        
        // Initialize with the entire memory as one free block
        freeLists[maxOrder].append(0)
    }
    
    func allocate(size: Int) -> Int? {
        let order = max(Int(ceil(log2(Double(size) / Double(minBlockSize)))), 0)
        
        if order > maxOrder {
            return nil // Requested size is too large
        }
        
        // Find the smallest available block that fits
        for i in order...maxOrder {
            if !freeLists[i].isEmpty {
                let block = freeLists[i].removeLast()
                
                // Split larger blocks if necessary
                for j in stride(from: i - 1, through: order, by: -1) {
                    let buddy = block + (1 << j) * minBlockSize
                    freeLists[j].append(buddy)
                }
                
                return block
            }
        }
        
        return nil // No suitable block found
    }
    
    func deallocate(block: Int, size: Int) {
        var order = max(Int(ceil(log2(Double(size) / Double(minBlockSize)))), 0)
        var currentBlock = block
        
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
        
        freeLists[order].append(currentBlock)
    }
}
// Initialize a Buddy System with 1MB of memory and 4KB minimum block size
let buddySystem = BuddySystem(memorySize: 1024 * 1024, minBlockSize: 4 * 1024)

// Allocate memory
var allocatedBlocks: [(address: Int, size: Int)] = []

if let block1 = buddySystem.allocate(size: 16 * 1024) {
    print("Allocated 16KB block at address: \(block1)")
    allocatedBlocks.append((address: block1, size: 16 * 1024))
} else {
    print("Failed to allocate 16KB block")
}

if let block2 = buddySystem.allocate(size: 64 * 1024) {
    print("Allocated 64KB block at address: \(block2)")
    allocatedBlocks.append((address: block2, size: 64 * 1024))
} else {
    print("Failed to allocate 64KB block")
}

// Deallocate the first block (if it was allocated)
if let firstBlock = allocatedBlocks.first {
    buddySystem.deallocate(block: firstBlock.address, size: firstBlock.size)
    print("Deallocated \(firstBlock.size / 1024)KB block at address: \(firstBlock.address)")
    allocatedBlocks.removeFirst()
}

// Try to allocate a large block
if let largeBlock = buddySystem.allocate(size: 512 * 1024) {
    print("Allocated 512KB block at address: \(largeBlock)")
    allocatedBlocks.append((address: largeBlock, size: 512 * 1024))
} else {
    print("Failed to allocate 512KB block")
}

// Deallocate all remaining blocks
for block in allocatedBlocks {
    buddySystem.deallocate(block: block.address, size: block.size)
    print("Deallocated \(block.size / 1024)KB block at address: \(block.address)")
}