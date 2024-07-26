import Foundation

// Cryptography Helper
// This struct provides a simple hashing function for strings.
struct CryptoHelper {
    static func simpleHash(_ string: String) -> String {
        var hash = 0
        for char in string.unicodeScalars {
            hash = 31 &* hash &+ Int(char.value) // Calculate hash using a simple algorithm.
        }
        return String(format: "%016x", hash) // Return the hash as a hexadecimal string.
    }
}

// Transaction
// This struct represents a transaction with a sender, recipient, and amount.
struct Transaction: Codable {
    let sender: String
    let recipient: String
    let amount: Double
}

// Block
// This class represents a block in the blockchain.
class Block: Codable {
    let timestamp: Date
    let transactions: [Transaction]
    let previousHash: String
    let shardId: Int
    var hash: String
    var nonce: Int
    
    // Initialize a new block with the given parameters.
    init(timestamp: Date, transactions: [Transaction], previousHash: String, shardId: Int) {
        self.timestamp = timestamp
        self.transactions = transactions
        self.previousHash = previousHash
        self.shardId = shardId
        self.hash = ""
        self.nonce = 0
    }
    
    // Calculate the hash of the block.
    func calculateHash() -> String {
        let data = "\(timestamp.timeIntervalSince1970)\(transactions)\(previousHash)\(nonce)\(shardId)"
        return CryptoHelper.simpleHash(data)
    }
    
    // Mine the block by finding a hash that meets the difficulty criteria.
    func mine(difficulty: Int) {
        let target = String(repeating: "0", count: difficulty)
        while !hash.hasPrefix(target) {
            nonce += 1
            hash = calculateHash()
        }
    }
}

// Merkle Tree
// This class represents a Merkle tree for the transactions in a block.
class MerkleTree {
    private let transactions: [Transaction]
    
    // Initialize the Merkle tree with the given transactions.
    init(transactions: [Transaction]) {
        self.transactions = transactions
    }
    
    // Build the Merkle tree and return the root hash.
    func buildTree() -> String {
        var nodes = transactions.map { CryptoHelper.simpleHash("\($0.sender)\($0.recipient)\($0.amount)") }
        
        while nodes.count > 1 {
            var newLevel: [String] = []
            for i in stride(from: 0, to: nodes.count, by: 2) {
                if i + 1 < nodes.count {
                    let combined = nodes[i] + nodes[i + 1]
                    let hash = CryptoHelper.simpleHash(combined)
                    newLevel.append(hash)
                } else {
                    newLevel.append(nodes[i])
                }
            }
            nodes = newLevel
        }
        
        return nodes.first ?? ""
    }
}

// Blockchain
// This class represents the blockchain with multiple shards.
class Blockchain {
    private var shards: [[Block]]
    private let difficulty: Int
    private let shardCount: Int
    
    // Initialize the difficulty, shard count, and shards for the blockchain.
    init(difficulty: Int, shardCount: Int) {
        self.difficulty = difficulty
        self.shardCount = shardCount
        self.shards = Array(repeating: [], count: shardCount)
        
        // Create genesis block for each shard.
        for i in 0..<shardCount {
            let genesisBlock = Block(timestamp: Date(), transactions: [], previousHash: "0", shardId: i)
            genesisBlock.mine(difficulty: difficulty)
            shards[i].append(genesisBlock)
        }
    }
    
    // Add the transaction to the block based on the shard.
    func addTransaction(_ transaction: Transaction) {
        let shardId = abs(transaction.sender.hashValue) % shardCount
        let lastBlock = shards[shardId].last!
        let newBlock = Block(timestamp: Date(), transactions: [transaction], previousHash: lastBlock.hash, shardId: shardId)
        mineBlock(newBlock)
        shards[shardId].append(newBlock)
    }
    
    // Mine the block.
    private func mineBlock(_ block: Block) {
        block.mine(difficulty: difficulty)
    }
    
    // Check if the chain is valid.
    func isChainValid() -> Bool {
        for shard in shards {
            for i in 1..<shard.count {
                let currentBlock = shard[i]
                let previousBlock = shard[i - 1]
                
                if currentBlock.hash != currentBlock.calculateHash() {
                    return false
                }
                
                if currentBlock.previousHash != previousBlock.hash {
                    return false
                }
                
                if !currentBlock.hash.hasPrefix(String(repeating: "0", count: difficulty)) {
                    return false
                }
            }
        }
        return true
    }
    
    // Get the blockchain.
    func getBlockchain() -> [[Block]] {
        return shards
    }
}

// Usage Example

let blockchain = Blockchain(difficulty: 4, shardCount: 3)

let transaction1 = Transaction(sender: "Alice", recipient: "Bob", amount: 50)
let transaction2 = Transaction(sender: "Bob", recipient: "Charlie", amount: 30)
let transaction3 = Transaction(sender: "Charlie", recipient: "David", amount: 20)

blockchain.addTransaction(transaction1)
blockchain.addTransaction(transaction2)
blockchain.addTransaction(transaction3)

print("Blockchain valid: \(blockchain.isChainValid())")

for (index, shard) in blockchain.getBlockchain().enumerated() {
    print("Shard \(index):")
    for block in shard {
        print("  Block Hash: \(block.hash)")
        print("  Transactions: \(block.transactions)")
        print("  Merkle Root: \(MerkleTree(transactions: block.transactions).buildTree())")
        print("")
    }
}
