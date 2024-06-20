const elliptic = require('elliptic');
const crypto = require('crypto');

// Define a Block class
class Block {
    constructor(index, previousHash, timestamp, data, hash) {
        this.index = index;
        this.previousHash = previousHash;
        this.timestamp = timestamp;
        this.data = data;
        this.hash = hash;
    }
}

// Define a Blockchain class
class Blockchain {
    constructor() {
        this.chain = [this.createGenesisBlock()];
        this.difficulty = 2;
        this.pendingTransactions = [];
        this.miningReward = 100;
    }

    createGenesisBlock() {
        return new Block(0, "0", Date.now(), "Genesis Block", this.calculateHash(0, "0", Date.now(), "Genesis Block"));
    }

    calculateHash(index, previousHash, timestamp, data) {
        return crypto.createHash('sha256').update(index + previousHash + timestamp + JSON.stringify(data)).digest('hex');
    }

    getLatestBlock() {
        return this.chain[this.chain.length - 1];
    }

    minePendingTransactions(miningRewardAddress) {
        let transactions = [...this.pendingTransactions];
        transactions.push({
            fromAddress: null,
            toAddress: miningRewardAddress,
            amount: this.miningReward
        });
        let block = new Block(this.chain.length, this.getLatestBlock().hash, Date.now(), transactions, this.calculateHash(this.chain.length, this.getLatestBlock().hash, Date.now(), transactions));
        block.hash = this.proofOfWork(block);
        this.chain.push(block);
        this.pendingTransactions = [];
    }

    proofOfWork(block) {
        let hash = block.hash;
        while (!hash.startsWith('0'.repeat(this.difficulty))) {
            block.timestamp++;
            hash = this.calculateHash(block.index, block.previousHash, block.timestamp, block.data);
        }
        return hash;
    }

    createTransaction(transaction) {
        this.pendingTransactions.push(transaction);
    }

    getBalanceOfAddress(address) {
        let balance = 0;
        for (const block of this.chain) {
            for (const trans of block.data) {
                if (trans.fromAddress === address) {
                    balance -= trans.amount;
                }
                if (trans.toAddress === address) {
                    balance += trans.amount;
                }
            }
        }
        return balance;
    }

    verifyRingSignature(transaction) {
        const ec = new elliptic.ec('secp256k1');
        const pubKeys = transaction.ringMembers.map((member) => ec.keyFromPublic(member, 'hex'));
        const message = transaction.id;
        const signature = transaction.signature;
        const keyImage = transaction.keyImage;

        for (let i = 0; i < pubKeys.length; i++) {
            const pubKey = pubKeys[i];
            const verify = ec.verify(message, signature, pubKey);
            if (verify) {
                const keyImageVerify = ec.verify(message, keyImage, pubKey);
                if (keyImageVerify) {
                    return true;
                }
            }
        }

        return false;
    }
}

// Define a Transaction class
class Transaction {
    constructor(id, fromAddress, toAddress, amount, ringMembers, signature, keyImage) {
        this.id = id;
        this.fromAddress = fromAddress;
        this.toAddress = toAddress;
        this.amount = amount;
        this.ringMembers = ringMembers;
        this.signature = signature;
        this.keyImage = keyImage;
    }
}

// Define a Peer class
class Peer {
    constructor(blockchain, address, privateKey) {
        this.blockchain = blockchain;
        this.address = address;
        this.privateKey = privateKey;
    }

    sendTransaction(toAddress, amount, ringMembers) {
        const ec = new elliptic.ec('secp256k1');
        const keyPair = ec.keyFromPrivate(this.privateKey, 'hex');
        const signature = keyPair.sign(toAddress + amount.toString()).toDER('hex');
        const keyImage = keyPair.sign(toAddress + amount.toString()).toDER('hex');
        const transaction = new Transaction(crypto.randomBytes(32).toString('hex'), this.address, toAddress, amount, ringMembers, signature, keyImage);
        this.blockchain.createTransaction(transaction);
    }

    mine() {
        this.blockchain.minePendingTransactions(this.address);
    }

    getBalance() {
        return this.blockchain.getBalanceOfAddress(this.address);
    }
}

// Create a new blockchain and peers
const blockchain = new Blockchain();
const peer1 = new Peer(blockchain, "peer1", crypto.randomBytes(32).toString('hex'));
const peer2 = new Peer(blockchain, "peer2", crypto.randomBytes(32).toString('hex'));

// Send transactions
peer1.sendTransaction("peer2", 10, ["peer1", "peer3", "peer4"]);
peer2.sendTransaction("peer1", 5, ["peer2", "peer5", "peer6"]);

// Mine transactions
peer1.mine();
peer2.mine();

// Print balances
console.log("Peer 1 balance: " + peer1.getBalance());
console.log("Peer 2 balance: " + peer2.getBalance());