class SpectralBloomFilter  {
  constructor(size) {
    this.size = size;
    this.bitArray = new Array(size).fill(0);
    this.hashFunctions = [
      (key) => this.djb2Hash(key),
      (key) => this.sdbmHash(key),
    ];
    this.numElements = 0; // Initialize the numElements property
  }

  // DJB2 hash function
  djb2Hash(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) + hash + str.charCodeAt(i);
    }
    return hash % this.size;
  }

  // SDBM hash function
  sdbmHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + (hash << 6) + (hash << 16) - hash;
    }
    return hash % this.size;
  }

  // Add an element to the filter with a count
  add(item, count = 1) {
    this.hashFunctions.forEach((hashFunction) => {
      const position = hashFunction(item);
      this.bitArray[position] = Math.min(this.bitArray[position] + count, 255);
    });
    this.numElements += count;
  }

  // Check the count of an element in the filter
  check(item) {
    return Math.min(...this.hashFunctions.map((hashFunction) => {
      const position = hashFunction(item);
      return this.bitArray[position];
    })) / this.hashFunctions.length;
  }

  // Remove a count of an element from the filter
  remove(item, count = 1) {
    this.hashFunctions.forEach((hashFunction) => {
      const position = hashFunction(item);
      this.bitArray[position] = Math.max(this.bitArray[position] - count, 0);
    });
    this.numElements -= count;
  }

  // Calculate the probability of a false positive
  falsePositiveProbability() {
    const n = this.numElements;
    const m = this.size;
    const k = this.hashFunctions.length;
    return Math.pow(1 - Math.exp(-k * n / m), k);
  }
}

// Create a new Bloom filter with 100 bits and 3 hash functions
const filter = new SpectralBloomFilter(100);

// Add some elements to the filter
filter.add('apple');
filter.add('banana');
filter.add('orange');

console.log(filter.check('apple')); // Output: 1
console.log(filter.check('test')); // Output: 0
filter.remove('apple')

console.log(filter.check("apple")); // Output: 0