class BloomFilter {
  constructor(size, hashFunctions) {
    this.size = size;
    this.hashFunctions = hashFunctions;
    this.bitArray = new Array(size).fill(false);
    this.numElements = 0; // Track the number of elements in the filter
  }

  // Hash function to generate indices
  _hash(str, seed) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash * seed + str.charCodeAt(i)) % this.size;
    }
    return hash;
  }

  // Add an element to the filter
  add(element) {
    for (let i = 0; i < this.hashFunctions.length; i++) {
      const index = this._hash(element, this.hashFunctions[i]);
      this.bitArray[index] = true;
    }
    this.numElements++; // Increment the number of elements
  }

  // Check if an element is possibly in the filter
  query(element) {
    for (let i = 0; i < this.hashFunctions.length; i++) {
      const index = this._hash(element, this.hashFunctions[i]);
      if (!this.bitArray[index]) {
        return false;
      }
    }
    return true;
  }

  // Simulate removing an element by creating a new filter without the element
  remove(element) {
    const newFilter = new BloomFilter(this.size, this.hashFunctions);
    for (let i = 0; i < this.bitArray.length; i++) {
      if (this.bitArray[i]) {
        let found = false;
        for (let j = 0; j < this.hashFunctions.length; j++) {
          const index = this._hash(element, this.hashFunctions[j]);
          if (i === index) {
            found = true;
            break;
          }
        }
        if (!found) {
          newFilter.bitArray[i] = true;
        }
      }
    }
    newFilter.numElements = this.numElements - 1; // Update the number of elements
    return newFilter;
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
const filter = new BloomFilter(100, [31, 37, 41]);

// Add some elements to the filter
filter.add('apple');
filter.add('banana');
filter.add('orange');

// Calculate the probability of a false positive
console.log(filter.falsePositiveProbability()); // Output the probability

// Simulate removing an element from the filter
const newFilter = filter.remove('banana');

// Calculate the probability of a false positive for the new filter
console.log(newFilter.falsePositiveProbability()); // Output the updated probability