func getPrimes(_ numbers: [Int]) -> [Int] {
    guard let max = numbers.max() else { return [] }
    
    var sieve = [Bool](repeating: true, count: max + 1)
    sieve[0] = false
    sieve[1] = false
    
    let sqrtMax = Int(Double(max).squareRoot())
    for i in 2...sqrtMax {
        if sieve[i] {
            for j in stride(from: i * i, through: max, by: i) {
                sieve[j] = false
            }
        }
    }
    
    return numbers.filter { $0 > 1 && sieve[$0] }
}

// Example usage:
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
let primes = getPrimes(numbers)
print(primes) // Output: [2, 3, 5, 7, 11, 13]