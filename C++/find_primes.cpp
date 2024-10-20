#include <iostream>
#include <vector>

using namespace std;

// Function to check if a number is prime
bool isPrime(int n) {
  if (n <= 1) {
    return false;
  }
  for (int i = 2; i * i <= n; i++) {
    if (n % i == 0) {
      return false;
    }
  }
  return true;
}

int main() {
  vector<int> primes;
  int num = 2;

  while (primes.size() < 10000) {
    if (isPrime(num)) {
      primes.push_back(num);
    }
    num++;
  }

  // Print the first 10,000 prime numbers
  for (int i = 0; i < primes.size(); i++) {
    cout << primes[i] << " ";
  }
  cout << endl;

  return 0;
}