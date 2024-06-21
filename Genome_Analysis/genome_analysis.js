function levenshteinDistance(s1, s2) {
    let m = s1.length;
    let n = s2.length;
    let dp = new Array(2).fill(0).map(() => new Array(n + 1).fill(0));
    let substitutions = new Array(2).fill(0).map(() => new Array(n + 1).fill(0));

    for (let j = 0; j <= n; j++) {
        dp[0][j] = j;
    }

    for (let i = 1; i <= m; i++) {
        dp[i % 2][0] = i;
        for (let j = 1; j <= n; j++) {
            let cost = (s1[i - 1] === s2[j - 1]) ? 0 : 1;
            if (dp[(i - 1) % 2][j - 1] + cost <= dp[i % 2][j - 1] + 1 && dp[(i - 1) % 2][j - 1] + cost <= dp[(i - 1) % 2][j] + 1) {
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + cost;
                substitutions[i % 2][j] = substitutions[(i - 1) % 2][j - 1] + cost;
            } else if (dp[i % 2][j - 1] + 1 <= dp[(i - 1) % 2][j] + 1) {
                dp[i % 2][j] = dp[i % 2][j - 1] + 1;
                substitutions[i % 2][j] = substitutions[i % 2][j - 1];
            } else {
                dp[i % 2][j] = dp[(i - 1) % 2][j] + 1;
                substitutions[i % 2][j] = substitutions[(i - 1) % 2][j];
            }
        }
    }

    return { distance: dp[m % 2][n], substitutions: substitutions[m % 2][n] };
}

function hasNInsertionsOrMutationsOrFewer(s1, s2, n) {
    let { distance, substitutions } = levenshteinDistance(s1, s2);
    let insertions = distance - substitutions;
    return insertions <= n && substitutions <= n;
}

console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCD", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCND", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCD", 2));    // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCC", 2));    // false

console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCD", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCND", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCD", 2));    // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCC", 2));    // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGTCT", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGTCTT", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "CCGG", 2));    // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATCTG", 2));    // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "CCCC", 2));    // false
// Test the function
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ACTGA", 2)); // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ACTTAGC", 2)); // false

console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "TG", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGTCTT", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ACTGTT", 2)); // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGCTT", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATTTGCT", 2)); // false