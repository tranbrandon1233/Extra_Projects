function levenshteinDistance(s1, s2) {
    let m = s1.length;
    let n = s2.length;
    let dp = new Array(m + 1).fill(0).map(() => new Array(n + 1).fill(0));
    let substitutions = new Array(m + 1).fill(0).map(() => new Array(n + 1).fill(0));

    for (let i = 0; i <= m; i++) {
        dp[i][0] = i;
    }

    for (let j = 0; j <= n; j++) {
        dp[0][j] = j;
    }

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            let cost = (s1[i - 1] === s2[j - 1]) ? 0 : 1;
            if (dp[i - 1][j - 1] + cost <= dp[i][j - 1] + 1 && dp[i - 1][j - 1] + cost <= dp[i - 1][j] + 1) {
                dp[i][j] = dp[i - 1][j - 1] + cost;
                substitutions[i][j] = substitutions[i - 1][j - 1] + cost;
            } else if (dp[i][j - 1] + 1 <= dp[i - 1][j] + 1) {
                dp[i][j] = dp[i][j - 1] + 1;
                substitutions[i][j] = substitutions[i][j - 1];
            } else {
                dp[i][j] = dp[i - 1][j] + 1;
                substitutions[i][j] = substitutions[i - 1][j];
            }
        }
    }

    return { distance: dp[m][n], substitutions: substitutions[m][n] };
}

function hasNInsertionsOrMutationsOrFewer(s1, s2, n) {
    if (s2.length > s1.length) {
        return false;
    }

    for (let i = 0; i <= s1.length - s2.length; i++) {
        let { distance, substitutions } = levenshteinDistance(s1.substring(i, i + s2.length), s2);
        let insertions = distance - substitutions;
        if (insertions <= n && substitutions <= n) {
            return true;
        }
    }

    return false;
}



console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCD", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCND", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCD", 2));    // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCC", 2));    // false

console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCD", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCND", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCD", 2));    // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "CCCC", 2));    // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGTCT", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGTCTT", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "CCGG", 2));    // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATCTG", 2));    // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "CCCC", 2));    // false
// Test the function
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ACTGA", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ACTTAGC", 2)); // false

console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "TG", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGTCTT", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ACTGTT", 2)); // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATGCTT", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ACTG", "ATTTGCT", 2)); // false

console.log(hasNInsertionsOrMutationsOrFewer("ANBNCD", "ABCD", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ANBNCND", "ABCD", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ANXNCD", "ABCD", 3));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ANXNCND", "ABCD", 3));  // false
console.log("\n")
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCD", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANBNCND", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANXNCD", 3));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ANXNCND", 3));  // false
console.log("\n")
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABCD", 0));  // true
console.log("\n")
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABCDE", 0));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABCDE", 1));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABCDED", 1));  // false
console.log("\n")

console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "AB", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "A", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "AB", 1));  // false
console.log("\n")

console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABEE", 1));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABED", 1));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABE", 2));  // true
console.log("\n")
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABE", 1));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABEDD", 2));  // false
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "AACDD", 2));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ABEDD", 1));  // false

console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ACN", 3));  // true
console.log(hasNInsertionsOrMutationsOrFewer("ABCD", "ACN", 1));  // false