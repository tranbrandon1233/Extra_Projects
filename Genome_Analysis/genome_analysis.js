function levenshteinDistance(s1, s2) {
    if (s1.length < s2.length) {
        return levenshteinDistance(s2, s1);
    }

    if (s2.length === 0) {
        return { distance: s1.length, substitutions: 0 };
    }

    let previousRow = new Array(s2.length + 1);
    let previousSubstitutions = new Array(s2.length + 1);
    for (let i = 0; i <= s2.length; i++) {
        previousRow[i] = i;
        previousSubstitutions[i] = 0;
    }

    for (let i = 1; i <= s1.length; i++) {
        let currentRow = new Array(s2.length + 1);
        let currentSubstitutions = new Array(s2.length + 1);
        currentRow[0] = i;
        currentSubstitutions[0] = 0;

        for (let j = 1; j <= s2.length; j++) {
            let cost = (s1[i - 1] === s2[j - 1]) ? 0 : 1;
            if (previousRow[j - 1] + cost <= currentRow[j - 1] + 1 && previousRow[j - 1] + cost <= previousRow[j] + 1) {
                currentRow[j] = previousRow[j - 1] + cost;
                currentSubstitutions[j] = previousSubstitutions[j - 1] + cost;
            } else if (currentRow[j - 1] + 1 <= previousRow[j] + 1) {
                currentRow[j] = currentRow[j - 1] + 1;
                currentSubstitutions[j] = currentSubstitutions[j - 1];
            } else {
                currentRow[j] = previousRow[j] + 1;
                currentSubstitutions[j] = previousSubstitutions[j];
            }
        }

        previousRow = currentRow;
        previousSubstitutions = currentSubstitutions;
    }

    return { distance: previousRow[s2.length], substitutions: previousSubstitutions[s2.length] };
}

function hasNInsertionsOrMutationsOrFewer(s1, s2, n) {
    let { distance, substitutions } = levenshteinDistance(s1, s2);
    let insertions = distance - substitutions;
    return insertions <= n && substitutions <= n;
}

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