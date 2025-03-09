function isOneEditDistance(s,t){
        const m = s.length;
        const n = t.length;
        if (Math.abs(m-n) > 1) return False;
        if (m>n) return isOneEditDistance(t,s);
        let i = 0;
        while (i<m && s[i] === t[i]){
            i++;
        }
        if (i === m) return n-m === 1;
        if (m === n){
            i++;
            while (i < m && s[i] === t[i]){
                i++;
            }
            return i === m;
        }
        while (i<m && s[i] === t[i+1]){
            i++;
        }
        return i === m;
}

// Test cases
console.log(isOneEditDistance("abc", "ab")); // true
console.log(isOneEditDistance("abc", "abcd")); // true
console.log(isOneEditDistance("ab", "acb")); // true
console.log(isOneEditDistance("abc", "abc")); // false 
console.log(isOneEditDistance("", "")); // false 
