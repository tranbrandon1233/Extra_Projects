func strongPasswordChecker(_ s: String) -> Int {
    var res = 0, lower = 1, upper = 1, digit = 1
    let characters = Array(s)
    var repeatCounts = [Int]()
    
    var i = 0
    while i < characters.count {
        if characters[i].isLowercase { lower = 0 }
        if characters[i].isUppercase { upper = 0 }
        if characters[i].isNumber { digit = 0 }
        
        let j = i
        while i < characters.count && characters[i] == characters[j] {
            i += 1
        }
        repeatCounts.append(i - j)
    }
    
    let totalMissing = lower + upper + digit
    
    if characters.count < 6 {
        res += totalMissing + max(0, 6 - (characters.count + totalMissing))
    } else {
        var overLength = max(characters.count - 20, 0)
        res += overLength
        
        for k in 1...2 {
            for (index, count) in repeatCounts.enumerated() where count >= 3 && overLength > 0 {
                if count % 3 == k - 1 {
                    let reduction = min(overLength, k)
                    repeatCounts[index] -= reduction
                    overLength -= reduction
                }
            }
        }
        
        var leftOver = 0
        for count in repeatCounts where count >= 3 {
            leftOver += count / 3
        }
        
        res += max(totalMissing, leftOver)
    }
    
    return res
}

print(strongPasswordChecker("aA1"))