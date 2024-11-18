import Foundation

class Calculator {
    var result: Double = 0.0

    func add(_ numbers: [Double]) -> Double {
        result = numbers.reduce(0, +)
        return result
    }
    
    func subtract(_ numbers: [Double]) -> Double {
        guard !numbers.isEmpty else { return 0.0 }
        result = numbers.dropFirst().reduce(numbers[0]) { $0 - $1 }
        return result
    }
    
    func multiply(_ numbers: [Double]) -> Double {
        guard !numbers.isEmpty else { return 0.0 }
        result = numbers.reduce(1, *)
        return result
    }
    
    func divide(_ numbers: [Double]) -> Double {
        guard !numbers.isEmpty else { return 0.0 }
        result = numbers.dropFirst().reduce(numbers[0]) { $1 == 0 ? 0 : $0 / $1 }
        return result
    }
}

class ScientificCalculator: Calculator {
    var memory: Double = 0.0
    var angleUnit: String = "radians" 
    
    // Trigonometric functions with angle unit handling
    func sine(_ angle: Double) -> Double {
        let angleInRadians = angleUnit == "degrees" ? angle * .pi / 180 : angle
        return sin(angleInRadians)
    }
    
    func cosine(_ angle: Double) -> Double {
        let angleInRadians = angleUnit == "degrees" ? angle * .pi / 180 : angle
        return cos(angleInRadians)
    }
    
    func tangent(_ angle: Double) -> Double {
        let angleInRadians = angleUnit == "degrees" ? angle * .pi / 180 : angle
        return tan(angleInRadians)
    }
    
    // Logarithmic functions with error handling
    func naturalLog(_ value: Double) -> Double {
        guard value > 0 else {
            print("Error: Natural log undefined for zero or negative numbers.")
            return Double.nan
        }
        return log(value)
    }
    
    func log10(_ value: Double) -> Double {
        guard value > 0 else {
            print("Error: Log base 10 undefined for zero or negative numbers.")
            return Double.nan
        }
        return Foundation.log10(value) // Explicitly call Foundation's log10
    }
    
    // Exponentiation with special case handling
    func power(base: Double, exponent: Double) -> Double {
        if base == 0 && exponent == 0 {
            print("Error: 0^0 is undefined.")
            return Double.nan
        } else if base < 0 && exponent.truncatingRemainder(dividingBy: 1) != 0 {
            print("Error: Negative base with fractional exponent is undefined.")
            return Double.nan
        }
        return pow(base, exponent)
    }
    
    // Memory functions with clear functionality
    func storeInMemory(_ value: Double) {
        memory = value
    }
    
    func recallMemory() -> Double {
        return memory
    }
    
    func clearMemory() {
        memory = 0.0
    }
}

// Test function to compare floating point numbers with tolerance
func areEqual(_ a: Double, _ b: Double, tolerance: Double = 0.000001) -> Bool {
    return abs(a - b) < tolerance
}

// Function to print test results
func assertTest(_ testName: String, _ condition: Bool) {
    print("\(testName): \(condition ? "PASSED" : "FAILED")")
}

// Basic Calculator Tests
func testBasicCalculator() {
    let calc = Calculator()
    
    // Addition Tests
    assertTest("Add empty array", areEqual(calc.add([]), 0))
    assertTest("Add single number", areEqual(calc.add([5]), 5))
    assertTest("Add positive numbers", areEqual(calc.add([1, 2, 3]), 6))
    assertTest("Add negative numbers", areEqual(calc.add([-1, -2, -3]), -6))
    assertTest("Add mixed numbers", areEqual(calc.add([1.5, -2.5, 3.5]), 2.5))
    assertTest("Add very large numbers", areEqual(calc.add([Double.greatestFiniteMagnitude, 1]), Double.greatestFiniteMagnitude))
    
    // Subtraction Tests
    assertTest("Subtract empty array", areEqual(calc.subtract([]), 0))
    assertTest("Subtract single number", areEqual(calc.subtract([5]), 5))
    assertTest("Subtract multiple numbers", areEqual(calc.subtract([10, 3, 2]), 5))
    assertTest("Subtract negative numbers", areEqual(calc.subtract([-10, -3, -2]), -5))
    assertTest("Subtract to zero", areEqual(calc.subtract([10, 5, 5]), 0))
    assertTest("Subtract decimals", areEqual(calc.subtract([10.5, 3.2, 2.1]), 5.2))
    
    // Multiplication Tests
    assertTest("Multiply empty array", areEqual(calc.multiply([]), 0))
    assertTest("Multiply single number", areEqual(calc.multiply([5]), 5))
    assertTest("Multiply positive numbers", areEqual(calc.multiply([2, 3, 4]), 24))
    assertTest("Multiply by zero", areEqual(calc.multiply([5, 0, 3]), 0))
    assertTest("Multiply negative numbers", areEqual(calc.multiply([-2, -3]), 6))
    assertTest("Multiply mixed numbers", areEqual(calc.multiply([-2, 3, -4]), 24))
    
    // Division Tests
    assertTest("Divide empty array", areEqual(calc.divide([]), 0))
    assertTest("Divide single number", areEqual(calc.divide([5]), 5))
    assertTest("Basic division", areEqual(calc.divide([10, 2]), 5))
    assertTest("Divide by zero", areEqual(calc.divide([10, 0]), 0))
    assertTest("Multiple divisions", areEqual(calc.divide([100, 2, 2]), 25))
    assertTest("Divide negative numbers", areEqual(calc.divide([-10, -2]), 5))
    assertTest("Divide decimals", areEqual(calc.divide([10.5, 2.5]), 4.2))
}

// Scientific Calculator Tests
func testScientificCalculator() {
    let sciCalc = ScientificCalculator()
    
    // Trigonometric Tests
    sciCalc.angleUnit = "radians"
    assertTest("Sine of 0 radians", areEqual(sciCalc.sine(0), 0))
    assertTest("Sine of π/2 radians", areEqual(sciCalc.sine(Double.pi/2), 1))
    assertTest("Cosine of 0 radians", areEqual(sciCalc.cosine(0), 1))
    assertTest("Cosine of π radians", areEqual(sciCalc.cosine(Double.pi), -1))
    assertTest("Tangent of 0 radians", areEqual(sciCalc.tangent(0), 0))
    
    sciCalc.angleUnit = "degrees"
    assertTest("Sine of 90 degrees", areEqual(sciCalc.sine(90), 1))
    assertTest("Cosine of 180 degrees", areEqual(sciCalc.cosine(180), -1))
    assertTest("Tangent of 45 degrees", areEqual(sciCalc.tangent(45), 1))
    
    // Logarithm Tests
    assertTest("Natural log of 1", areEqual(sciCalc.naturalLog(1), 0))
    assertTest("Natural log of e", areEqual(sciCalc.naturalLog(2.71), 1))
    assertTest("Natural log of negative", sciCalc.naturalLog(-1).isNaN)
    assertTest("Natural log of zero", sciCalc.naturalLog(0).isNaN)
    
    assertTest("Log10 of 1", areEqual(sciCalc.log10(1), 0))
    assertTest("Log10 of 10", areEqual(sciCalc.log10(10), 1))
    assertTest("Log10 of negative", sciCalc.log10(-1).isNaN)
    assertTest("Log10 of zero", sciCalc.log10(0).isNaN)
    
    // Power Tests
    assertTest("Power positive base and exponent", areEqual(sciCalc.power(base: 2, exponent: 3), 8))
    assertTest("Power negative base, even exponent", areEqual(sciCalc.power(base: -2, exponent: 2), 4))
    assertTest("Power negative base, odd exponent", areEqual(sciCalc.power(base: -2, exponent: 3), -8))
    assertTest("Power zero base, positive exponent", areEqual(sciCalc.power(base: 0, exponent: 5), 0))
    assertTest("Power zero base, zero exponent", sciCalc.power(base: 0, exponent: 0).isNaN)
    assertTest("Power negative base, fractional exponent", sciCalc.power(base: -2, exponent: 0.5).isNaN)
    
    // Memory Tests
    sciCalc.storeInMemory(5.5)
    assertTest("Memory store and recall", areEqual(sciCalc.recallMemory(), 5.5))
}

// Run all tests
print("Running Basic Calculator Tests:")
testBasicCalculator()
print("\nRunning Scientific Calculator Tests:")
testScientificCalculator()