
import Foundation

func prefixCheck(_ prefix: String) -> String? {
    // Validate that the prefix contains only '0' and '1'
    for char in prefix {
        if char != "1" && char != "0" {
            return nil // Return nil for invalid characters
        }
    }
    return prefix // Return the prefix as is if valid
}

func generateNumbers(prefix: String, k: Int) {
    // Base case: if no more bits to add, print the complete sequence
    if k == 0 {
        print(prefix)
        return
    }
    // Recursive case: append '0' and '1' and reduce k
    generateNumbers(prefix: prefix + "0", k: k - 1)
    generateNumbers(prefix: prefix + "1", k: k - 1)
}

if let prefix = prefixCheck("0101") {
    // Calculate the correct k value based on the desired total length
    let totalLength = 7 // Example total length
    let additionalBits = totalLength - prefix.count
    if additionalBits >= 0 {
        generateNumbers(prefix: prefix, k: additionalBits)
    } else {
        print("Prefix is longer than the desired total length")
    }
} else {
    print("Invalid prefix")
}

// Test helper function to capture console output
func captureConsoleOutput(_ closure: () -> Void) -> String {
    let pipe = Pipe()
    let originalStdout = dup(STDOUT_FILENO)
    setvbuf(stdout, nil, _IONBF, 0)
    dup2(pipe.fileHandleForWriting.fileDescriptor, STDOUT_FILENO)
    
    closure()
    
    pipe.fileHandleForWriting.closeFile()
    fflush(stdout)
    dup2(originalStdout, STDOUT_FILENO)
    close(originalStdout)
    
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return String(data: data, encoding: .utf8) ?? ""
}

// Test suite structure
struct TestResult {
    let testName: String
    let passed: Bool
    let message: String
}

class BinaryGeneratorTests {
    var testResults: [TestResult] = []
    
    // Test prefixCheck function
    func testPrefixCheck() {
        // Test valid binary prefix
        if let result = prefixCheck("0101") {
            assert(result == "0101", "Valid binary prefix should return as is")
            testResults.append(TestResult(testName: "Valid Binary Prefix", passed: true, message: "Passed"))
        } else {
            testResults.append(TestResult(testName: "Valid Binary Prefix", passed: false, message: "Failed to validate correct binary prefix"))
        }
        
        // Test invalid characters
        if prefixCheck("012") != nil {
            testResults.append(TestResult(testName: "Invalid Character Check", passed: false, message: "Failed to reject invalid character"))
        } else {
            testResults.append(TestResult(testName: "Invalid Character Check", passed: true, message: "Passed"))
        }
        
        // Test empty string
        if let result = prefixCheck("") {
            assert(result == "", "Empty string should be valid")
            testResults.append(TestResult(testName: "Empty String", passed: true, message: "Passed"))
        } else {
            testResults.append(TestResult(testName: "Empty String", passed: false, message: "Failed to handle empty string"))
        }
    }
    
    // Test generateNumbers function
    func testGenerateNumbers() {
        // Test with prefix "0" and k=2
        let output1 = captureConsoleOutput {
            generateNumbers(prefix: "0", k: 2)
        }
        let expectedOutputs = Set(["000", "001", "010", "011"])
        let actualOutputs = Set(output1.components(separatedBy: .newlines).filter { !$0.isEmpty })
        
        testResults.append(TestResult(
            testName: "Generate Numbers Basic",
            passed: expectedOutputs == actualOutputs,
            message: expectedOutputs == actualOutputs ? "Passed" : "Failed: Expected \(expectedOutputs), got \(actualOutputs)"
        ))
        
        // Test with empty prefix and k=1
        let output2 = captureConsoleOutput {
            generateNumbers(prefix: "", k: 1)
        }
        let expectedOutputs2 = Set(["0", "1"])
        let actualOutputs2 = Set(output2.components(separatedBy: .newlines).filter { !$0.isEmpty })
        
        testResults.append(TestResult(
            testName: "Generate Numbers Empty Prefix",
            passed: expectedOutputs2 == actualOutputs2,
            message: expectedOutputs2 == actualOutputs2 ? "Passed" : "Failed: Expected \(expectedOutputs2), got \(actualOutputs2)"
        ))
    }
    
    // Test the complete workflow
    func testCompleteWorkflow() {
        // Test valid case
        let output = captureConsoleOutput {
            if let prefix = prefixCheck("01") {
                let totalLength = 4
                let additionalBits = totalLength - prefix.count
                if additionalBits >= 0 {
                    generateNumbers(prefix: prefix, k: additionalBits)
                }
            }
        }
        let expectedOutputs = Set(["0100", "0101", "0110", "0111"])
        let actualOutputs = Set(output.components(separatedBy: .newlines).filter { !$0.isEmpty })
        
        testResults.append(TestResult(
            testName: "Complete Workflow",
            passed: expectedOutputs == actualOutputs,
            message: expectedOutputs == actualOutputs ? "Passed" : "Failed: Expected \(expectedOutputs), got \(actualOutputs)"
        ))
        
        // Test invalid prefix
        let outputInvalid = captureConsoleOutput {
            if let prefix = prefixCheck("012") {
                generateNumbers(prefix: prefix, k: 2)
            } else {
                print("Invalid prefix")
            }
        }
        
        testResults.append(TestResult(
            testName: "Invalid Prefix Workflow",
            passed: outputInvalid.trimmingCharacters(in: .whitespacesAndNewlines) == "Invalid prefix",
            message: outputInvalid.trimmingCharacters(in: .whitespacesAndNewlines) == "Invalid prefix" ? "Passed" : "Failed"
        ))
    }
    
    // Run all tests
    func runTests() {
        testPrefixCheck()
        testGenerateNumbers()
        testCompleteWorkflow()
        
        // Print results
        print("\n=== Test Results ===")
        for result in testResults {
            print("\(result.testName): \(result.passed ? "✅" : "❌") - \(result.message)")
        }
        
        let passedTests = testResults.filter { $0.passed }.count
        let totalTests = testResults.count
        print("\nPassed \(passedTests)/\(totalTests) tests")
    }
}

// Run the tests
let tests = BinaryGeneratorTests()
tests.runTests()