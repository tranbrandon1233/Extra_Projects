import Foundation

// Define the types of tokens that can appear in the input
enum Token: Equatable {
    case number(Double)  // A number token with a value
    case plus            // A plus token
    case minus           // A minus token
    case multiply        // A multiply token
    case divide          // A divide token
    case leftParen       // A left parenthesis token
    case rightParen      // A right parenthesis token
    
// Define equality for tokens
static func == (lhs: Token, rhs: Token) -> Bool {
    // Switch on both the left-hand side and right-hand side tokens
    switch (lhs, rhs) {
    // If both tokens are number tokens
    case (.number(let a), .number(let b)):
        // Return true if the values of the numbers are equal
        return a == b
    // If both tokens are plus, minus, multiply, divide, left parenthesis, or right parenthesis tokens
    case (.plus, .plus), (.minus, .minus), (.multiply, .multiply), (.divide, .divide),
         (.leftParen, .leftParen), (.rightParen, .rightParen):
        // Return true because these tokens do not have any associated values to compare
        return true
    // If the tokens are of different types
    default:
        // Return false because tokens of different types are not equal
        return false
        }
    }
}

/// Lexer class is responsible for breaking the input into tokens
class Lexer {
    let input: String
    var position: String.Index

    /// Initialize a new Lexer with the input string
    init(input: String) {
        self.input = input
        self.position = input.startIndex
    }

    /// This function returns the next token in the input
    /// - Returns: The next token, or nil if there are no more tokens
    func nextToken() -> Token? {
        // Skip whitespace
        while position < input.endIndex && input[position].isWhitespace {
            position = input.index(after: position)
        }

        // If we've reached the end of the input, return nil
        guard position < input.endIndex else {
            return nil
        }

        let currentChar = input[position]
        position = input.index(after: position)

        // Depending on the character, return the appropriate token
        switch currentChar {
        case "+": return .plus
        case "-": return .minus
        case "*": return .multiply
        case "/": return .divide
        case "(": return .leftParen
        case ")": return .rightParen
        // Case when the current character is a digit or a decimal point
        case "0"..."9":
            // Start a new number string with the current character
            var numberString = String(currentChar)
            // While the next character is a digit or a decimal point
            while position < input.endIndex && (input[position].isNumber || input[position] == ".") {
                // Append the next character to the number string
                numberString.append(input[position])
                // Move to the next character
                position = input.index(after: position)
            }
            // Return a number token with the value of the number string
            return .number(Double(numberString) ?? 0)
        // Case when the current character is unexpected
        default:
            // Throw an error with a message containing the unexpected character
            fatalError("Unexpected character: \(currentChar)")
        }
    }
}

/// Parser class is responsible for constructing an abstract syntax tree from the tokens
class Parser {
    let lexer: Lexer
    var currentToken: Token?

    /// Initialize a new Parser with the given Lexer
    init(lexer: Lexer) {
        self.lexer = lexer
        self.currentToken = lexer.nextToken()
    }

    /// This function returns the result of parsing the input
    /// - Returns: The result of the expression
    func parse() -> Double {
        return expression()
    }

    /// This function parses an expression, which is a series of terms separated by plus or minus signs
    /// - Returns: The result of the expression
    private func expression() -> Double {
        var result = term()

        while let token = currentToken { // Loop for each token and run the function based on the selected one
            switch token { 
            case .plus:
                eat(.plus)
                result += term()
            case .minus:
                eat(.minus)
                result -= term()
            default:
                return result
            }
        }

        return result
    }

    /// This function parses a term, which is a series of factors separated by multiply or divide signs
    /// - Returns: The result of the term
    private func term() -> Double {
        var result = factor()

        while let token = currentToken { // Loop for each token and run the function based on the selected one
            switch token {
            case .multiply:
                eat(.multiply)
                result *= factor()
            case .divide:
                eat(.divide)
                result /= factor()
            default:
                return result
            }
        }

        return result
    }

    /// This function parses a factor, which is either a number or an expression in parentheses
    /// - Returns: The result of the factor
    private func factor() -> Double {
        guard let token = currentToken else {
            fatalError("Unexpected end of input")
        }

        switch token { // Run function based on the input token
        case .number(let value):
            eat(.number(value))
            return value
        case .leftParen:
            eat(.leftParen)
            let result = expression()
            eat(.rightParen)
            return result
        case .minus:
            eat(.minus)
            return -factor()
        default:
            fatalError("Unexpected token: \(token)")
        }
    }

    /// This function consumes a token of the expected type
    /// - Parameter expectedToken: The expected token to consume
    private func eat(_ expectedToken: Token) {
        guard let token = currentToken, case expectedToken = token else {
            fatalError("Unexpected token: \(String(describing: currentToken))") // Error if there is an invalid token
        }
        currentToken = lexer.nextToken()
    }
}

/// This function compiles the input into a result
/// - Parameter input: The input string to compile
/// - Returns: The result of the compilation
func compile(input: String) -> Double {
    let lexer = Lexer(input: input)
    let parser = Parser(lexer: lexer)
    return parser.parse()
}

// Example usage
print("Enter an arithmetic expression:")
if let input = readLine() {
    let result = compile(input: input)
    print("Result: \(result)")
}