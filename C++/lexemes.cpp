#include <iostream>
#include <cstring>

enum TokenType {
    LEFT_PARENTHESIS, RIGHT_PARENTHESIS, LEFT_BRACKET, RIGHT_BRACKET,
    WHILE_KEYWORD, RETURN_KEYWORD, EQUAL, COMMA, EOL, VARTYPE,
    BINOP, NUMBER, IDENTIFIER
};

struct Lexeme {
    char lexeme[50];
    TokenType token;
};

// Function prototypes
bool validNumber(const char* str);
bool validIdentifier(const char* str);

// The function to process the lexemes
bool processLexemes(Lexeme aLex[], int lexCount) {
    for (int i = 0; i < lexCount; i++) {
        if (strcmp(aLex[i].lexeme, "(") == 0) aLex[i].token = LEFT_PARENTHESIS;
        else if (strcmp(aLex[i].lexeme, ")") == 0) aLex[i].token = RIGHT_PARENTHESIS;
        else if (strcmp(aLex[i].lexeme, "{") == 0) aLex[i].token = LEFT_BRACKET;
        else if (strcmp(aLex[i].lexeme, "}") == 0) aLex[i].token = RIGHT_BRACKET;
        else if (strcmp(aLex[i].lexeme, "while") == 0) aLex[i].token = WHILE_KEYWORD;
        else if (strcmp(aLex[i].lexeme, "return") == 0) aLex[i].token = RETURN_KEYWORD;
        else if (strcmp(aLex[i].lexeme, "=") == 0) aLex[i].token = EQUAL;
        else if (strcmp(aLex[i].lexeme, ",") == 0) aLex[i].token = COMMA;
        else if (strcmp(aLex[i].lexeme, ";") == 0) aLex[i].token = EOL;
        else if (strcmp(aLex[i].lexeme, "int") == 0 || strcmp(aLex[i].lexeme, "void") == 0) aLex[i].token = VARTYPE;
        else if (strcmp(aLex[i].lexeme, "+") == 0 || strcmp(aLex[i].lexeme, "!=") == 0 || strcmp(aLex[i].lexeme, "==") == 0 || strcmp(aLex[i].lexeme, "%%") == 0 || strcmp(aLex[i].lexeme, "*") == 0) aLex[i].token = BINOP;
        else if (validNumber(aLex[i].lexeme)) aLex[i].token = NUMBER;
        else if (validIdentifier(aLex[i].lexeme)) aLex[i].token = IDENTIFIER;
    }
    return true;
}

// Dummy implementations for validNumber and validIdentifier
bool validNumber(const char* str) {
    for (int i = 0; str[i] != '\0'; i++) {
        if (!isdigit(str[i])) {
            return false;
        }
    }
    return true;
}

bool validIdentifier(const char* str) {
    if (!isalpha(str[0])) {
        return false;
    }
    for (int i = 1; str[i] != '\0'; i++) {
        if (!isalnum(str[i])) {
            return false;
        }
    }
    return true;
}

// Test function
void testProcessLexemes() {
    Lexeme aLex[] = {
        {"(", LEFT_PARENTHESIS},
        {"}", RIGHT_BRACKET},
        {"while", WHILE_KEYWORD},
        {"42", NUMBER},
        {"myVar", IDENTIFIER},
        {"==", BINOP}
    };
    int lexCount = sizeof(aLex) / sizeof(Lexeme);
    processLexemes(aLex, lexCount);

    for (int i = 0; i < lexCount; i++) {
        std::cout << "Lexeme: " << aLex[i].lexeme << ", Token: " << aLex[i].token << std::endl;
    }
}

int main() {
    testProcessLexemes();
    return 0;
}
