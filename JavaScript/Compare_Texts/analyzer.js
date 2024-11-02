const natural = require('natural');
const fs = require('fs');

// Read the contents of the two text files
const text1 = fs.readFileSync('text1.txt', 'utf8');
const text2 = fs.readFileSync('text2.txt', 'utf8');

// Tokenize the text into individual words
const tokenizer = new natural.WordTokenizer();
const tokens1 = tokenizer.tokenize(text1);
const tokens2 = tokenizer.tokenize(text2);

// Calculate TF-IDF vectors for both texts
const TfIdf = new natural.TfIdf();
TfIdf.addDocument(tokens1);
TfIdf.addDocument(tokens2);

// Function to get TF-IDF vector for a document
function getTfIdfVector(TfIdf, tokens, docIndex) {
    const vector = {};
    tokens.forEach(token => {
        const tfidfValue = TfIdf.tfidf(token, docIndex);
        if (tfidfValue > 0) {
            vector[token] = tfidfValue;
        }
    });
    return vector;
}

// Get TF-IDF vectors
const vector1 = getTfIdfVector(TfIdf, tokens1, 0);
const vector2 = getTfIdfVector(TfIdf, tokens2, 1);

// Calculate cosine similarity manually
function cosineSimilarity(vec1, vec2) {
    // Get unique tokens from both vectors
    const allTokens = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
    
    // Calculate dot product and magnitudes
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;
    
    for (const token of allTokens) {
        const val1 = vec1[token] || 0;
        const val2 = vec2[token] || 0;
        
        dotProduct += val1 * val2;
        magnitude1 += val1 * val1;
        magnitude2 += val2 * val2;
    }
    
    // Prevent division by zero
    if (magnitude1 === 0 || magnitude2 === 0) return 0;
    
    return dotProduct / (Math.sqrt(magnitude1) * Math.sqrt(magnitude2));
}

// Calculate cosine similarity
const similarity = cosineSimilarity(vector1, vector2);

// Print the similarity score (between -1 and 1)
console.log(`Similarity: ${similarity}`);

// Interpret the similarity score as a probability
const probability = (similarity + 1) / 2; // Scale the similarity score to a probability between 0 and 1
console.log(`Probability that the texts were written by the same author: ${(probability * 100).toFixed(2)}%`);