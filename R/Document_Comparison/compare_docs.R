# Load required libraries
library(tm)

setwd("C:\\path\\to\\your\\text\\files")

# Load documents (replace with your file paths)
doc1_path <- "con_tax_doc.txt"
doc2_path <- "left_tax_doc.txt"

# Function to preprocess text
preprocess_text <- function(text) {
  text <- tolower(text)
  text <- removePunctuation(text)
  text <- removeNumbers(text)
  text <- stripWhitespace(text)
  return(text)
}

# Read and preprocess documents
doc1 <- readLines(doc1_path)
doc2 <- readLines(doc2_path)

doc1 <- preprocess_text(paste(doc1, collapse = " "))
doc2 <- preprocess_text(paste(doc2, collapse = " "))

# Create a corpus
corpus <- Corpus(VectorSource(c(doc1, doc2)))

# Create a document-term matrix
dtm <- DocumentTermMatrix(corpus)

# Convert DTM to matrix
dtm_matrix <- as.matrix(dtm)

# Ensure the matrix is not empty
if (ncol(dtm_matrix) == 0) {
  stop("The document-term matrix is empty. Check if the documents contain any common terms after preprocessing.")
}

# Function to calculate cosine similarity
cosine_similarity_manual <- function(vec1, vec2) {
  dot_product <- sum(vec1 * vec2)
  magnitude1 <- sqrt(sum(vec1^2))
  magnitude2 <- sqrt(sum(vec2^2))
  if (magnitude1 == 0 || magnitude2 == 0) {
    return(0) # Avoid division by zero
  }
  return(dot_product / (magnitude1 * magnitude2))
}

# Calculate cosine similarity manually
similarity_score <- cosine_similarity_manual(dtm_matrix[1, ], dtm_matrix[2, ])

# Print raw similarity score for debugging
print(paste("Raw similarity score:", similarity_score))

# Set a threshold for similarity
similarity_threshold <- 0.7

# Determine if the documents share similar arguments
if (!is.na(similarity_score) && length(similarity_score) == 1) {
  if (similarity_score > similarity_threshold) {
    print("The documents appear to share similar political arguments.")
  } else {
    print("The documents do not appear to share similar political arguments.")
  }
  # Print the similarity score
  print(paste("Cosine similarity score:", similarity_score))
} else {
  print("Unable to calculate a valid similarity score. Check the document contents and preprocessing steps.")
  print("Similarity score details:")
  print(similarity_score)
}

freq_terms1 <- findFreqTerms(dtm[1, ], lowfreq = 5)
freq_terms2 <- findFreqTerms(dtm[2, ], lowfreq = 5)
print("Frequent terms in document 1:")
print(freq_terms1)
print("Frequent terms in document 2:")
print(freq_terms2)
