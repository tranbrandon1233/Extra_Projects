from transformers import AutoTokenizer, AutoModel
import Python.process_query as process_query

nlp = process_query.load("en_core_web_sm")

# Example model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def preprocess_query(query):
    doc = nlp(query)
    new_query = ""
    for token in doc:
        if token.tag_ == "NN": # Replace nouns with 'NOUN'
            new_query += " NOUN "
        else:
            new_query += " " + token.text
    return new_query.strip()

query = "The quick brown fox jumps over the lazy dog."
preprocessed_query = preprocess_query(query) 

inputs = tokenizer(preprocessed_query, return_tensors="pt") 
embeddings = model(**inputs).last_hidden_state  # Get embeddings