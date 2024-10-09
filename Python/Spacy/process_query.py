from transformers import AutoTokenizer, AutoModel
import spacy

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

query = "Find information about natural language processing."
doc = nlp(query)
span = doc[3:6]  # Example span: "natural language processing"

# Incorrect (will raise error):
# embeddings = model.embed_query(span.replace("language", "text")) 

# Correct:
modified_query = span.text.replace("language", "text")
embeddings = model.embed_query(modified_query) 
print(modified_query)
print(embeddings)