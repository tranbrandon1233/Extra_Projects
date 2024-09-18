import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

encoder_model_name = "alexyalunin/RuBioRoBERTa"
tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
embedding_model = AutoModel.from_pretrained(encoder_model_name)
embedding_model.eval()
embedding_model.to("cuda")

def get_embedding(text, chunk_size=256, overlap=128):
    tokens = tokenizer(text.lower(), padding=True, truncation=False, return_tensors='pt')
    input_ids = tokens['input_ids'].to("cuda")
    attention_mask = tokens['attention_mask'].to("cuda")

    chunk_embeddings = []
    for i in range(0, input_ids.size(1), chunk_size - overlap):
        chunk_input_ids = input_ids[:, i:i + chunk_size]
        chunk_attention_mask = attention_mask[:, i:i + chunk_size]

        with torch.inference_mode():
            chunk_output = embedding_model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
            chunk_embedding = chunk_output.last_hidden_state.mean(dim=1).cpu().detach().numpy()
            chunk_embeddings.append(chunk_embedding)

    return np.mean(chunk_embeddings, axis=0) 

# Example usage
text = "Your long medical text here..."
embedding = get_embedding(text)
print(embedding.shape) 