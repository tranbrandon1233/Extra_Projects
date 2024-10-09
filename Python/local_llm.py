import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

# Model and tokenizer loading (replace with your preferred model)
model_name = "facebook/bart-large-cnn"  # Example: facebook/bart-large-cnn
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# FastAPI app
app = FastAPI()

# Request/response model for structured data handling
class Question(BaseModel):
    text: str

class Answer(BaseModel):
    text: str

# Store conversation history for context
conversation_history = {}

@app.post("/chat", response_model=Answer)
async def chat(question: Question):
    user_question = question.text
    user_id = "default_user"  # You can implement user identification later

    # Retrieve or initialize conversation history
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Add current question to history
    conversation_history[user_id].append(f"User: {user_question}")

    # Format conversation history for the model
    formatted_history = "\n".join(conversation_history[user_id])

    # Generate response using the model
    input_text = f"{formatted_history}\nBot: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=1000, num_beams=5, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract bot's response from generated text
    bot_response = generated_text.split("Bot: ")[-1]

    # Update conversation history
    conversation_history[user_id].append(f"Bot: {bot_response}")

    return {"text": bot_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)