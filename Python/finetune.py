from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd

# Load your FAQ data
df = pd.read_csv("your_faq_data.csv")

# Prepare data in the format expected by the model
train_data = [{"question": row["question"], "answer": row["answer"]} for _, row in df.iterrows()]

# Load pre-trained GPT-2 model and tokenizer
model_name = "google/flan-t5-small"  # Choose a smaller GPT-2 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    # ... other training arguments ...
)

# Create a Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,  # You'll likely need to split into train/validation
    # ... other arguments for the trainer ...
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")