import gradio as gr
import random
# Sample greetings and responses (replace with your actual chatbot logic)
greetings = ["Hello!", "Hi there", "What can I help you with?"]
responses = [
    "Hi! I'm a simple chatbot. Ask me anything.",
    "Hello. What's on your mind today?",
    "Happy to help! How can I assist you?",
]
default_response = "I am still learning. Could you rephrase that?"

def respond(text,etc):
  """
  Replaces keywords with predefined responses.
  """
  for keyword, phrase in [("hi", greetings), ("hello", greetings), ("thank you", ["You're welcome!", "No problem.", "Happy to help."]), ("bye", ["Goodbye!", "See you later!", "Talk to you soon."])]:
    if keyword in text.lower():
      return random.choice(phrase)
  return default_response

# Create the chat interface
iface = gr.ChatInterface(
    fn=respond,
    title="Simple Chatbot",
    description="Ask me anything!",
)

iface.launch(share=True)