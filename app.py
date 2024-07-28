import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Generate response using distilGPT-2
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=170, num_return_sequences=1, temperature=0.9, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Context management for maintaining conversation
class Chatbot:
    def __init__(self):
        self.history = []

    def add_to_history(self, user_input, bot_response):
        self.history.append({"user": user_input, "bot": bot_response})

    def get_chat_history(self):
        return "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in self.history])

chatbot = Chatbot()

def generate_response_with_context(user_input):
    prompt = chatbot.get_chat_history() + f"\nUser: {user_input}\nBot:"
    response = generate_response(prompt)
    chatbot.add_to_history(user_input, response)
    return response

def analyze_sentiment(text):
    return sid.polarity_scores(text)

def generate_response_with_sentiment(user_input):
    sentiment = analyze_sentiment(user_input)
    if sentiment['compound'] >= 0.05:
        sentiment_label = "positive"
    elif sentiment['compound'] <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    prompt = chatbot.get_chat_history() + f"\nUser: {user_input}\nSentiment: {sentiment_label}\nBot:"
    response = generate_response(prompt)
    chatbot.add_to_history(user_input, response)
    return response

# Flask web interface
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_input = request.args.get('msg')
    bot_response = generate_response_with_sentiment(user_input)
    return jsonify(bot_response)

if __name__ == "__main__":
    app.run(port=5000)
