from flask import Flask, render_template, request, jsonify
import openai
from textblob import TextBlob
import re
from collections import Counter


app = Flask(__name__)

# Initialize translator


# For demo purposes - replace with your actual OpenAI API key
# openai.api_key = "your-openai-api-key-here"

def summarize_text(text, max_sentences=3):
    """Simple extractive summarization"""
    if not text.strip():
        return "No text provided for summarization."
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    # Simple word frequency based summarization
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)
    
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = re.findall(r'\w+', sentence.lower())
        score = sum(word_freq[word] for word in sentence_words)
        sentence_scores[sentence] = score
    
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
    summary = '. '.join([sentence for sentence, score in top_sentences]) + '.'
    
    return summary

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if not text.strip():
        return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0}
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 2),
        "subjectivity": round(subjectivity, 2)
    }



def chat_with_ai(message, conversation_history=[]):
    """Simple rule-based chatbot - replace with actual AI integration"""
    # This is a simple rule-based response system
    # Replace with actual OpenAI API call when you have the key
    
    message_lower = message.lower()
    
    if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
        return "Hello! How can I help you today?"
    elif any(word in message_lower for word in ['how are you', 'how do you do']):
        return "I'm doing well, thank you! I'm here to help with text analysis tasks."
    elif any(word in message_lower for word in ['summarize', 'summary']):
        return "I can help you summarize text! Just paste your text in the summarization tool."
    elif any(word in message_lower for word in ['translate', 'translation']):
        return "I can translate text between different languages. Try the translation feature!"
    elif any(word in message_lower for word in ['sentiment', 'emotion']):
        return "I can analyze the sentiment of your text. Use the sentiment analysis tool to get started."
    elif 'help' in message_lower:
        return "I can help you with: text summarization, sentiment analysis, language translation, and general questions. What would you like to know?"
    else:
        return "That's interesting! I'm here to help with text analysis tasks. You can use my summarization, sentiment analysis, or translation features."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.get_json()
    text = data.get('text', '')
    max_sentences = data.get('max_sentences', 3)
    
    summary = summarize_text(text, max_sentences)
    return jsonify({'summary': summary})

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    
    result = analyze_sentiment(text)
    return jsonify(result)



@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    message = data.get('message', '')
    
    response = chat_with_ai(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)