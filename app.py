from flask import Flask, render_template, request, jsonify
import openai
from textblob import TextBlob
import re
from collections import Counter
from transformers import pipeline

app = Flask(__name__)

# Initialize text generation pipeline
try:
    text_generator = pipeline('text-generation', model='gpt2')
except Exception as e:
    print(f"Warning: Could not load text generation model: {e}")
    text_generator = None

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

def generate_text(prompt, max_length=100, temperature=0.7):
    """Generate text using GPT-2 model"""
    if not text_generator:
        return "Text generation model not available. Please install transformers: pip install transformers torch"
    
    if not prompt.strip():
        return "Please provide a prompt for text generation."
    
    try:
        # Generate text with specified parameters
        result = text_generator(
            prompt, 
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=text_generator.tokenizer.eos_token_id,
            do_sample=True
        )
        
        generated_text = result[0]['generated_text']
        
        # Remove the original prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text if generated_text else "Unable to generate meaningful text. Try a different prompt."
        
    except Exception as e:
        return f"Error generating text: {str(e)}"
    
from transformers import pipeline, MarianMTModel, MarianTokenizer

# Add this after your text_generator initialization
try:
    # Initialize translation pipeline - you can change the model as needed
    translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-de')  # English to German
    # For multiple languages, you might want to load different models dynamically
except Exception as e:
    print(f"Warning: Could not load translation model: {e}")
    translator = None

def translate_text(text, source_lang='en', target_lang='de'):
    """Translate text using Hugging Face transformers"""
    if not text.strip():
        return "Please provide text to translate."
    
    if not translator:
        return "Translation model not available. Please install transformers: pip install transformers torch"
    
    try:
        # For this example, using English to German
        # You can expand this to support multiple language pairs
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        
        # Create specific translator for the language pair
        specific_translator = pipeline('translation', model=model_name)
        
        result = specific_translator(text, max_length=512)
        translated_text = result[0]['translation_text']
        
        return translated_text
        
    except Exception as e:
        # Fallback to simple translation mapping for demo
        return f"Translation from {source_lang} to {target_lang}: {text} (Demo - install language models for actual translation)"

# Alternative simpler version using pre-defined language pairs
def translate_text_simple(text, target_lang='es'):
    """Simple translation function with popular language pairs"""
    if not text.strip():
        return "Please provide text to translate."
    
    # Language pair mappings (you can expand this)
    language_models = {
        'es': 'Helsinki-NLP/opus-mt-en-es',  # English to Spanish
        'fr': 'Helsinki-NLP/opus-mt-en-fr',  # English to French
        'de': 'Helsinki-NLP/opus-mt-en-de',  # English to German
        'it': 'Helsinki-NLP/opus-mt-en-it',  # English to Italian
        'pt': 'Helsinki-NLP/opus-mt-en-pt',  # English to Portuguese
    }
    
    try:
        if target_lang not in language_models:
            return f"Language '{target_lang}' not supported. Available: {', '.join(language_models.keys())}"
        
        model_name = language_models[target_lang]
        translator = pipeline('translation', model=model_name)
        
        result = translator(text, max_length=512)
        return result[0]['translation_text']
        
    except Exception as e:
        return f"Translation error: {str(e)}"

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
    elif any(word in message_lower for word in ['generate', 'text generation', 'writing']):
        return "I can help you generate text! Try the text generation feature - just provide a prompt and I'll continue the story or text."
    elif 'help' in message_lower:
        return "I can help you with: text summarization, sentiment analysis, text generation, and general questions. What would you like to know?"
    else:
        return "That's interesting! I'm here to help with text analysis tasks. You can use my summarization, sentiment analysis, text generation, or chat features."

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

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    
    generated_text = generate_text(prompt, max_length, temperature)
    return jsonify({'generated_text': generated_text})

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.get_json()
    text = data.get('text', '')
    target_lang = data.get('target_lang', 'es')  # Default to Spanish
    
    translated_text = translate_text_simple(text, target_lang)
    return jsonify({
        'translated_text': translated_text,
        'source_lang': 'en',
        'target_lang': target_lang
    })

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    message = data.get('message', '')
    
    response = chat_with_ai(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)