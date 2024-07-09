from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('svm_sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and word not in punctuation]
    return ' '.join(filtered_tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json['text']
    processed_text = preprocess_text(text)
    inputs = vectorizer.transform([processed_text])
    
    # Use decision function or probabilities for more granular score
    decision_function = model.decision_function(inputs)
    polarity_score = decision_function[0]  # Get the score for the first (and only) input
    
    sentiment_label = 'positive' if polarity_score > 0 else 'negative'
    
    # Normalize polarity score to be between -10 and 10 for simplicity
    normalized_score = int((polarity_score / np.max(np.abs(decision_function))) * 10)
    
    return jsonify({'sentiment': sentiment_label, 'polarity_score': normalized_score})

if __name__ == '__main__':
    app.run(debug=True)
