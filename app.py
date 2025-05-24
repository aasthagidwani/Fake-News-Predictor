import pandas as pd
import pickle
import logging
import spacy
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from utils.preprocessing import clean_text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def train_model():
    try:
        # Load dataset
        df = pd.read_excel('dataset.xlsx')
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(clean_text)
        
        # Prepare features and labels
        X = df['processed_text']
        y = df['fake/real'].map({'real': 1, 'fake': 0})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Save model and vectorizer
        with open('models/fake_news_model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
        
        logging.info(f"Model trained successfully. Metrics: {metrics}")
        return model, vectorizer, metrics
    
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

def scrape_article(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content (this is a basic implementation; adjust based on website structure)
        article = soup.find('article') or soup.find('div', class_='content')
        text = article.get_text(separator=' ', strip=True) if article else soup.get_text(separator=' ', strip=True)
        
        return clean_text(text)
    except Exception as e:
        logging.error(f"Error scraping URL {url}: {str(e)}")
        return None

def summarize_text(text):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 3)  # 3 sentences for ~60 words
        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        logging.error(f"Error summarizing text: {str(e)}")
        return text[:200] + "..."  # Fallback to truncated text

def extract_keywords(text):
    try:
        doc = nlp(text)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop 
                   and token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
        return list(set(keywords))[:10]  # Return top 10 unique keywords
    except Exception as e:
        logging.error(f"Error extracting keywords: {str(e)}")
        return []

# Train model on startup
model, vectorizer, metrics = train_model()

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
            
        # Scrape and process article
        text = scrape_article(url)
        if not text:
            return jsonify({'error': 'Failed to scrape article'}), 400
            
        # Vectorize text
        text_vec = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0][prediction]
        
        # Summarize and extract keywords
        summary = summarize_text(text)
        keywords = extract_keywords(text)
        
        response = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': float(probability),
            'summary': summary,
            'keywords': keywords
        }
        
        logging.info(f"Analysis completed for URL: {url}")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)