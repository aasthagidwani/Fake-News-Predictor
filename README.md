# Fake News Detector

## Project Overview
This is a web application for detecting fake news articles using machine learning and NLP techniques. It allows users to input a news article URL, scrapes the content, preprocesses the text, and predicts whether the article is real or fake. It also provides a summary and keywords for the article.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Virtual environment tool (e.g., virtualenv)

### Installation
1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Place the dataset**:
   - Ensure `dataset.xlsx` is in the project root directory.

### Running the Application
1. Start the Flask app:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`.

### Example Usage
- Input URL: `https://www.example.com/news-article`
- Output:
  - Prediction: Real/Fake
  - Confidence: Percentage
  - Summary: ~60-word summary of the article
  - Keywords: List of extracted keywords

### Project Structure
- `/models`: Stores the trained machine learning model
- `/static`: Contains CSS files
- `/templates`: Contains HTML templates
- `/utils`: Contains text preprocessing utilities
- `app.py`: Main Flask application
- `requirements.txt`: Project dependencies
- `README.md`: This file

### Notes
- The application uses Logistic Regression for classification
- Web scraping uses BeautifulSoup
- NLP tasks (summarization and keyword extraction) use spaCy and Sumy
- Ensure the dataset (`dataset.xlsx`) is available in the project root
