# Fake News Detector

## Project Overview
This is a web application for detecting fake news articles using machine learning and NLP techniques. It allows users to input a news article URL, scrapes the content, preprocesses the text, and predicts whether the article is real or fake. It also provides a summary and keywords for the article.


### Example Usage
[Watch the demo video](./sample_test_video.mov)

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
