# Sentiment Analysis Web App

A web-based sentiment analysis application built with Machine Learning (scikit-learn) and a modern frontend.

## Features

- **ML-Powered Sentiment Analysis**: Classifies text as Positive, Negative, or Neutral
- **Clean Web Interface**: Modern, responsive UI built with HTML, CSS, and JavaScript
- **FastAPI Backend**: High-performance Python backend serving the ML model
- **Real-time Predictions**: Instant sentiment analysis with confidence scores

## Project Structure

```
sentiment-analysis-web-app/
├── app.py              # FastAPI backend server
├── model.py            # ML model pipeline and training
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── static/
    ├── index.html      # Frontend HTML
    ├── style.css       # Styles and animations
    └── index.js        # Frontend JavaScript logic
```

## Technologies Used

- **Backend**: Python, FastAPI, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API)
- **ML Model**: TF-IDF Vectorizer + Logistic Regression
- **Dataset**: IMDB Movie Reviews

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rajveer-666/sentiment-analysis-web-app.git
cd sentiment-analysis-web-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python model.py
```

### 4. Run the server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port $PORT
```

### 5. Open in browser

Navigate to `http://localhost:8000`

## How It Works

1. The user enters text in the web interface
2. The JavaScript sends the text to the FastAPI backend via Fetch API
3. The backend uses the trained ML model to predict sentiment
4. The result (Positive/Negative/Neutral with confidence) is displayed

## API Endpoint

- **POST /predict**: Sends text for sentiment analysis
  - Request: `{"text": "your input text"}`
  - Response: `{"sentiment": "Positive/Negative/Neutral", "confidence": 0.85}`

## License

MIT License
