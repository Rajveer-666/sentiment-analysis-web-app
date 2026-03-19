from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def create_model():
    data = {
        'text': [
            'This movie was absolutely amazing! I loved every minute of it.',
            'Terrible film. The plot was boring and the acting was awful.',
            'What a wonderful experience! Highly recommend to everyone.',
            'I hated this movie. Waste of time and money.',
            'Outstanding performance by the entire cast. Truly inspiring.',
            'The worst movie I have ever seen. Do not watch it.',
            'A masterpiece of cinema. Beautifully crafted story.',
            'Boring and predictable. I fell asleep halfway through.',
            'Absolutely loved this! The best movie of the year.',
            'Disappointing and dull. Not worth watching at all.',
            'Heartwarming and funny. A perfect feel-good movie.',
            'A complete disaster. Poor writing and bad direction.',
            'Incredible! This movie exceeded all my expectations.',
            'Horrible. The storyline made no sense whatsoever.',
            'Brilliant acting and a captivating storyline.',
            'One of the most boring films I have ever watched.',
            'Loved the characters and the emotional depth.',
            'A terrible waste of time. Very frustrating to watch.',
            'Fantastic movie! Entertaining from start to finish.',
            'Awful movie. I want my money back.',
            'A beautiful story with great performances.',
            'This film was painful to sit through.',
            'Excellent direction and amazing cinematography.',
            'Poorly made. The script was terrible.',
            'I really enjoyed this movie. Well worth watching.',
            'Not recommended. Very confusing and poorly acted.',
            'Superb! One of the best films this decade.',
            'I regret watching this. Completely unenjoyable.',
            'A delightful movie that made me laugh and cry.',
            'Dreadful. Nothing redeeming about this film.',
            'Amazing visuals and an engaging plot.',
            'Boring from start to finish. Skip this one.',
            'Truly spectacular. A must-watch for everyone.',
            'Terrible acting and a weak storyline.',
            'Wonderful movie with an inspiring message.',
            'I cannot believe how bad this movie was.',
            'Phenomenal! This movie is a work of art.',
            'The most disappointing film of the year.',
            'A charming and entertaining experience.',
            'Utterly horrible. Do yourself a favor and skip.',
            'Stunning performances and a gripping narrative.',
            'A tedious and uninteresting movie.',
            'Remarkable storytelling and brilliant execution.',
            'Very poorly done. Lacked any real substance.',
            'An unforgettable cinematic experience.',
            'Completely forgettable and badly made.',
            'A gem of a movie. Watched it twice!',
            'The acting was cringe-worthy and awkward.',
            'Exceptional film. Thoroughly enjoyed every scene.',
            'A mess of a movie. Nothing worked.'
        ],
        'label': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative'
        ]
    }
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(data['text'], data['label'])
    return pipeline

def load_or_create_model():
    model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        model = create_model()
        joblib.dump(model, model_path)
        return model
