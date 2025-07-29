# sentiment_model.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    tokens = tokenizer.tokenize(text.lower())
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(filtered)

def train_model():
    df = pd.read_csv('Tweets.csv')
    df = df[['text', 'airline_sentiment']].dropna()
    df = df[df['airline_sentiment'] != 'neutral']
    df['label'] = df['airline_sentiment'].map({'positive': 1, 'negative': 0})
    df['clean_text'] = df['text'].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def predict_sentiment(tweet):
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    cleaned = preprocess(tweet)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "Positive 😊" if pred == 1 else "Negative 😞"
