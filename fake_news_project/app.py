import streamlit as st
import joblib
import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model & vectorizer
vectorizer = joblib.load("outputs/tfidf.joblib")
model = joblib.load("outputs/logreg.joblib")

# NLTK setup
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Streamlit UI
st.title("📰 Fake News Classifier")
text = st.text_area("Paste a news article")

if st.button("Classify"):
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    
    st.subheader(f"Prediction: {pred}")
    
    # Show probability (confidence)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        st.write(f"Confidence → FAKE: {prob[0]*100:.2f}% | REAL: {prob[1]*100:.2f}%")
