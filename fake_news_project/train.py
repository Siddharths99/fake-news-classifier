import pandas as pd
import re, string, nltk, joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

#Load dataset
fake = pd.read_csv("datasets/Fake.csv")
true = pd.read_csv("datasets/True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

#Clean text
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

df["clean"] = df["text"].apply(clean_text)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

#TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Logistic Regression
lr = LogisticRegression(max_iter=500)
lr.fit(X_train_tfidf, y_train)
pred = lr.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

#Save model & vectorizer
os.makedirs("outputs", exist_ok=True)
joblib.dump(vectorizer, "outputs/tfidf.joblib")
joblib.dump(lr, "outputs/logreg.joblib")

print("Model and vectorizer saved in outputs/")
