import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk dependencies
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -------------------------------
# Text Preprocessing Function
# -------------------------------
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# -------------------------------
# Load Trained Model and Vectorizer
# -------------------------------
@st.cache_resource  # Caches the model so it doesn't reload every time
def load_model():
    with open("news.pickle", "rb") as f:
        logistic, vectorizer = pickle.load(f)
    return logistic, vectorizer

logistic, vectorizer = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📰 News Category Predictor", layout="centered")

st.title("📰 News Category Classifier")
st.write("By Abdullah Ibrahim Dar")
st.write("Enter a news headline or short text to predict its category using a trained Logistic Regression model.")

# Text Input
user_input = st.text_area("Enter text:", placeholder="e.g., Charlie Kirk shot")

# Predict Button
if st.button("🔍 Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        cleaned_text = preprocess(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = logistic.predict(vectorized_text)
        st.success(f"**Predicted Category:** {prediction[0]}")
