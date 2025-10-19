import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import easyocr
import cv2
import numpy as np
# -------------------------------
# NLTK Setup
# -------------------------------
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
@st.cache_resource
def load_model():
    with open("news.pickle", "rb") as f:
        logistic, vectorizer = pickle.load(f)
    return logistic, vectorizer

logistic, vectorizer = load_model()

# -------------------------------
# Initialize EasyOCR Reader (cached for performance)
# -------------------------------
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)  # Disable GPU for deployment safety

reader = load_ocr_reader()

# -------------------------------
# OCR Function (Extract Text from Image)
# -------------------------------
def extract_text_from_image(uploaded_image):
    # Convert the uploaded image file into a NumPy array
    image = Image.open(uploaded_image).convert("RGB")
    img_array = np.array(image)

    # EasyOCR expects BGR format like OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform OCR
    results = reader.readtext(img_bgr)

    # Join all detected text pieces
    extracted_text = " ".join([res[1] for res in results])
    return extracted_text.strip()

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="📰 News Category Predictor", layout="centered")

st.title("📰 News Category Classifier")
st.write(
    "Enter text manually **or upload an image** with text to predict its news category using a trained Logistic Regression model."
)

# --- Input Options ---
option = st.radio("Choose input method:", ("✍️ Enter Text", "🖼️ Upload Image"))

user_input = ""

if option == "✍️ Enter Text":
    user_input = st.text_area("Enter text:", placeholder="e.g., Charlie Kirk shot")

elif option == "🖼️ Upload Image":
    uploaded_image = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Extracting text from image..."):
            user_input = extract_text_from_image(uploaded_image)
        if user_input:
            st.text_area("Extracted Text:", user_input, height=150)
        else:
            st.warning("No readable text detected in the image.")

# --- Predict Button ---
if st.button("🔍 Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter or upload text to classify.")
    else:
        cleaned_text = preprocess(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = logistic.predict(vectorized_text)
        st.success(f"**Predicted Category:** {prediction[0]}")
