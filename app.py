import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import string
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer






# Preprocessing functions

def remove_digits(text): return text.translate(str.maketrans('', '', string.digits))
def remove_punc(text): return text.translate(str.maketrans('', '', string.punctuation))
def remove_special_characters(text): return re.sub(r'[^a-zA-Z0-9.,!?/:;"\'\s]', '', text)
def remove_urls(text): return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
def remove_html(text): return BeautifulSoup(text, 'html.parser').get_text()
def remove_stopwords(text): return ' '.join([w for w in text.split() if w not in stop_words])
def lemmatize_text(text): return ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
def preprocess_user_text(text):
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_punc(text)
    text = remove_digits(text)
    text = remove_stopwords(text)
    text = remove_special_characters(text)
    text = lemmatize_text(text)
    return text
def clean_texts(texts): return [preprocess_user_text(t) for t in texts]


# NLTK setup

@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    return True

setup_nltk()  # Call once at app start

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()



# Load model

@st.cache_resource
def load_model():
    return joblib.load("text_pipeline.pkl")

pipeline = load_model()
label_map = {0: "No Depression", 1: "Depression Risk"}


# Page layout

st.set_page_config(page_title="Mental Health Assessment", layout="wide")
st.title("🧠 Mental Health Assessment")

project_description = """
A mental wellness app using **Natural Language Processing (NLP) and Machine Learning (ML)** to analyze text, identify signs of depression, and provide early insights for emotional health monitoring.
"""


st.markdown(
    f"""
    <div style="
        background-color:#ACB2BD;
        color:#0D47A1;              
        padding:20px;
        border-radius:15px;
        box-shadow: 3px 3px 15px rgba(0,0,0,0.2);
        font-size:20px;
        margin-bottom:5px;
        line-height:1.6;
        transition: transform 0.2s;
    " onmouseover="this.style.transform='scale(1.02)';" onmouseout="this.style.transform='scale(1)';">
        {project_description}
    
    """,
    unsafe_allow_html=True
)


st.markdown("<p style='font-weight:bold; font-size:20px;margin-bottom:5px;'>Type your thoughts here:</p>", unsafe_allow_html=True)
user_input = st.text_area("Text", height=200,label_visibility="collapsed")


if st.button("Predict") and user_input.strip():
    
    # Prediction
    
    probs = pipeline.predict_proba([user_input])[0]
    classes = pipeline.classes_
    pred_idx = np.argmax(probs)
    pred_label_raw = classes[pred_idx]
    pred_label = label_map.get(pred_label_raw, str(pred_label_raw))
    confidence = probs[pred_idx]

    # Colored card prediction
    color = "#d73c3c" if pred_label_raw == 1 else "green"
    st.markdown(
        f"<div style='padding:20px;border-radius:12px;background-color:{color};"
        f"color:white;font-weight:bold;font-size:20px;text-align:center;'>"
        f"{pred_label} ({confidence*100:.2f}%)</div>",
        unsafe_allow_html=True,
    )

    
    # Interactive Confidence Bar
    
    st.subheader("Confidence Scores")
    conf_labels = [label_map.get(c, str(c)) for c in classes]
    conf_probs = [p*100 for p in probs]

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.barh(conf_labels, conf_probs, color=["#27afed" if "No" in l else "#f77c29" for l in conf_labels])
    ax.set_xlim(0, 100)
    for i, v in enumerate(conf_probs):
        ax.text(v + 1, i, f"{v:.2f}%", color='black', fontweight='bold')
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

    
    # LIME Explanation
    
    st.subheader("LIME Explanation")
    explainer = LimeTextExplainer(class_names=[label_map.get(c, str(c)) for c in classes])
    exp = explainer.explain_instance(user_input, pipeline.predict_proba, num_features=10)
    # Display as matplotlib figure (same professional notebook style)

    html_exp = exp.as_html()

    html_exp_colored = f"""
<div style="color:red;">
{html_exp}
</div>
"""

    st.components.v1.html(html_exp_colored, height=500)

    #fig2 = exp.as_pyplot_figure()
    #fig2.set_size_inches(20, 4)  # Adjust size to not be too big
    #st.pyplot(fig2)
