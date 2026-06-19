import streamlit as st
import sys
sys.path.append('.')

from src.signals import get_signal
from transformers import pipeline
import joblib

st.title("Live Headline Analyzer")
st.markdown("Enter a financial headline to get a sentiment classification and trading signal.")

@st.cache_resource
def load_models():
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    lsvc = joblib.load('models/linearsvc_model.pkl')
    finbert = pipeline("text-classification", model="ProsusAI/finbert")
    return vectorizer, lsvc, finbert

vectorizer, lsvc, finbert = load_models()

headline = st.text_input("Enter a financial headline:")

if headline:
    from src.preprocess import preprocess
    cleaned = preprocess(headline)
    X = vectorizer.transform([cleaned])
    lsvc_pred = lsvc.predict(X)[0]

    fb_result = finbert(headline[:512])[0]
    fb_label = fb_result['label'].lower()
    fb_score = fb_result['score']

    if fb_label == 'positive':
        score = fb_score
    elif fb_label == 'negative':
        score = -fb_score
    else:
        score = 0.0

    signal = get_signal(score)

    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("LinearSVC", lsvc_pred.upper())
        st.metric("FinBERT", fb_label.upper())
        st.metric("Confidence", f"{fb_score:.1%}")

    with col2:
        if signal == 'BUY':
            st.success("🟢 BUY")
        elif signal == 'SELL':
            st.error("🔴 SELL")
        else:
            st.info("⚪ HOLD")

        if signal == 'BUY':
            st.caption("High positive sentiment detected — historically, high-confidence signals like this led to positive returns 53.3% of the time in our test data.")
        elif signal == 'SELL':
            st.caption("High negative sentiment detected — a signal like this historically preceded downward price movement in our test data.")
        else:
            st.caption("Sentiment isn't strong enough either way to generate a confident trading signal.")