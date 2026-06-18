import streamlit as st
import sys
sys.path.append('.')

from src.signals import generate_signals
import pandas as pd
import joblib
from transformers import pipeline

st.title("NLP + Finance: Sentiment-Driven Trading Signals")

page = st.sidebar.selectbox("Navigate", ["Live Headline Analyzer", "Backtesting Results"])

if page == "Live Headline Analyzer":
    st.header("Live Headline Analyzer")
    headline = st.text_input("Enter a financial headline:")
    if headline:
        st.write("analyzing...")

elif page == "Backtesting Results":
    st.header("Backtesting Results")
    st.write("coming soon")