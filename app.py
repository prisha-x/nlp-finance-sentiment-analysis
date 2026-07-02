import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
