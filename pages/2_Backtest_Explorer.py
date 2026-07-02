import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

st.title("Backtesting Results Explorer")
st.markdown("Comparing sentiment-driven strategy against buy-and-hold on AAPL 2020-2023.")

# cumulative returns chart
st.subheader("Cumulative Returns")
st.image(ROOT_DIR / "images" / "cumulative_returns.png")

# model comparison table
st.subheader("Model Comparison")
comparison = pd.DataFrame({
    'Model': ['VADER', 'LinearSVC', 'FinBERT'],
    'Weighted F1': [0.5499, 0.9238, 0.9009],
    'Negative F1': [0.2857, 0.8750, 0.9062],
    'Neutral F1': [0.6166, 0.9410, 0.9165],
    'Positive F1': [0.5202, 0.9085, 0.8673],
})

st.dataframe(comparison, hide_index=True)

st.subheader("Backtest Results")
col1, col2 = st.columns(2)
with col1:
    st.metric("Strategy Return", "22.85%")
    st.metric("Win Rate", "50.9%")
    st.metric("Number of Trades", "289")
with col2:
    st.metric("Buy-and-Hold Return", "148.41%")
    st.metric("High-Confidence Win Rate", "53.3%")
