# NLP + Finance: Sentiment-Driven Trading Signals

I did Inspirit AI earlier this year and we built a sentiment classifier. 
I kept thinking- if a model can read financial 
news and understand whether it's positive or negative, can it actually tell 
you something useful about what the market's going to do next?

This project is my attempt to answer that.

## The idea

Financial markets move on information. News breaks, sentiment shifts, 
prices react. The question I'm exploring: can NLP extract a clean enough 
signal from financial headlines to drive real trading decisions — and does 
a domain-specific model like FinBERT do this meaningfully better than a 
traditional TF-IDF approach?

## What this builds toward

- A preprocessing and classification pipeline for financial news sentiment
- Multiple ML models compared rigorously (not just accuracy — weighted F1
  given class imbalance in the dataset)
- Sentiment scores mapped to BUY / HOLD / SELL signals
- Backtesting those signals against Buy-and-Hold on real stock data
- FinBERT vs traditional ML comparison
- An interactive Streamlit dashboard to explore the results

## Stack

Python · NLTK · Scikit-learn · HuggingFace Transformers ·
yfinance · Matplotlib · Seaborn · Streamlit

## Dataset

Financial PhraseBank (Malo et al., 2014) — ~4,800 financial headlines
annotated by domain experts. Supplemented with real price data via yfinance.

## Structure

- `data/` — raw and processed datasets
- `notebooks/` — one notebook per section
- `src/` — reusable Python modules
- `images/` — charts and visualizations
- `results/` — model outputs and metrics
- `models/` — saved trained models

## How to run

```bash
git clone https://github.com/prisha-x/nlp-finance-sentiment-analysis
cd nlp-finance-sentiment-analysis
pip install -r requirements.txt
```

Dataset downloads automatically via HuggingFace when you run
the first notebook.

## Status

Work in progress. Started May 2026.
Results, findings, and honest limitations will be documented here
as the project develops.


*Extended independently from my Inspirit AI Scholars Program project
(March 2026). The program got me started — the questions it left
unanswered are what actually got me building.*
