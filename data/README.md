# Data

Raw data isn't committed to this repo — `data/raw/` and `data/processed/` are gitignored.

To get the dataset:
1. Download Financial PhraseBank from HuggingFace using the command shown in `notebooks/01_eda.ipynb`
2. Place `all-data.csv` in `data/raw/`

Stock price data (AAPL) doesn't need manual downloading — `03_trading.ipynb` fetches it automatically via `yfinance`.