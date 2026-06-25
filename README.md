# 📈 Stock Market Intelligence: Sentiment-driven Forecasting with GenAI Insights

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Stocks](https://img.shields.io/badge/NSE%20Stocks-96-orange)
![Models](https://img.shields.io/badge/Models-SARIMA%20%7C%20XGBoost%20%7C%20LSTM-purple)

## 🔗 Live Demo
👉 [Click here to try the AI Stock Assistant](https://your-app.streamlit.app)

## 📌 Project Overview
An end-to-end stock market intelligence system that combines:
- **Financial news sentiment analysis** using FinBERT
- **Technical indicators** (25+) for feature engineering
- **Three forecasting models** — SARIMA, XGBoost, LSTM
- **RAG-based Q&A system** — "Why did this stock move today?"

Built entirely to run on **CPU** — no GPU required.

---

## 🏗️ Architecture
yfinance (5Y OHLCV) ──┐

├──► SQLite DB (4 tables) ──► Technical Indicators ──► XGBoost / LSTM

GDELT News Headlines ──┤──► SARIMA

└──► FinBERT Sentiment ─────► FAISS Vector Store ──► TinyLlama RAG

## 📊 Results

| Model | Avg Normalized RMSE | Uses Sentiment |
|---|---|---|
| SARIMA | 6.53% | No |
| LSTM | 9.62% | Yes |
| XGBoost | 11.18% | Yes |

---

## 🗄️ Database Schema

```sql
stocks            — ticker, company, sector, exchange
prices            — OHLCV data (1,23,082 rows)
news              — headlines, source, published_at (13,440 rows)
sentiment_scores  — positive, negative, neutral, compound, label (12,960 rows)
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Data Collection | yfinance, GDELT |
| Database | SQLite |
| Sentiment | FinBERT (ProsusAI) |
| Technical Indicators | ta library |
| Time Series Analysis | ADF Test, Seasonal Decomposition, ACF/PACF |
| Forecasting | SARIMA, XGBoost, LSTM (PyTorch) |
| RAG System | FAISS + all-MiniLM-L6-v2 + TinyLlama 1.1B |
| Frontend | Streamlit |

---

## 📁 Project Structure
stock-market-intelligence/

├── notebooks/
│   └── full_pipeline.ipynb       ← Complete Kaggle notebook
├── app.py                        ← Streamlit web app
├── requirements.txt
├── assets/
│   ├── tcs_decomposition.png
│   ├── rmse_comparison.png
│   └── rmse_comparison_normalized.png
└── README.md
