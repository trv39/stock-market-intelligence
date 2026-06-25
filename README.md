import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title = "Stock Market Intelligence",
    page_icon  = "📈",
    layout     = "wide",
)

DB_PATH       = "stock_intelligence.db"
FAISS_PATH    = "news_faiss.index"
METADATA_PATH = "news_metadata.pkl"

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss():
    index    = faiss.read_index(FAISS_PATH)
    metadata = pd.read_pickle(METADATA_PATH)
    return index, metadata

@st.cache_resource
def load_llm():
    return hf_pipeline(
        "text-generation",
        model          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device         = -1,
        max_new_tokens = 256,
        temperature    = 0.7,
        do_sample      = True,
    )

@st.cache_data
def load_stocks():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("""
        SELECT ticker, company, sector
        FROM stocks
        ORDER BY company ASC
    """, conn)
    conn.close()
    return df

@st.cache_data
def load_price_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("""
        SELECT p.date, p.open, p.high, p.low, p.close, p.volume
        FROM prices p
        JOIN stocks s ON s.stock_id = p.stock_id
        WHERE s.ticker = ?
        ORDER BY p.date ASC
    """, conn, params=(ticker,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_sentiment_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("""
        SELECT
            ss.date,
            AVG(ss.compound)  AS compound,
            AVG(ss.positive)  AS positive,
            AVG(ss.negative)  AS negative,
            COUNT(*)          AS news_count
        FROM sentiment_scores ss
        JOIN stocks s ON s.stock_id = ss.stock_id
        WHERE s.ticker = ?
        GROUP BY ss.date
        ORDER BY ss.date ASC
    """, conn, params=(ticker,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_recent_news(ticker: str, limit: int = 10) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("""
        SELECT
            n.headline,
            n.published_at,
            n.source,
            ss.label,
            ROUND(ss.compound, 3) AS compound
        FROM news n
        JOIN stocks s          ON s.stock_id  = n.stock_id
        LEFT JOIN sentiment_scores ss ON ss.news_id = n.news_id
        WHERE s.ticker = ?
        ORDER BY n.published_at DESC
        LIMIT ?
    """, conn, params=(ticker, limit))
    conn.close()
    return df

def rag_query(question: str, ticker: str,
              embedder, faiss_index, metadata,
              llm, top_k: int = 5) -> tuple[str, pd.DataFrame]:
    q_emb              = embedder.encode(
                             [question],
                             convert_to_numpy=True
                         ).astype(np.float32)
    distances, indices = faiss_index.search(q_emb, top_k * 3)
    hits               = metadata.iloc[indices[0]].copy()

    if ticker and ticker != "All":
        hits = hits[hits["ticker"] == ticker]
    hits = hits.head(top_k)

    context = "\n".join([
        f"[{row.published_at}] {row.ticker} — {row.headline} "
        f"(sentiment: {row.get('label','N/A')}, "
        f"score: {row.get('compound', 0):.2f})"
        for _, row in hits.iterrows()
    ])

    prompt = f"""<|system|>
You are a financial analyst AI specializing in Indian stock markets.
Answer questions about stock movements using the provided news context.
Be concise, specific, and mention relevant sentiment where applicable.
<|user|>
Context:
{context}

Question: {question}
<|assistant|>"""

    output = llm(prompt)[0]["generated_text"]
    answer = output.split("<|assistant|>")[-1].strip()
    return answer, hits

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/stocks.png",
    width=60,
)
st.sidebar.title("Stock Market Intelligence")
st.sidebar.markdown("*Sentiment-driven Forecasting with GenAI*")
st.sidebar.divider()

df_stocks  = load_stocks()
all_tickers = df_stocks["ticker"].tolist()
all_companies = df_stocks["company"].tolist()

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Stock Analysis", "AI Assistant", "About"],
    index=0,
)

# ── Page 1: Dashboard ────────────────────────────────────
if page == "Dashboard":
    st.title("📈 Stock Market Intelligence Dashboard")
    st.markdown("*96 NSE-listed stocks | 5 years data | FinBERT Sentiment | RAG Q&A*")

    conn = sqlite3.connect(DB_PATH)
    n_stocks = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    n_news   = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    n_sent   = conn.execute("SELECT COUNT(*) FROM sentiment_scores").fetchone()[0]
    conn.close()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks Tracked",    f"{n_stocks}")
    col2.metric("Price Records",     f"{n_prices:,}")
    col3.metric("News Headlines",    f"{n_news:,}")
    col4.metric("Sentiment Scores",  f"{n_sent:,}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Sector Distribution")
        conn     = sqlite3.connect(DB_PATH)
        df_sec   = pd.read_sql("""
            SELECT sector, COUNT(*) as count
            FROM stocks GROUP BY sector
            ORDER BY count DESC
        """, conn)
        conn.close()
        fig_pie = px.pie(
            df_sec, values="count", names="sector",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_layout(margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.subheader("Top 10 Stocks by Sentiment Score")
        conn    = sqlite3.connect(DB_PATH)
        df_sent = pd.read_sql("""
            SELECT s.company, s.ticker,
                   ROUND(AVG(ss.compound), 3) AS avg_sentiment
            FROM sentiment_scores ss
            JOIN stocks s ON s.stock_id = ss.stock_id
            GROUP BY s.ticker
            ORDER BY avg_sentiment DESC
            LIMIT 10
        """, conn)
        conn.close()
        fig_bar = px.bar(
            df_sent, x="avg_sentiment", y="company",
            orientation="h",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=0,b=0,l=0,r=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Model Performance Summary")
    df_perf = pd.DataFrame({
        "Model":        ["SARIMA", "XGBoost", "LSTM"],
        "Avg RMSE (%)": [6.53,     11.18,     9.62],
        "Type":         ["Statistical Baseline",
                         "ML + Sentiment + Technical",
                         "Deep Learning Sequential"],
        "Uses Sentiment": ["No", "Yes", "Yes"],
    })
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

# ── Page 2: Stock Analysis ───────────────────────────────
elif page == "Stock Analysis":
    st.title("📊 Stock Analysis")

    selected_company = st.selectbox(
        "Select Company",
        options=all_companies,
        index=0,
    )
    selected_ticker = df_stocks[
        df_stocks["company"] == selected_company
    ]["ticker"].values[0]

    df_price = load_price_data(selected_ticker)
    df_sent  = load_sentiment_data(selected_ticker)
    df_news  = load_recent_news(selected_ticker)

    if df_price.empty:
        st.warning("No price data available for this stock.")
    else:
        col1, col2, col3 = st.columns(3)
        latest  = df_price.iloc[-1]
        prev    = df_price.iloc[-2]
        chg     = latest["close"] - prev["close"]
        chg_pct = (chg / prev["close"]) * 100

        col1.metric("Latest Close",
                    f"₹{latest['close']:,.2f}",
                    f"{chg:+.2f} ({chg_pct:+.2f}%)")
        col2.metric("52W High",
                    f"₹{df_price['close'].tail(252).max():,.2f}")
        col3.metric("52W Low",
                    f"₹{df_price['close'].tail(252).min():,.2f}")

        st.subheader("Price Chart")
        period = st.radio(
            "Period", ["1Y", "2Y", "5Y"], horizontal=True
        )
        period_map = {"1Y": 252, "2Y": 504, "5Y": 1260}
        df_plot    = df_price.tail(period_map[period])

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x     = df_plot["date"],
            open  = df_plot["open"],
            high  = df_plot["high"],
            low   = df_plot["low"],
            close = df_plot["close"],
            name  = selected_ticker,
        ))
        fig.update_layout(
            xaxis_rangeslider_visible = False,
            height = 400,
            margin = dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        if not df_sent.empty:
            st.subheader("Sentiment Over Time")
            df_sent_plot = df_sent.tail(period_map[period])
            fig_sent     = go.Figure()
            fig_sent.add_trace(go.Bar(
                x    = df_sent_plot["date"],
                y    = df_sent_plot["compound"],
                name = "Sentiment",
                marker_color = df_sent_plot["compound"].apply(
                    lambda x: "green" if x > 0 else "red"
                ),
            ))
            fig_sent.update_layout(
                height = 200,
                margin = dict(t=10, b=10),
                yaxis_title = "Compound Score",
            )
            st.plotly_chart(fig_sent, use_container_width=True)

        st.subheader("Recent News & Sentiment")
        if not df_news.empty:
            for _, row in df_news.iterrows():
                color = ("🟢" if row.get("label") == "positive"
                         else "🔴" if row.get("label") == "negative"
                         else "⚪")
                st.markdown(
                    f"{color} **{row['headline']}**  \n"
                    f"*{row['published_at']} | {row['source']} | "
                    f"Score: {row['compound']}*"
                )

# ── Page 3: AI Assistant ─────────────────────────────────
elif page == "AI Assistant":
    st.title("🤖 AI Stock Assistant")
    st.markdown(
        "Ask any question about Indian stocks. "
        "The assistant retrieves relevant news and generates an explanation."
    )

    with st.spinner("Loading AI models — this takes 1-2 minutes on first load..."):
        embedder    = load_embedder()
        faiss_index, metadata = load_faiss()
        llm         = load_llm()

    st.success("AI Assistant ready!")

    ticker_options = ["All"] + all_tickers
    col1, col2     = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="Why did TCS stock drop recently?",
        )
    with col2:
        filter_ticker = st.selectbox(
            "Filter by stock",
            options=ticker_options,
            index=0,
        )

    example_questions = [
        "Why did HDFC Bank shares move recently?",
        "What is the overall sentiment around IT stocks?",
        "Which stocks have the most positive news?",
        "Why did Reliance Industries fluctuate?",
    ]
    st.markdown("**Try these:**")
    cols = st.columns(4)
    for i, eq in enumerate(example_questions):
        if cols[i].button(eq, use_container_width=True):
            question = eq

    if question:
        with st.spinner("Retrieving news and generating answer..."):
            answer, hits = rag_query(
                question, filter_ticker,
                embedder, faiss_index,
                metadata, llm,
            )

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources Used")
        for _, row in hits.iterrows():
            label   = row.get("label", "N/A")
            compound = row.get("compound", 0)
            color   = ("🟢" if label == "positive"
                       else "🔴" if label == "negative"
                       else "⚪")
            st.markdown(
                f"{color} **{row['headline']}**  \n"
                f"*{row['ticker']} | {row['published_at']} | "
                f"Sentiment: {label} ({compound:.2f})*"
            )

# ── Page 4: About ────────────────────────────────────────
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    ## Stock Market Intelligence: Sentiment-driven Forecasting with GenAI Insights

    ### Tech Stack
    | Component | Technology |
    |---|---|
    | Data Collection | yfinance, GDELT |
    | Database | SQLite (4 normalized tables) |
    | Sentiment Analysis | FinBERT |
    | Technical Indicators | ta library (25+) |
    | Time Series Analysis | ADF Test, Seasonal Decomposition |
    | Baseline Forecasting | SARIMA |
    | ML Forecasting | XGBoost |
    | Deep Learning | LSTM (2 layers, hidden=64) |
    | Vector Store | FAISS (CPU) |
    | Embeddings | all-MiniLM-L6-v2 |
    | LLM | TinyLlama 1.1B |
    | Orchestration | LangChain |
    | Frontend | Streamlit |

    ### Key Results
    - 96 NSE stocks tracked across 11 sectors
    - 5 years of historical OHLCV data
    - 13,440 news headlines scored with FinBERT
    - SARIMA: 6.53% normalized RMSE
    - XGBoost: 11.18% normalized RMSE
    - LSTM: 9.62% normalized RMSE
    - RAG system answers natural language queries about stock movements

    ### Project by
    *CS Student | Stock Market Intelligence Project*
    """)
