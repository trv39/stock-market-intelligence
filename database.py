import sqlite3
import pandas as pd

from config import DB_PATH


def get_connection():
    """
    Returns a SQLite connection.
    """
    return sqlite3.connect(DB_PATH)


def execute_query(query, params=None):
    """
    Execute any SQL query and return a DataFrame.
    """
    conn = get_connection()

    try:
        if params:
            df = pd.read_sql(query, conn, params=params)
        else:
            df = pd.read_sql(query, conn)

    finally:
        conn.close()

    return df


def get_database_summary():

    conn = get_connection()

    cursor = conn.cursor()

    summary = {

        "stocks": cursor.execute(
            "SELECT COUNT(*) FROM stocks"
        ).fetchone()[0],

        "prices": cursor.execute(
            "SELECT COUNT(*) FROM prices"
        ).fetchone()[0],

        "news": cursor.execute(
            "SELECT COUNT(*) FROM news"
        ).fetchone()[0],

        "sentiments": cursor.execute(
            "SELECT COUNT(*) FROM sentiment_scores"
        ).fetchone()[0],

    }

    conn.close()

    return summary


def get_all_stocks():

    query = """
    SELECT
        ticker,
        company,
        sector
    FROM stocks
    ORDER BY company
    """

    return execute_query(query)


def get_price_data(ticker):

    query = """
    SELECT

        p.date,
        p.open,
        p.high,
        p.low,
        p.close,
        p.volume

    FROM prices p

    JOIN stocks s

    ON p.stock_id = s.stock_id

    WHERE s.ticker = ?

    ORDER BY p.date
    """

    return execute_query(query, (ticker,))


def get_recent_news(ticker, limit=10):

    query = """
    SELECT

        n.headline,
        n.source,
        n.published_at,

        ss.label,
        ss.compound

    FROM news n

    JOIN stocks s

    ON s.stock_id = n.stock_id

    LEFT JOIN sentiment_scores ss

    ON ss.news_id = n.news_id

    WHERE s.ticker = ?

    ORDER BY n.published_at DESC

    LIMIT ?
    """

    return execute_query(query, (ticker, limit))
