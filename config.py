from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent

# Database & AI files
DB_PATH = BASE_DIR / "stock_intelligence.db"
FAISS_PATH = BASE_DIR / "news_faiss.index"
METADATA_PATH = BASE_DIR / "news_metadata.pkl"

# App
APP_NAME = "Stock Market Intelligence"
APP_ICON = "📈"
LAYOUT = "wide"
