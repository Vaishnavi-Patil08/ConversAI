import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT=Path(__file__).parent
DATA_DIR=ROOT/"data"
DATA_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH=DATA_DIR/"faiss_index.idx"
MEMORY_TEXTS_PATH=DATA_DIR/"memory_texta.json"
SQLITE_DB=DATA_DIR/"assistant.db"

EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2")
LLM_MODEL=os.getenv("LLM_MODEL","gemini-2.0-flash")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY=os.getenv("NEWS_API_KEY")
SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY=os.getenv("SUPABASE_ANON_KEY")