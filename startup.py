"""
MedAssist — Startup Script
Runs automatically on Streamlit Cloud to build FAISS index
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

def ensure_index_exists():
    """Build FAISS index on first run if not present."""
    index_path = ROOT / "data" / "faiss_index"

    if index_path.exists() and any(index_path.iterdir()):
        return  # Already built

    print("[STARTUP] FAISS index not found — building now...")
    try:
        from data.ingest import (
            load_builtin_guidelines,
            load_kaggle_csvs,
            load_pdf_files,
            load_txt_files,
            chunk_documents,
            load_embeddings,
            build_faiss_index,
        )
        docs   = load_builtin_guidelines() + load_kaggle_csvs() + load_pdf_files() + load_txt_files()
        chunks = chunk_documents(docs)
        emb    = load_embeddings()
        build_faiss_index(chunks, emb, force_rebuild=False)
        print("[STARTUP] FAISS index built successfully!")
    except Exception as e:
        print(f"[STARTUP] Index build failed: {e}")
        print("[STARTUP]  App will run with built-in guidelines only.")
