"""
Multi-Source Chunking + Pinecone Vector DB
==========================================
Sources covered:
  1. Plain Text
  2. PDF
  3. Image  (OCR)
  4. SQL Database
  5. Video  (audio transcript via speech recognition)

Pattern from RAG project:
  load -> chunk (RecursiveCharacterTextSplitter style) -> embed (SentenceTransformer) -> Pinecone upsert
"""

import os
import re
import sqlite3
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# ENV + CONFIG
# ─────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME       = "multi-source-rag"
DIMENSION        = 384          # all-MiniLM-L6-v2
CHUNK_SIZE       = 200          # words per chunk
CHUNK_OVERLAP    = 40           # overlap words

# ─────────────────────────────────────────────
# EMBEDDING MODEL  (same as RAG project)
# ─────────────────────────────────────────────
print("[Init] Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list:
    return model.encode(text).tolist()


# ─────────────────────────────────────────────
# SHARED: Fixed-size chunker with overlap
# (same logic as LangChain RecursiveCharacterTextSplitter)
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ══════════════════════════════════════════════
# SOURCE 1 — PLAIN TEXT
# ══════════════════════════════════════════════
def load_text(file_path: str) -> list:
    """Read a .txt file and return chunks with metadata."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    return [
        {
            "id"      : f"text_{i}",
            "text"    : chunk,
            "source"  : os.path.basename(file_path),
            "type"    : "text",
            "chunk_no": i
        }
        for i, chunk in enumerate(chunks)
    ]


# ══════════════════════════════════════════════
# SOURCE 2 — PDF
# ══════════════════════════════════════════════
def load_pdf(file_path: str) -> list:
    """Extract text page-by-page using PyMuPDF, then chunk."""
    try:
        import fitz  # pymupdf
    except ImportError:
        print("  [PDF] pymupdf not installed. Run: pip install pymupdf")
        return []

    doc    = fitz.open(file_path)
    result = []
    chunk_no = 0

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text").strip()
        if not page_text:
            continue

        chunks = chunk_text(page_text)
        for chunk in chunks:
            result.append({
                "id"      : f"pdf_{chunk_no}",
                "text"    : chunk,
                "source"  : os.path.basename(file_path),
                "type"    : "pdf",
                "page"    : page_num + 1,
                "chunk_no": chunk_no
            })
            chunk_no += 1

    doc.close()
    return result


# ══════════════════════════════════════════════
# SOURCE 3 — IMAGE  (OCR → chunk)
# ══════════════════════════════════════════════
def load_image(file_path: str) -> list:
    """OCR the image with pytesseract, then chunk the extracted text."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        print("  [Image] Install: pip install pillow pytesseract")
        return []

    img  = Image.open(file_path).convert("L")  # grayscale for better OCR
    text = pytesseract.image_to_string(img).strip()

    if not text:
        print(f"  [Image] No text extracted from {file_path}")
        return []

    chunks = chunk_text(text)
    return [
        {
            "id"      : f"img_{i}",
            "text"    : chunk,
            "source"  : os.path.basename(file_path),
            "type"    : "image",
            "chunk_no": i
        }
        for i, chunk in enumerate(chunks)
    ]


# ══════════════════════════════════════════════
# SOURCE 4 — SQL DATABASE
# ══════════════════════════════════════════════
def load_sql(db_path: str, table: str, text_columns: list) -> list:
    """
    Each DB row → combine text columns → chunk if long.
    Pattern from RAG/6_sql_rag/sql_rag_project.py
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(f"SELECT * FROM {table}")
    rows     = cur.fetchall()
    col_names = [d[0] for d in cur.description]
    conn.close()

    result   = []
    chunk_no = 0

    for row in rows:
        row_dict  = dict(zip(col_names, row))
        full_text = " | ".join(
            str(row_dict[col]) for col in text_columns if col in row_dict
        )
        chunks = chunk_text(full_text)
        for chunk in chunks:
            result.append({
                "id"      : f"sql_{chunk_no}",
                "text"    : chunk,
                "source"  : f"{os.path.basename(db_path)}.{table}",
                "type"    : "sql",
                "row_id"  : row_dict.get("id", chunk_no),
                "chunk_no": chunk_no
            })
            chunk_no += 1

    return result


# ══════════════════════════════════════════════
# SOURCE 5 — VIDEO  (audio → transcript → chunk)
# ══════════════════════════════════════════════
def load_video(file_path: str) -> list:
    """
    Extract audio from video → transcribe with SpeechRecognition → chunk.
    Requires: pip install moviepy SpeechRecognition
    """
    try:
        from moviepy import VideoFileClip
        import speech_recognition as sr
        import tempfile
    except ImportError:
        print("  [Video] Install: pip install moviepy SpeechRecognition")
        return []

    print(f"  [Video] Extracting audio from {file_path}...")
    clip       = VideoFileClip(file_path)
    audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        transcript = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        print("  [Video] Could not understand audio")
        return []
    except sr.RequestError as e:
        print(f"  [Video] Speech recognition error: {e}")
        return []
    finally:
        os.remove(audio_path)

    chunks = chunk_text(transcript)
    return [
        {
            "id"      : f"video_{i}",
            "text"    : chunk,
            "source"  : os.path.basename(file_path),
            "type"    : "video",
            "chunk_no": i
        }
        for i, chunk in enumerate(chunks)
    ]


# ══════════════════════════════════════════════
# PINECONE — connect + create index
# ══════════════════════════════════════════════
def connect_pinecone():
    print("\n[Pinecone] Connecting...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"[Pinecone] Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name      = INDEX_NAME,
            dimension = DIMENSION,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"[Pinecone] Index '{INDEX_NAME}' already exists.")

    index = pc.Index(INDEX_NAME)
    print(f"[Pinecone] Connected. Stats: {index.describe_index_stats()}")
    return index


# ══════════════════════════════════════════════
# UPSERT — embed + store in batches
# ══════════════════════════════════════════════
def upsert_to_pinecone(index, records: list, batch_size=100):
    """
    records = list of dicts with keys: id, text, source, type, ...
    Embeds each chunk and upserts with full metadata.
    """
    vectors = []
    for rec in records:
        embedding = get_embedding(rec["text"])
        vectors.append({
            "id"     : rec["id"],
            "values" : embedding,
            "metadata": {k: v for k, v in rec.items() if k not in ("id",)}
        })

    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        total += len(batch)
        print(f"  Upserted {total}/{len(vectors)} vectors")

    return len(vectors)


# ══════════════════════════════════════════════
# QUERY — similarity search
# ══════════════════════════════════════════════
def query_pinecone(index, query: str, top_k=3, source_type=None):
    """
    Search Pinecone. Optionally filter by source type
    (text | pdf | image | sql | video).
    """
    query_vec = get_embedding(query)
    filter_   = {"type": {"$eq": source_type}} if source_type else None

    results = index.query(
        vector          = query_vec,
        top_k           = top_k,
        include_metadata= True,
        filter          = filter_
    )

    print(f"\n[Query] '{query}'" + (f"  [filter: type={source_type}]" if source_type else ""))
    for m in results["matches"]:
        print(f"  ID={m['id']}  score={m['score']:.4f}  type={m['metadata'].get('type')}  source={m['metadata'].get('source')}")
        print(f"  >> {m['metadata'].get('text','')[:120]}...")
    return results


# ══════════════════════════════════════════════
# DEMO — creates sample files and runs pipeline
# ══════════════════════════════════════════════
def create_demo_files():
    """Create small sample files to demo every source type."""
    base = os.path.dirname(__file__)

    # --- text ---
    txt_path = os.path.join(base, "sample.txt")
    with open(txt_path, "w") as f:
        f.write("""Artificial intelligence is the simulation of human intelligence.
Machine learning is a subset of AI that learns from data.
Deep learning uses neural networks with many layers.
Natural language processing allows machines to understand text.
Computer vision enables machines to interpret images and videos.""")

    # --- sql (in-memory exported to file) ---
    db_path = os.path.join(base, "sample.db")
    conn    = sqlite3.connect(db_path)
    cur     = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS articles")
    cur.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, content TEXT)")
    cur.executemany("INSERT INTO articles VALUES (?,?,?)", [
        (1, "Vector Databases",  "Vector databases store embeddings for fast similarity search. Pinecone is a popular managed vector DB."),
        (2, "RAG Systems",       "Retrieval Augmented Generation combines retrieval and generation for accurate answers."),
        (3, "LangChain",         "LangChain is a framework for building LLM-powered applications with chains and agents."),
    ])
    conn.commit()
    conn.close()

    return txt_path, db_path


def main():
    print("=" * 60)
    print("Multi-Source Chunking + Pinecone Pipeline")
    print("=" * 60)

    # Create demo sample files
    txt_path, db_path = create_demo_files()

    # Connect Pinecone
    index = connect_pinecone()

    all_records = []

    # ── 1. Text ──────────────────────────────
    print("\n[1] Loading TEXT...")
    records = load_text(txt_path)
    print(f"    Chunks: {len(records)}")
    all_records.extend(records)

    # ── 2. PDF ───────────────────────────────
    pdf_path = os.path.join(os.path.dirname(__file__), "sample.pdf")
    if os.path.exists(pdf_path):
        print("\n[2] Loading PDF...")
        records = load_pdf(pdf_path)
        print(f"    Chunks: {len(records)}")
        all_records.extend(records)
    else:
        print("\n[2] PDF skipped — place 'sample.pdf' in embeddings/ folder")

    # ── 3. Image ─────────────────────────────
    img_path = os.path.join(os.path.dirname(__file__), "sample_image.png")
    if os.path.exists(img_path):
        print("\n[3] Loading IMAGE (OCR)...")
        records = load_image(img_path)
        print(f"    Chunks: {len(records)}")
        all_records.extend(records)
    else:
        print("\n[3] Image skipped — place 'sample_image.png' in embeddings/ folder")

    # ── 4. SQL ───────────────────────────────
    print("\n[4] Loading SQL DB...")
    records = load_sql(db_path, table="articles", text_columns=["title", "content"])
    print(f"    Chunks: {len(records)}")
    all_records.extend(records)

    # ── 5. Video ─────────────────────────────
    vid_path = os.path.join(os.path.dirname(__file__), "sample_video.mp4")
    if os.path.exists(vid_path):
        print("\n[5] Loading VIDEO (speech-to-text)...")
        records = load_video(vid_path)
        print(f"    Chunks: {len(records)}")
        all_records.extend(records)
    else:
        print("\n[5] Video skipped — place 'sample_video.mp4' in embeddings/ folder")

    # ── Upsert all ───────────────────────────
    print(f"\n[Upsert] Total records to store: {len(all_records)}")
    total = upsert_to_pinecone(index, all_records)
    print(f"[Upsert] Done. {total} vectors stored in Pinecone.")
    print(f"[Stats]  {index.describe_index_stats()}")

    # ── Query examples ───────────────────────
    print("\n" + "=" * 60)
    print("QUERY EXAMPLES")
    print("=" * 60)

    query_pinecone(index, "what is machine learning?")
    query_pinecone(index, "vector database similarity search", source_type="sql")
    query_pinecone(index, "neural networks deep learning",    source_type="text")

    print("\n[Done] Pipeline complete!")


if __name__ == "__main__":
    main()
