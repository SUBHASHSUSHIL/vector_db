"""
Chunking Examples for different data sources:
  1. Plain Text
  2. PDF
  3. Image (OCR -> text -> chunk)
  4. SQL Database
"""

import re
import numpy as np

# ─────────────────────────────────────────────────────────────
# HELPER: Cosine similarity (shared across all examples)
# ─────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_vocab(chunks):
    vocab = set()
    for chunk in chunks:
        for word in chunk.lower().split():
            vocab.add(re.sub(r'[^a-z0-9]', '', word))
    return sorted(vocab)


def embed(text, vocab):
    words = [re.sub(r'[^a-z0-9]', '', w) for w in text.lower().split()]
    return np.array([words.count(w) for w in vocab], dtype=float)


def search(query, chunks, vocab, top_k=2):
    chunk_vecs = [embed(c, vocab) for c in chunks]
    query_vec  = embed(query, vocab)
    scores = [(i, cosine_similarity(query_vec, cv)) for i, cv in enumerate(chunk_vecs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Query: '{query}'")
    for rank, (i, score) in enumerate(scores[:top_k], 1):
        print(f"  Rank {rank} | Chunk {i} | Score: {score:.4f}")
        print(f"  >> {chunks[i][:120]}...")


# ══════════════════════════════════════════════════════════════
# 1. PLAIN TEXT CHUNKING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. PLAIN TEXT CHUNKING")
print("=" * 60)

sample_text = """
Python is a high-level programming language known for simplicity.
It was created by Guido van Rossum and released in 1991.
Python supports multiple programming paradigms including procedural and object-oriented.

Python is widely used in web development, data science, and automation.
Popular frameworks include Django, Flask, and FastAPI for web development.
Libraries like NumPy, Pandas, and Matplotlib are used for data analysis.

Machine learning in Python is powered by scikit-learn and TensorFlow.
Deep learning frameworks like PyTorch are also very popular.
Python's ecosystem makes it the top choice for AI research.
"""

# --- Strategy A: Paragraph chunking ---
def chunk_by_paragraph(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

# --- Strategy B: Fixed-size with overlap ---
def chunk_fixed_size(text, size=100, overlap=20):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

# --- Strategy C: Sentence chunking ---
def chunk_by_sentences(text, n=3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [" ".join(sentences[i:i+n]) for i in range(0, len(sentences), n)]

para_chunks    = chunk_by_paragraph(sample_text)
fixed_chunks   = chunk_fixed_size(sample_text, size=30, overlap=5)
sentence_chunks = chunk_by_sentences(sample_text, n=2)

print(f"\n[Paragraph chunks]: {len(para_chunks)}")
for i, c in enumerate(para_chunks):
    print(f"  Chunk {i}: {c[:80]}...")

print(f"\n[Fixed-size chunks (30 words, overlap 5)]: {len(fixed_chunks)}")
for i, c in enumerate(fixed_chunks[:3]):
    print(f"  Chunk {i}: {c[:80]}...")

print(f"\n[Sentence chunks (2 sentences each)]: {len(sentence_chunks)}")
for i, c in enumerate(sentence_chunks[:3]):
    print(f"  Chunk {i}: {c[:80]}...")

vocab = build_vocab(para_chunks)
search("what is Python used for?", para_chunks, vocab)


# ══════════════════════════════════════════════════════════════
# 2. PDF CHUNKING  (requires: pip install pymupdf)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. PDF CHUNKING  (requires: pip install pymupdf)")
print("=" * 60)

try:
    import fitz  # PyMuPDF

    def chunk_pdf(pdf_path, chunk_size=300, overlap=50):
        """
        Extract text page-by-page from PDF, then apply
        fixed-size word chunking with overlap.
        """
        doc    = fitz.open(pdf_path)
        chunks = []
        for page_num, page in enumerate(doc):
            text  = page.get_text("text").strip()
            if not text:
                continue
            words = text.split()
            i = 0
            while i < len(words):
                chunk_words = words[i:i + chunk_size]
                chunks.append({
                    "page"  : page_num + 1,
                    "chunk" : " ".join(chunk_words)
                })
                i += chunk_size - overlap
        doc.close()
        return chunks

    # --- Demo (replace with your actual PDF path) ---
    PDF_PATH = "sample.pdf"
    pdf_chunks = chunk_pdf(PDF_PATH)
    print(f"\n  Total chunks from PDF: {len(pdf_chunks)}")
    for c in pdf_chunks[:3]:
        print(f"  Page {c['page']}: {c['chunk'][:100]}...")

    raw_chunks = [c["chunk"] for c in pdf_chunks]
    vocab_pdf  = build_vocab(raw_chunks)
    search("introduction", raw_chunks, vocab_pdf)

except ImportError:
    print("\n  [SKIPPED] pymupdf not installed.")
    print("  Install: pip install pymupdf")
    print("""
  Code preview:
    import fitz
    doc = fitz.open("file.pdf")
    for page in doc:
        text = page.get_text("text")
        # apply fixed-size or paragraph chunking on text
  """)
except FileNotFoundError:
    print("\n  [SKIPPED] sample.pdf not found — replace PDF_PATH with your file.")


# ══════════════════════════════════════════════════════════════
# 3. IMAGE CHUNKING  (OCR → text → chunk)
#    requires: pip install pillow pytesseract
#    system:   install Tesseract OCR from https://github.com/tesseract-ocr/tesseract
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. IMAGE CHUNKING  (OCR -> text -> chunk)")
print("=" * 60)

try:
    from PIL import Image
    import pytesseract

    def chunk_image(image_path, chunk_size=100, overlap=20):
        """
        1. Open image with Pillow
        2. Run OCR with Tesseract to extract text
        3. Apply fixed-size chunking on extracted text
        """
        img  = Image.open(image_path)

        # Optional pre-processing for better OCR accuracy
        img  = img.convert("L")          # grayscale
        # img = img.resize((img.width*2, img.height*2))  # upscale if small

        text  = pytesseract.image_to_string(img)
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i:i + chunk_size]))
            i += chunk_size - overlap
        return chunks

    IMAGE_PATH = "sample_image.png"
    img_chunks = chunk_image(IMAGE_PATH)
    print(f"\n  Total chunks from image: {len(img_chunks)}")
    for i, c in enumerate(img_chunks[:3]):
        print(f"  Chunk {i}: {c[:100]}...")

    vocab_img = build_vocab(img_chunks)
    search("main topic", img_chunks, vocab_img)

except ImportError:
    print("\n  [SKIPPED] pillow or pytesseract not installed.")
    print("  Install: pip install pillow pytesseract")
    print("""
  Code preview:
    from PIL import Image
    import pytesseract
    text = pytesseract.image_to_string(Image.open("image.png"))
    # apply fixed-size chunking on text
  """)
except FileNotFoundError:
    print("\n  [SKIPPED] sample_image.png not found — replace IMAGE_PATH with your file.")


# ══════════════════════════════════════════════════════════════
# 4. SQL DATABASE CHUNKING
#    No extra install needed — sqlite3 is built into Python
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. SQL DATABASE CHUNKING  (sqlite3 - built-in)")
print("=" * 60)

import sqlite3

# --- Create a demo in-memory database ---
def create_demo_db():
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE articles (
            id      INTEGER PRIMARY KEY,
            title   TEXT,
            content TEXT
        )
    """)
    rows = [
        (1, "Intro to AI",
         "Artificial intelligence is the simulation of human intelligence by machines. "
         "It includes machine learning, deep learning, and natural language processing. "
         "AI is used in healthcare, finance, and autonomous vehicles."),

        (2, "Python for Data Science",
         "Python is the most popular language for data science. "
         "Libraries like Pandas and NumPy make data manipulation easy. "
         "Visualization tools include Matplotlib and Seaborn."),

        (3, "Vector Databases",
         "Vector databases store high-dimensional embeddings for fast similarity search. "
         "Popular options include Pinecone, Weaviate, and ChromaDB. "
         "They are essential for RAG (Retrieval Augmented Generation) pipelines."),
    ]
    cur.executemany("INSERT INTO articles VALUES (?, ?, ?)", rows)
    conn.commit()
    return conn


def chunk_sql_rows(conn, table="articles", text_columns=None,
                   chunk_size=50, overlap=10):
    """
    Two strategies:
      A) Row-as-chunk  — each DB row becomes one chunk (good for short rows)
      B) Fixed-size    — split long row text into sub-chunks (good for long content)
    """
    if text_columns is None:
        text_columns = ["title", "content"]

    cur  = conn.cursor()
    cur.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    col_names = [d[0] for d in cur.description]

    chunks = []
    for row in rows:
        row_dict = dict(zip(col_names, row))

        # Combine selected text columns into one string
        full_text = " | ".join(
            str(row_dict[col]) for col in text_columns if col in row_dict
        )

        words = full_text.split()

        if len(words) <= chunk_size:
            # Strategy A: whole row is one chunk
            chunks.append({
                "source" : f"{table}[id={row_dict.get('id', '?')}]",
                "chunk"  : full_text
            })
        else:
            # Strategy B: split into sub-chunks
            i = 0
            part = 0
            while i < len(words):
                chunks.append({
                    "source" : f"{table}[id={row_dict.get('id', '?')}] part {part}",
                    "chunk"  : " ".join(words[i:i + chunk_size])
                })
                i    += chunk_size - overlap
                part += 1

    return chunks


conn       = create_demo_db()
sql_chunks = chunk_sql_rows(conn, table="articles",
                            text_columns=["title", "content"],
                            chunk_size=30, overlap=5)

print(f"\n  Total chunks from SQL: {len(sql_chunks)}")
for c in sql_chunks:
    print(f"  [{c['source']}] {c['chunk'][:100]}...")

raw_sql = [c["chunk"] for c in sql_chunks]
vocab_sql = build_vocab(raw_sql)
search("vector database embeddings", raw_sql, vocab_sql)

conn.close()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Which strategy for which source?")
print("=" * 60)
print("""
  Source        | Extract tool          | Chunk strategy
  ──────────────┼───────────────────────┼──────────────────────────
  Plain text    | built-in              | paragraph / sentence / fixed-size
  PDF           | pymupdf (fitz)        | page-by-page + fixed-size
  Image         | pytesseract + pillow  | OCR -> fixed-size
  SQL DB        | sqlite3 / sqlalchemy  | row-as-chunk or fixed-size
""")
