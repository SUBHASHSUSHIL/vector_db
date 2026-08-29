# 🗄️ Vector Database — Complete Guide

> A full explanation of what Vector DB is, how it works, and what this project contains.

---

## 📌 Table of Contents

1. [What is a Vector DB?](#1-what-is-a-vector-db)
2. [Why Use It?](#2-why-use-it)
3. [How It Works — Workflow](#3-how-it-works--workflow)
4. [Types of Vector Databases](#4-types-of-vector-databases)
5. [How Data is Stored](#5-how-data-is-stored)
6. [How to Get / Search Data](#6-how-to-get--search-data)
7. [Project Files](#7-project-files)
8. [Installation — What to Install](#8-installation--what-to-install)
9. [Environment Setup (.env)](#9-environment-setup-env)
10. [How to Run the Project](#10-how-to-run-the-project)

---

## 1. What is a Vector DB?

A traditional database (like MySQL or SQLite) stores **text, numbers, and dates** and searches using exact matches.

A **Vector Database** stores data as a **list of numbers (vectors/embeddings)** that capture the **meaning** of the content.

```
"King"   →  [0.5, 0.1, 0.3, ...]
"Queen"  →  [0.45, 0.15, 0.35, ...]
"Apple"  →  [0.9, 0.8, 0.1, ...]
```

> King and Queen will have **close** vectors because their meanings are related.
> Apple will have a **distant** vector because it belongs to a different category.

This similarity is calculated using **AI/ML models** — the process is called **embedding**.

---

## 2. Why Use It?

| Problem | Traditional DB | Vector DB |
|---|---|---|
| Search "what is machine learning?" | Needs exact word match | Matches by **meaning** |
| Find similar products | Apply category filters | Automatically finds similar items |
| Give context to a chatbot | Feed the entire document | Retrieve only relevant chunks |
| Find similar images | Not possible | ✅ Possible |

**Real-world use cases:**
- 🤖 **RAG (Retrieval Augmented Generation)** — systems like ChatGPT that use a custom knowledge base
- 🔍 **Semantic Search** — Google-style search that understands meaning
- 💊 **Medical Records** — finding patients with similar symptoms
- 🛍️ **E-commerce** — recommending similar products
- 📄 **Document QA** — "What is the salary mentioned in this document?" — answered via vector DB

---

## 3. How It Works — Workflow

```
Raw Data (Text / PDF / Image / DB / Video)
         │
         ▼
   ┌─────────────┐
   │  CHUNKING   │  ← Large document is split into small pieces
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │  EMBEDDING  │  ← Each chunk is converted to numbers using an AI model
   └─────────────┘   (e.g., all-MiniLM-L6-v2 → 384 numbers)
         │
         ▼
   ┌─────────────┐
   │   UPSERT    │  ← Store in Pinecone/ChromaDB (id + vector + metadata)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │    QUERY    │  ← Embed the user's question → find similar vectors
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │   RESULTS   │  ← Return Top-K most similar chunks
   └─────────────┘
```

### Step-by-step breakdown:

**Step 1 — Chunking:** Split the document into smaller pieces
```
"Python is a language. Python is used in data science..."
         ↓  (paragraph chunking)
Chunk 0: "Python is a language..."
Chunk 1: "Python is used in data science..."
```

**Step 2 — Embedding:** Convert each chunk into a vector using an AI model
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("Python is a language")
# → [0.12, -0.45, 0.78, ... 384 numbers]
```

**Step 3 — Store (Upsert):** Save to a Pinecone index
```python
index.upsert(vectors=[{
    "id": "chunk-0",
    "values": [0.12, -0.45, 0.78, ...],
    "metadata": {"text": "Python is a language...", "source": "doc.txt"}
}])
```

**Step 4 — Query:** Embed the user's question and find similar chunks
```python
query_vec = model.encode("What is Python?")
results = index.query(vector=query_vec, top_k=3, include_metadata=True)
```

**Step 5 — Results:** Results are ranked by Cosine Similarity (0 = completely different, 1 = identical)

---

## 4. Types of Vector Databases

### 🌐 Cloud / Managed (Hosted)

| Database | Company | Free Tier | Best For |
|---|---|---|---|
| **Pinecone** | Pinecone Inc. | ✅ 2GB free | Production RAG apps |
| **Weaviate Cloud** | Weaviate | ✅ Limited | Multi-modal search |
| **Qdrant Cloud** | Qdrant | ✅ 1GB free | High-performance |
| **Zilliz Cloud** | Zilliz | ✅ Available | Enterprise |

### 💻 Local / Self-Hosted

| Database | Install | Best For |
|---|---|---|
| **ChromaDB** | `pip install chromadb` | Local dev, prototyping |
| **FAISS** | `pip install faiss-cpu` | Fast in-memory search (by Meta) |
| **Qdrant** | Docker | Production self-hosted |
| **Weaviate** | Docker | Multi-modal local |
| **Milvus** | Docker | Large-scale enterprise |

### 📦 In-Memory / Lightweight

| Database | Best For |
|---|---|
| **FAISS** | Research, fast prototyping |
| **Annoy** | Recommendation systems (like Spotify) |
| **HNSWlib** | Pure Python implementation |

> **This project uses `Pinecone`** — because it is managed, easy to set up, and the free tier is sufficient for learning.

---

## 5. How Data is Stored

Every vector record contains **3 things**:

```python
{
    "id"      : "chunk-0",          # ← Unique identifier
    "values"  : [0.12, -0.45, ...], # ← 384-dim float vector (embedding)
    "metadata": {                   # ← Original data (used for filtering)
        "text"   : "Python is a language...",
        "source" : "python_docs.txt",
        "type"   : "text",
        "page"   : 1
    }
}
```

### Supported Data Sources (in this project):

| Source | Library | Chunking Strategy |
|---|---|---|
| 📄 Plain Text (.txt) | Built-in | Paragraph / Fixed-size / Sentence |
| 📑 PDF | `pymupdf` (fitz) | Page-by-page + Fixed-size |
| 🖼️ Image | `pillow` + `pytesseract` | OCR → Fixed-size |
| 🗃️ SQL Database | `sqlite3` | Row-as-chunk or Fixed-size |
| 🎥 Video | `moviepy` + `SpeechRecognition` | Audio transcript → Fixed-size |

### Chunking Strategies:

```python
# 1. Paragraph chunking — split on blank lines
chunks = text.split("\n\n")

# 2. Fixed-size with overlap — N words, with overlap
# "the cat sat on the mat" (size=3, overlap=1)
# → ["the cat sat", "sat on the", "the mat"]

# 3. Sentence chunking — N sentences per chunk
sentences = re.split(r'(?<=[.!?])\s+', text)
chunks = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
```

> **Why overlap?** — So that no important context is cut off at a chunk boundary.

---

## 6. How to Get / Search Data

### Basic Query:
```python
query = "what is machine learning?"
query_vector = model.encode(query).tolist()

results = index.query(
    vector           = query_vector,
    top_k            = 3,           # Top 3 similar chunks
    include_metadata = True         # Also return the original text
)

for match in results["matches"]:
    print(f"Score: {match['score']:.4f}")  # 1.0 = perfect match
    print(f"Text:  {match['metadata']['text']}")
```

### Query with Metadata Filter:
```python
# Only return results from PDF source
results = index.query(
    vector  = query_vector,
    top_k   = 3,
    filter  = {"type": {"$eq": "pdf"}}   # only PDF chunks
)

# Multiple filters
results = index.query(
    vector  = query_vector,
    top_k   = 5,
    filter  = {
        "source": {"$eq": "python_docs.txt"},
        "page"  : {"$gte": 5}             # page 5 or later
    }
)
```

### Understanding Similarity Scores:
```
Score 0.95+  →  Very similar (almost the same meaning)
Score 0.75+  →  Related topic
Score 0.50+  →  Loosely related
Score < 0.50 →  Unrelated topic
```

### Delete Operations:
```python
# Delete specific vectors
index.delete(ids=["chunk-0", "chunk-1"])

# Delete by filter (Pinecone)
index.delete(filter={"source": {"$eq": "old_doc.txt"}})

# Delete the entire index
pc.delete_index("index-name")
```

---

## 7. Project Files

```
vector_db/
├── .env                          ← API keys (PINECONE_API_KEY)
├── .gitignore                    ← Ignores .env and myenv
├── README.md                     ← This file
└── embeddings/
    ├── simple_embedding.py       ← Basics: word vectors + cosine similarity
    ├── pinecone_store.py         ← Pinecone store + query (full pipeline)
    ├── chunking_sources.py       ← Chunking demo for 4 sources (Text/PDF/Image/SQL)
    ├── multi_source_pinecone.py  ← All sources + Pinecone (production-ready)
    ├── requirements.txt          ← All dependencies
    ├── sample.txt                ← Demo text file
    └── sample.db                 ← Demo SQLite database
```

### What each file teaches:

| File | What You Learn |
|---|---|
| `simple_embedding.py` | What word embeddings are, cosine similarity, bag-of-words |
| `pinecone_store.py` | Connecting to Pinecone, upsert, query, metadata filtering |
| `chunking_sources.py` | How to create chunks from Text / PDF / Image / SQL |
| `multi_source_pinecone.py` | Everything together — complete RAG pipeline |

---

## 8. Installation — What to Install

### Prerequisites:
- Python 3.9+ must be installed
- Pinecone account (free): [https://app.pinecone.io](https://app.pinecone.io)

### Step 1 — Create a Virtual Environment:
```bash
# Go to the project folder
cd d:\vector_db

# Create virtual environment
python -m venv myenv

# Activate (Windows CMD)
myenv\Scripts\activate.bat

# Activate (Windows PowerShell)
myenv\Scripts\Activate.ps1

# Activate (Linux / Mac)
source myenv/bin/activate
```

### Step 2 — Install Python packages:
```bash
pip install -r embeddings/requirements.txt
```

### Package list and purpose:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.5.2 | Vector and math operations |
| `matplotlib` | 3.11.1 | Visualize embeddings |
| `pinecone` | 9.1.0 | Pinecone vector DB client |
| `sentence-transformers` | 3.4.1 | Real AI embeddings (384-dim) |
| `pymupdf` | 1.26.1 | Extract text from PDFs |
| `pillow` | 11.2.1 | Image processing |
| `pytesseract` | 0.3.13 | OCR — extract text from images |
| `python-dotenv` | 1.1.0 | Load API key from `.env` file |

### Step 3 — Install Tesseract OCR (only needed for image chunking):

**Windows:**
1. Download from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install (default path: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH or set in code:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### Step 4 — Video support (optional):
```bash
pip install moviepy SpeechRecognition
```

---

## 9. Environment Setup (.env)

The `.env` file already exists in the project root. Set your Pinecone API key inside it:

```
PINECONE_API_KEY=your-actual-api-key-here
```

**Where to get your API key:**
1. Go to [https://app.pinecone.io](https://app.pinecone.io)
2. Create a free account
3. Navigate to the "API Keys" section
4. Copy the key and paste it into `.env`

> ⚠️ **Important:** Never push the `.env` file to Git — it is already listed in `.gitignore`.

---

## 10. How to Run the Project

```bash
# Activate the virtual environment
myenv\Scripts\activate.bat

# 1. Learn the basics — word embeddings
python embeddings/simple_embedding.py

# 2. Pinecone pipeline — store + query
python embeddings/pinecone_store.py

# 3. Multi-source chunking demo
python embeddings/chunking_sources.py

# 4. Full production pipeline (text + SQL → Pinecone)
python embeddings/multi_source_pinecone.py
```

### Expected Output (pinecone_store.py):
```
[Step 1] Total chunks: 4
[Step 2] Creating embeddings (USE_REAL_EMBEDDINGS=False)
[Step 3] Connecting to Pinecone...
  Index 'vector-db-demo' created.
[Step 4] Upserting vectors to Pinecone...
  Total vectors stored: 4
[Step 5] Querying Pinecone...
  Query: 'what is natural language processing?'
  ID: chunk-2  Score: 0.9123  Text: Natural language processing...
[Done] All steps completed successfully!
```

---

## 🔑 Key Concepts Summary

| Concept | Meaning |
|---|---|
| **Embedding** | Converting text/image into a list of numbers |
| **Vector** | A list of numbers that represents meaning |
| **Dimension** | How many numbers are in a vector (e.g., 384) |
| **Cosine Similarity** | How "close" two vectors are (0–1 scale) |
| **Chunking** | Splitting a large document into smaller pieces |
| **Upsert** | Saving to a vector store (update + insert) |
| **Top-K** | Returning the K most similar results after a query |
| **Metadata** | Extra info stored with a vector (text, source, page) |
| **RAG** | Retrieval Augmented Generation — LLM + Vector DB |

---

## 📚 Learn More

- [Pinecone Docs](https://docs.pinecone.io)
- [Sentence Transformers](https://www.sbert.net)
- [LangChain + Vector DB](https://python.langchain.com/docs/integrations/vectorstores/)
- [ChromaDB (local alternative)](https://docs.trychroma.com)
