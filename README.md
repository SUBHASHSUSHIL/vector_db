# 🗄️ Vector Database — Complete Guide

> **Urdu/Hindi mein samjhaya gaya hai** — Vector DB kya hai, kaise kaam karta hai, aur is project mein kya kuch hai.

---

## 📌 Table of Contents

1. [Vector DB Kya Hai?](#1-vector-db-kya-hai)
2. [Kyun Use Karte Hain?](#2-kyun-use-karte-hain)
3. [Kaise Kaam Karta Hai — Workflow](#3-kaise-kaam-karta-hai--workflow)
4. [Vector DB ke Types](#4-vector-db-ke-types)
5. [Data Kaise Store Hota Hai](#5-data-kaise-store-hota-hai)
6. [Data Kaise Get/Search Karte Hain](#6-data-kaise-getsearch-karte-hain)
7. [Is Project ki Files](#7-is-project-ki-files)
8. [Installation — Kya Kya Install Karna Hai](#8-installation--kya-kya-install-karna-hai)
9. [Environment Setup (.env)](#9-environment-setup-env)
10. [Project Run Kaise Karo](#10-project-run-kaise-karo)

---

## 1. Vector DB Kya Hai?

Normal database (jaise MySQL ya SQLite) mein **text, numbers, dates** store hote hain aur hum exact match se search karte hain.

**Vector Database** mein data **numbers ki list (vectors/embeddings)** ke form mein store hota hai — jisme **meaning** capture hoti hai.

```
"King"   →  [0.5, 0.1, 0.3, ...]
"Queen"  →  [0.45, 0.15, 0.35, ...]
"Apple"  →  [0.9, 0.8, 0.1, ...]
```

> King aur Queen ke vectors **kareebi** (similar) honge kyunki unka matlab related hai.
> Apple ka vector **door** hoga kyunki wo alag category hai.

Yeh similarity **AI/ML models** use karke calculate ki jaati hai — ise **embedding** kehte hain.

---

## 2. Kyun Use Karte Hain?

| Problem | Normal DB | Vector DB |
|---|---|---|
| "Machine learning kya hai?" search karo | Exact word match chahiye | **Meaning** se match karega |
| Similar products dhundna | Category filter lagao | Automatically similar items mile |
| Chatbot ko context do | Poora document de do | Sirf relevant chunks retrieve karo |
| Image se similar images dhundo | Not possible | ✅ Possible |

**Real-world use cases:**
- 🤖 **RAG (Retrieval Augmented Generation)** — ChatGPT jaise systems jo apna knowledge base use karte hain
- 🔍 **Semantic Search** — Google jaise search jo meaning samjhe
- 💊 **Medical Records** — Similar symptoms wale patients dhundna
- 🛍️ **E-commerce** — Similar products recommend karna
- 📄 **Document QA** — "Is document mein salary kya hai?" — vector DB se answer mile

---

## 3. Kaise Kaam Karta Hai — Workflow

```
Raw Data (Text/PDF/Image/DB/Video)
         │
         ▼
   ┌─────────────┐
   │  CHUNKING   │  ← Bada document chhote pieces mein toda jaata hai
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │  EMBEDDING  │  ← Har chunk ko AI model se numbers mein badla jaata hai
   └─────────────┘   (e.g., all-MiniLM-L6-v2 → 384 numbers)
         │
         ▼
   ┌─────────────┐
   │   UPSERT    │  ← Pinecone/ChromaDB mein store karo (id + vector + metadata)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │    QUERY    │  ← User ka sawaal bhi embed karo → similar vectors dhundo
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │   RESULTS   │  ← Top-K most similar chunks return karo
   └─────────────┘
```

### Step-by-step detail:

**Step 1 — Chunking:** Document ko chhote chhote pieces mein toda jaata hai
```
"Python ek language hai. Python data science mein use hoti hai..."
         ↓  (paragraph chunking)
Chunk 0: "Python ek language hai..."
Chunk 1: "Python data science mein..."
```

**Step 2 — Embedding:** Har chunk ko AI model se vector banao
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("Python ek language hai")
# → [0.12, -0.45, 0.78, ... 384 numbers]
```

**Step 3 — Store (Upsert):** Pinecone index mein save karo
```python
index.upsert(vectors=[{
    "id": "chunk-0",
    "values": [0.12, -0.45, 0.78, ...],
    "metadata": {"text": "Python ek language hai...", "source": "doc.txt"}
}])
```

**Step 4 — Query:** User ka sawaal embed karo aur similar chunks dhundo
```python
query_vec = model.encode("Python kya hai?")
results = index.query(vector=query_vec, top_k=3, include_metadata=True)
```

**Step 5 — Results:** Cosine Similarity se ranked results aate hain (0 = bilkul alag, 1 = same)

---

## 4. Vector DB ke Types

### 🌐 Cloud/Managed (Hosted)

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
| **FAISS** | `pip install faiss-cpu` | Fast in-memory search (Meta) |
| **Qdrant** | Docker | Production self-hosted |
| **Weaviate** | Docker | Multi-modal local |
| **Milvus** | Docker | Large-scale enterprise |

### 📦 In-Memory / Lightweight

| Database | Best For |
|---|---|
| **FAISS** | Research, fast prototyping |
| **Annoy** | Spotify jaise recommendation |
| **HNSWlib** | Pure Python implementation |

> **Is project mein `Pinecone` use kiya gaya hai** — kyunki yeh managed hai, setup asaan hai, aur free tier kaafi hai.

---

## 5. Data Kaise Store Hota Hai

Har vector record mein **3 cheezein** hoti hain:

```python
{
    "id"      : "chunk-0",          # ← Unique identifier
    "values"  : [0.12, -0.45, ...], # ← 384-dim float vector (embedding)
    "metadata": {                   # ← Original data (filtering ke liye)
        "text"   : "Python ek language hai...",
        "source" : "python_docs.txt",
        "type"   : "text",
        "page"   : 1
    }
}
```

### Supported Data Sources (is project mein):

| Source | Library | Chunking Strategy |
|---|---|---|
| 📄 Plain Text (.txt) | Built-in | Paragraph / Fixed-size / Sentence |
| 📑 PDF | `pymupdf` (fitz) | Page-by-page + Fixed-size |
| 🖼️ Image | `pillow` + `pytesseract` | OCR → Fixed-size |
| 🗃️ SQL Database | `sqlite3` | Row-as-chunk ya Fixed-size |
| 🎥 Video | `moviepy` + `SpeechRecognition` | Audio transcript → Fixed-size |

### Chunking Strategies:

```python
# 1. Paragraph chunking — blank lines par split
chunks = text.split("\n\n")

# 2. Fixed-size with overlap — N words, overlap ke saath
# "the cat sat on the mat" (size=3, overlap=1)
# → ["the cat sat", "sat on the", "the mat"]

# 3. Sentence chunking — N sentences ek chunk
sentences = re.split(r'(?<=[.!?])\s+', text)
chunks = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
```

> **Overlap kyun?** — Taaki chunk boundary par koi important context na cut ho.

---

## 6. Data Kaise Get/Search Karte Hain

### Basic Query:
```python
query = "what is machine learning?"
query_vector = model.encode(query).tolist()

results = index.query(
    vector           = query_vector,
    top_k            = 3,           # Top 3 similar chunks
    include_metadata = True         # Original text bhi return karo
)

for match in results["matches"]:
    print(f"Score: {match['score']:.4f}")  # 1.0 = perfect match
    print(f"Text:  {match['metadata']['text']}")
```

### Metadata Filter ke saath Query:
```python
# Sirf PDF source se results chahiye
results = index.query(
    vector  = query_vector,
    top_k   = 3,
    filter  = {"type": {"$eq": "pdf"}}   # sirf PDF chunks
)

# Multiple filters
results = index.query(
    vector  = query_vector,
    top_k   = 5,
    filter  = {
        "source": {"$eq": "python_docs.txt"},
        "page"  : {"$gte": 5}             # page 5 ya usse baad
    }
)
```

### Similarity Score samajhna:
```
Score 0.95+  →  Bohot zyada similar (almost same meaning)
Score 0.75+  →  Related topic
Score 0.50+  →  Thoda related
Score < 0.50 →  Alag topic
```

### Delete Operations:
```python
# Specific vectors delete karo
index.delete(ids=["chunk-0", "chunk-1"])

# Filter se delete (Pinecone)
index.delete(filter={"source": {"$eq": "old_doc.txt"}})

# Poora index delete karo
pc.delete_index("index-name")
```

---

## 7. Is Project ki Files

```
vector_db/
├── .env                          ← API keys (PINECONE_API_KEY)
├── .gitignore                    ← env aur myenv ignore
├── README.md                     ← Yeh file
└── embeddings/
    ├── simple_embedding.py       ← Basics: word vectors + cosine similarity
    ├── pinecone_store.py         ← Pinecone se store + query (full pipeline)
    ├── chunking_sources.py       ← 4 sources ka chunking demo (Text/PDF/Image/SQL)
    ├── multi_source_pinecone.py  ← Sab sources + Pinecone (production-ready)
    ├── requirements.txt          ← Sab dependencies
    ├── sample.txt                ← Demo text file
    └── sample.db                 ← Demo SQLite database
```

### File-wise kya seekhte hain:

| File | Kya Sikhata Hai |
|---|---|
| `simple_embedding.py` | Word embeddings kya hain, cosine similarity, bag-of-words |
| `pinecone_store.py` | Pinecone connect karna, upsert, query, metadata filter |
| `chunking_sources.py` | Text/PDF/Image/SQL se chunks kaise banate hain |
| `multi_source_pinecone.py` | Sab kuch ek saath — complete RAG pipeline |

---

## 8. Installation — Kya Kya Install Karna Hai

### Prerequisites:
- Python 3.9+ installed hona chahiye
- Pinecone account (free): [https://app.pinecone.io](https://app.pinecone.io)

### Step 1 — Virtual Environment banao:
```bash
# Project folder mein jaao
cd d:\vector_db

# Virtual environment banao
python -m venv myenv

# Activate karo (Windows CMD)
myenv\Scripts\activate.bat

# Activate karo (Windows PowerShell)
myenv\Scripts\Activate.ps1

# Activate karo (Linux/Mac)
source myenv/bin/activate
```

### Step 2 — Python packages install karo:
```bash
pip install -r embeddings/requirements.txt
```

### Packages ki list aur kaam:

| Package | Version | Kaam |
|---|---|---|
| `numpy` | 2.5.2 | Vectors aur math operations |
| `matplotlib` | 3.11.1 | Embeddings visualize karna |
| `pinecone` | 9.1.0 | Pinecone vector DB client |
| `sentence-transformers` | 3.4.1 | Real AI embeddings (384-dim) |
| `pymupdf` | 1.26.1 | PDF se text extract karna |
| `pillow` | 11.2.1 | Image processing |
| `pytesseract` | 0.3.13 | OCR — image se text nikalna |
| `python-dotenv` | 1.1.0 | `.env` file se API key load karna |

### Step 3 — Tesseract OCR install karo (sirf image chunking ke liye):

**Windows:**
1. Download karo: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install karo (default path: `C:\Program Files\Tesseract-OCR`)
3. PATH mein add karo ya code mein set karo:
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

### Step 4 — Video support ke liye (optional):
```bash
pip install moviepy SpeechRecognition
```

---

## 9. Environment Setup (.env)

Project root mein `.env` file already hai. Usme Pinecone API key set karo:

```
PINECONE_API_KEY=your-actual-api-key-here
```

**API Key kahan se milegi:**
1. [https://app.pinecone.io](https://app.pinecone.io) par jaao
2. Free account banao
3. "API Keys" section mein jaao
4. Key copy karo aur `.env` mein paste karo

> ⚠️ **Important:** `.env` file kabhi bhi Git mein push mat karo — `.gitignore` mein already add hai.

---

## 10. Project Run Kaise Karo

```bash
# Virtual environment activate karo
myenv\Scripts\activate.bat

# 1. Basics seekhna — Word embeddings
python embeddings/simple_embedding.py

# 2. Pinecone pipeline — Store + Query
python embeddings/pinecone_store.py

# 3. Multiple sources chunking demo
python embeddings/chunking_sources.py

# 4. Full production pipeline (text + sql → Pinecone)
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

| Concept | Matlab |
|---|---|
| **Embedding** | Text/image ko numbers ki list mein badalna |
| **Vector** | Numbers ki list jo meaning represent kare |
| **Dimension** | Vector mein kitne numbers hain (e.g., 384) |
| **Cosine Similarity** | Do vectors kitne "kareebi" hain (0-1) |
| **Chunking** | Bade document ko chhote pieces mein todna |
| **Upsert** | Vector store mein save karna (update + insert) |
| **Top-K** | Query ke baad K sabse similar results lena |
| **Metadata** | Vector ke saath extra info (text, source, page) |
| **RAG** | Retrieval Augmented Generation — LLM + Vector DB |

---

## 📚 Aur Seekhna Hai?

- [Pinecone Docs](https://docs.pinecone.io)
- [Sentence Transformers](https://www.sbert.net)
- [LangChain + Vector DB](https://python.langchain.com/docs/integrations/vectorstores/)
- [ChromaDB (local alternative)](https://docs.trychroma.com)


---

---

# 🗄️ Vector Database — Complete Guide (English)

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
