"""
Pinecone Vector Database - Data Store Example
Steps:
  1. Text ko chunks mein todo
  2. Har chunk ka embedding banao (numpy fake / real sentence-transformers)
  3. Pinecone index mein store karo
  4. Query karke similar chunks dhundo
"""

import os
import re
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# .env file se API key load karo
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv nahi hai to env variable directly set karo

# ─────────────────────────────────────────────
# CONFIG  — .env file mein PINECONE_API_KEY set karo
# ─────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "your-api-key-here")
INDEX_NAME       = "vector-db-demo"
DIMENSION        = 384        # sentence-transformers ka default dimension
CLOUD            = "aws"
REGION           = "us-east-1"


# ─────────────────────────────────────────────
# STEP 1: Chunking
# ─────────────────────────────────────────────

def chunk_by_paragraph(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


sample_document = """
Artificial intelligence is transforming every industry.
It enables machines to perform tasks that require human intelligence.
AI includes subfields like machine learning and deep learning.

Machine learning allows systems to learn from data automatically.
Supervised learning uses labeled datasets to train prediction models.
Unsupervised learning finds hidden patterns without labeled data.

Natural language processing helps computers understand human language.
Applications include chatbots, translation, and sentiment analysis.
Large language models like GPT have revolutionized NLP.

Vector databases are used to store and search embeddings efficiently.
Pinecone is a managed vector database built for AI applications.
It supports fast similarity search at scale.
"""

chunks = chunk_by_paragraph(sample_document)
print(f"[Step 1] Total chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: {c[:80]}...")


# ─────────────────────────────────────────────
# STEP 2: Embedding
# Option A: Fake embeddings (no extra install)
# Option B: Real embeddings using sentence-transformers
# ─────────────────────────────────────────────

USE_REAL_EMBEDDINGS = False   # True karo agar sentence-transformers installed ho

if USE_REAL_EMBEDDINGS:
    # pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim model

    def get_embedding(text):
        return model.encode(text).tolist()

else:
    # Fake deterministic embedding (demo ke liye)
    def get_embedding(text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(DIMENSION).tolist()


print(f"\n[Step 2] Creating embeddings (USE_REAL_EMBEDDINGS={USE_REAL_EMBEDDINGS})")
embeddings = [get_embedding(chunk) for chunk in chunks]
print(f"  Each embedding dimension: {len(embeddings[0])}")


# ─────────────────────────────────────────────
# STEP 3: Pinecone mein connect + index banao
# ─────────────────────────────────────────────

print("\n[Step 3] Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index exist nahi karta to create karo
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"  Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name      = INDEX_NAME,
        dimension = DIMENSION,
        metric    = "cosine",          # cosine | euclidean | dotproduct
        spec      = ServerlessSpec(
            cloud  = CLOUD,
            region = REGION
        )
    )
    print(f"  Index '{INDEX_NAME}' created.")
else:
    print(f"  Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)


# ─────────────────────────────────────────────
# STEP 4: Data upsert (store) karo
# Format: (id, vector, metadata)
# ─────────────────────────────────────────────

print("\n[Step 4] Upserting vectors to Pinecone...")

vectors = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vectors.append({
        "id"      : f"chunk-{i}",          # unique ID
        "values"  : embedding,              # float list
        "metadata": {                       # extra info for filtering
            "text"   : chunk,
            "source" : "sample_document",
            "chunk_index": i
        }
    })

# Batch upsert (100 vectors per batch recommended)
BATCH_SIZE = 100
for batch_start in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[batch_start : batch_start + BATCH_SIZE]
    index.upsert(vectors=batch)
    print(f"  Upserted batch: {batch_start} to {batch_start + len(batch) - 1}")

print(f"\n  Total vectors stored: {len(vectors)}")
print(f"  Index stats: {index.describe_index_stats()}")


# ─────────────────────────────────────────────
# STEP 5: Query — similar chunks dhundo
# ─────────────────────────────────────────────

print("\n[Step 5] Querying Pinecone...")

query = "what is natural language processing?"
query_embedding = get_embedding(query)

results = index.query(
    vector          = query_embedding,
    top_k           = 3,               # top 3 similar chunks
    include_metadata= True             # text bhi return karo
)

print(f"\n  Query: '{query}'")
print(f"  Top matches:")
for match in results["matches"]:
    print(f"\n  ID    : {match['id']}")
    print(f"  Score : {match['score']:.4f}")
    print(f"  Text  : {match['metadata']['text'][:120]}...")


# ─────────────────────────────────────────────
# STEP 6: Filter ke saath query (metadata filter)
# ─────────────────────────────────────────────

print("\n[Step 6] Query with metadata filter...")

filtered_results = index.query(
    vector          = query_embedding,
    top_k           = 2,
    include_metadata= True,
    filter          = {"source": {"$eq": "sample_document"}}  # sirf is source se
)

print(f"  Filtered matches (source=sample_document): {len(filtered_results['matches'])}")
for match in filtered_results["matches"]:
    print(f"  [{match['id']}] score={match['score']:.4f} | {match['metadata']['text'][:80]}...")


# ─────────────────────────────────────────────
# STEP 7: Delete (optional cleanup)
# ─────────────────────────────────────────────

# Specific vector delete karo:
# index.delete(ids=["chunk-0", "chunk-1"])

# Poora index delete karo:
# pc.delete_index(INDEX_NAME)

print("\n[Done] All steps completed successfully!")
