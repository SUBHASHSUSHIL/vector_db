import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PART 1: Simple Word Embeddings (original)
# ─────────────────────────────────────────────

word_embeddings = {
    "king":     [0.5,  0.1,  0.3],
    "queen":    [0.45, 0.15, 0.35],
    "man":      [0.6,  0.2,  0.4],
    "woman":    [0.55, 0.25, 0.45],
    "prince":   [0.52, 0.12, 0.32],
    "princess": [0.48, 0.18, 0.38],
    "throne":   [0.51, 0.11, 0.31],
    "crown":    [0.49, 0.19, 0.39],
}

print("=" * 50)
print("PART 1: Word Embeddings")
print("=" * 50)
for word, embedding in word_embeddings.items():
    print(f"{word}: {embedding}")

plt.figure(figsize=(10, 8))
for word, embedding in word_embeddings.items():
    plt.scatter(embedding[0], embedding[1], label=word)
    plt.annotate(word, (embedding[0], embedding[1]))

plt.title("Simple 2D Word Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid()
plt.show()


# ─────────────────────────────────────────────
# PART 2: Text Chunking + Retrieval Example
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("PART 2: Text Chunking + Similarity Search")
print("=" * 50)

# --- Sample document ---
document = """
Machine learning is a subset of artificial intelligence.
It allows computers to learn from data without being explicitly programmed.
There are three main types: supervised, unsupervised, and reinforcement learning.

Supervised learning uses labeled data to train models.
Examples include classification and regression tasks.
Common algorithms are linear regression, decision trees, and neural networks.

Unsupervised learning finds hidden patterns in unlabeled data.
Clustering and dimensionality reduction are popular techniques.
K-means and PCA are widely used unsupervised methods.

Reinforcement learning trains agents through rewards and penalties.
The agent takes actions in an environment to maximize cumulative reward.
It is used in robotics, game playing, and autonomous systems.
"""


# ─────────────────────────────────────────────
# Step 1: Chunk the document by paragraph
# ─────────────────────────────────────────────

def chunk_by_paragraph(text):
    """Split text on blank lines, discard empty chunks."""
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return chunks


chunks = chunk_by_paragraph(document)

print("\n[Chunks created]")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}: {chunk}")


# ─────────────────────────────────────────────
# Step 2: Fake embeddings (bag-of-words style)
# In a real project you'd call sentence-transformers
# or OpenAI embeddings here instead.
# ─────────────────────────────────────────────

# Vocabulary built from all chunk words
def build_vocab(chunks):
    vocab = set()
    for chunk in chunks:
        for word in chunk.lower().split():
            vocab.add(word.strip(".,"))
    return sorted(vocab)


def embed_text(text, vocab):
    """Simple bag-of-words vector (word frequency)."""
    words = [w.strip(".,") for w in text.lower().split()]
    vector = np.array([words.count(w) for w in vocab], dtype=float)
    return vector


vocab = build_vocab(chunks)

chunk_embeddings = [embed_text(chunk, vocab) for chunk in chunks]

print(f"\n[Vocabulary size]: {len(vocab)} words")
print(f"[Chunks embedded]: {len(chunk_embeddings)} vectors")


# ─────────────────────────────────────────────
# Step 3: Cosine similarity search
# ─────────────────────────────────────────────

def cosine_similarity(a, b):
    """Return cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, chunks, chunk_embeddings, vocab, top_k=2):
    """Find the top_k most similar chunks to the query."""
    query_vec = embed_text(query, vocab)
    scores = [cosine_similarity(query_vec, ce) for ce in chunk_embeddings]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# --- Run a query ---
query = "what is supervised learning?"

print(f"\n[Query]: '{query}'")
results = search(query, chunks, chunk_embeddings, vocab, top_k=2)

print("\n[Top matching chunks]")
for rank, (idx, score) in enumerate(results, 1):
    print(f"\n  Rank {rank} | Chunk {idx} | Score: {score:.4f}")
    print(f"  {chunks[idx]}")


# ─────────────────────────────────────────────
# Step 4: Visualise chunk similarity scores
# ─────────────────────────────────────────────

query_vec = embed_text(query, vocab)
scores = [cosine_similarity(query_vec, ce) for ce in chunk_embeddings]
labels = [f"Chunk {i}" for i in range(len(chunks))]

plt.figure(figsize=(8, 4))
bars = plt.bar(labels, scores, color=["tomato" if s == max(scores) else "steelblue" for s in scores])
plt.title(f"Chunk Similarity to Query:\n\"{query}\"")
plt.ylabel("Cosine Similarity")
plt.ylim(0, 1)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{score:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.show()
