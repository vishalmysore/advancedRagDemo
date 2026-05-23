# Browser-Native Advanced RAG: Building Local-First Retrieval Engines

Large Language Models (LLMs) are often treated as centralised cloud services. We send our queries, documents, and instructions over the internet to APIs run by OpenAI, Anthropic, or Google. While this is straightforward, it presents significant trade-offs in **privacy, cost, latency, and offline support**.

This article explores a different paradigm: **Browser-Native Advanced RAG (Retrieval-Augmented Generation)**. By using WebAssembly, modern client-side storage, and custom JS algorithms, we can run a complete, multi-stage hybrid RAG pipeline entirely inside a browser tab. No server-side databases, no cloud embedders, and no external document parsers.

---

## The Architecture of Client-Side RAG

A local RAG engine consists of five critical layers running inside the browser sandboxed environment:

```
                  ┌──────────────────────────────┐
                  │      User File Ingestion     │
                  │      (PDF.js / FileReader)   │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │    Sentence-Aware Chunker    │
                  └──────────────┬───────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
┌───────────────────────┐                 ┌───────────────────────┐
│   Transformers.js     │                 │      Custom BM25      │
│  Embedding Generation │                 │     Inverted Index    │
└───────────┬───────────┘                 └───────────┬───────────┘
            │                                         │
            ▼                                         ▼
┌───────────────────────┐                 ┌───────────────────────┐
│     Vector Store      │                 │      Term Store       │
│ (IndexedDB / Dexie)   │                 │ (IndexedDB / Dexie)   │
└───────────┬───────────┘                 └───────────┬───────────┘
            │                                         │
            └────────────────────┬────────────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │     Hybrid Fusion Engine     │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │     Two-Stage Reranking      │
                  │  (Proximity & Cross-Encoder) │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │     Post-Gen Verification    │
                  │     (Hallucination checks)   │
                  └──────────────────────────────┘
```

---

## 1. Sentence-Aware Ingestion & Normalisation

RAG quality is highly dependent on chunking quality. If a text splitter breaks a sentence in half, it destroys the semantic context. 

To solve this, our chunker uses **Sentence-Aware Sliding Windows**:
1. **Normalisation**: Text is converted to NFC Unicode, multiple spaces are collapsed to single spaces, and carriage returns are standardized.
2. **Boundary Detection**: A sentence splitter uses regex punctuation patterns `[^.!?]+[.!?]+(?:\s|$)` to capture full sentence boundaries.
3. **Sliding Window assembly**: Sentences are grouped into a single chunk until the target length (e.g. 500 characters) is reached.
4. **Overlapping**: To maintain continuity, the next chunk backtracks to incorporate the last few sentences of the previous chunk (e.g. up to 50 characters of overlap).

---

## 2. Hybrid Retrieval: Vector + Keyword

RAG search engines often suffer from two opposing failure modes:
* **Vector Search (ANN)**: Excellent at catching synonyms and high-level concepts, but struggles with exact numbers, alphanumeric IDs, or specific names.
* **Keyword Search (BM25)**: Excellent at catching exact strings, codes, and names, but blind to semantic synonyms.

By combining them, we create a hybrid index.

### Local Vector Search
We download the `Xenova/all-MiniLM-L6-v2` model. This model converts chunk texts into a 384-dimensional floating-point array. 

When a user query comes in, we generate its query vector $\vec{q}$ and compute the **Cosine Similarity** against every stored chunk vector $\vec{d}$ in IndexedDB:

$$Similarity(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

Since our embedding vectors are pre-normalized during generation, this simplifies to a simple dot product, which can be computed in Javascript in sub-milliseconds:

```javascript
function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  for (let i = 0; i < vecA.length; i++) dot += vecA[i] * vecB[i];
  return dot;
}
```

### Local BM25 Keyword Search
We index chunk tokens incrementally in IndexedDB. For each term, we store a list of document IDs and their frequency counts. The BM25 score for a document $d$ and query $Q$ is computed dynamically:

$$Score_{bm25}(d, Q) = \sum_{q \in Q} IDF(q) \cdot \frac{tf(q, d) \cdot (k_1 + 1)}{tf(q, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

Where:
* $tf(q, d)$ is the term frequency of query token $q$ in chunk $d$.
* $df(q)$ is the document frequency of token $q$ across all chunks.
* $IDF(q) = \ln\left(1 + \frac{N - df(q) + 0.5}{df(q) + 0.5}\right)$ is the Inverse Document Frequency.
* $|d|$ and $avgdl$ are the document length and average document length across the DB.
* $k_1 = 1.2$ controls term frequency saturation.
* $b = 0.75$ controls document length normalisation.

### Normalized Fusion
Since cosine similarity ranges from `[-1, 1]` (practically `[0, 1]` for texts) and BM25 scores are positive real numbers, we normalize the BM25 scores before combining them:

$$Score_{hybrid} = w_{vector} \cdot Similarity_{cosine} + w_{keyword} \cdot \left(\frac{Score_{bm25}}{Score_{maxBM25}}\right)$$

---

## 3. Two-Stage Reranking

Retrieval algorithms prioritize speed over precision. To improve accuracy without slowing down execution, we implement **Two-Stage Reranking**:

* **Stage 1 (Retrieval)**: Fuses Vector + BM25 scores to fetch the top 30 candidates.
* **Stage 2 (Reranking)**: Filters the top 30 down to the top 15, and scores them using one of two methods:
  
### Syntactic Proximity Reranking
A fast, model-free algorithm that scans chunks and calculates a proximity window. If a user queries *"toxicity dosage calc"*, a chunk where those three words appear adjacent is highly relevant. If they are separated by 100 words, it ranks lower.

$$\text{Proximity Score} = \text{density} \cdot 0.6 + \left(\frac{1}{1 + \frac{\text{minSpan}}{5}}\right) \cdot 0.4$$

### Neural Cross-Encoder Reranking
For deep semantic ranking, we run a local Cross-Encoder model (`Xenova/ms-marco-MiniLM-L-6-v2`) in WebAssembly. Unlike a vector model (which embeds query and chunk separately), a Cross-Encoder takes the query and chunk *together* and computes a direct sequence classification score, yielding a high-quality relevance score.

---

## 4. Groundedness & Hallucination Fallbacks

Generative models are prone to hallucinating facts. To safeguard browser-native generation, we build a **Post-Generation Verification Layer**:

```
                       ┌────────────────────────┐
                       │  Generated LLM Answer  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Extract Key Assertions│
                       │  (Numbers & Technical) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Scan Retrieved Chunks │
                       └───────────┬────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
     [All present]                                 [Mismatch found]
┌────────────────────────┐                    ┌────────────────────────┐
│     Verification OK    │                    │  Hallucination Alert!  │
│  Render answer normal  │                    │ Show red warning banner│
└────────────────────────┘                    └────────────────────────┘
```

1. **Assertion Extraction**: The system extracts all numbers, capitalized proper nouns, and key technical words (length > 4, excluding stopwords) from the generated LLM response.
2. **Source Search**: It scans the retrieved chunks to verify if those extracted numbers or entities exist.
3. **Score calculation**:
   $$\text{Faithfulness} = \frac{\text{Matched Assertions}}{\text{Total Assertions}}$$
4. **Fallback warning**: If faithfulness drops below 80%, a red warning banner is appended to the response, listing the exact terms (e.g. *Number "45%" not found in sources*) that could be hallucinations.

---

## 5. Local-First Benefits

* **Zero Cloud Costs**: Embeddings, BM25 indexing, page parsing, and reranking run entirely on the user's CPU/GPU via WebAssembly.
* **100% Privacy**: Uploaded PDFs and search query terms never leave the browser tab.
* **True Offline Support**: By storing model weights in the browser's Cache API, the RAG engine can perform document ingestion, index retrieval, and Mock LLM synthesis completely offline.
