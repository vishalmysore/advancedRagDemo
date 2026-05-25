# RAG Pipeline from Zero to Hero

Retrieval-Augmented Generation (RAG) is one of those ideas that is deceptively simple to understand and genuinely hard to master. The concept fits in a sentence: retrieve relevant documents, inject them into a prompt, generate a grounded answer. But the gap between understanding that sentence and building a pipeline that is fast, accurate, honest, and production-reliable is enormous.

This article is the journey from zero to hero. It starts with the fundamentals and works up through the nuanced territory that most guides skip: information retrieval metrics, the precision-recall tradeoff, ranking quality, embedding failure modes, and the specific decisions that determine whether your RAG system is trustworthy or just impressive in demos.

Every concept here is observable live in the [Advanced Local RAG Demo](https://vishalmysore.github.io/advancedRagDemo/) — a fully browser-native pipeline that runs vector search, BM25, hybrid fusion, reranking, hallucination detection, and continuous evaluation with no backend and no data leaving your device.

---

## Zero: What RAG Actually Is and Why It Exists

Language models are trained on static snapshots of data. Once training ends, their knowledge freezes. Ask GPT-4 about an internal policy document your company wrote last month, and it has nothing to work with — it will either say so or, worse, hallucinate a plausible-sounding answer.

RAG solves this by giving the model a retrieval step before generation. Instead of relying on parametric memory (knowledge baked into weights), the system retrieves relevant text from an external knowledge base and hands it to the model as context. The model then generates a response grounded in that retrieved text rather than in its training data.

```
Without RAG:
  User query → LLM (parametric memory) → Answer (may hallucinate)

With RAG:
  User query → Retrieval system → Relevant chunks → LLM + context → Grounded answer
```

This architecture does two things simultaneously: it keeps knowledge up to date without retraining, and it makes the model's answers verifiable — every claim can be traced back to a source document.

That is the zero. Now let's build up.

---

## Stage 1: Ingestion — Turning Raw Documents into Searchable Chunks

Before retrieval can happen, documents must be processed into a form the retrieval system can work with. This is called ingestion, and the quality of every downstream step depends on it.

### Normalisation: the silent killer of retrieval quality

Raw documents are messy. PDFs have extraction artifacts. HTML has tags and entities. Word documents embed formatting characters. Scanned documents have OCR errors. Before any indexing happens, text must be cleaned consistently:

- **Unicode normalisation (NFC):** The character `é` can be encoded as a single codepoint (`U+00E9`) or as two codepoints (`e` + combining accent `U+0301`). These look identical but are not equal in a byte comparison. A BM25 index built without normalisation will silently miss matches.
- **Whitespace standardisation:** Multiple spaces, tabs, non-breaking spaces, and zero-width joiners all collapse to a single standard space.
- **Control character removal:** PDF extraction routinely produces characters outside the printable ASCII range. These corrupt embeddings and break tokenization.

A chunk containing `"revenue growth"` (non-breaking space between words) will never BM25-match a query for `"revenue growth"`. You will spend days debugging retrieval quality before finding this.

### Chunking: the most consequential decision you make

Documents are too long to embed as a whole and too long to fit in a prompt. They must be split into chunks. The size and shape of those chunks determines retrieval precision more than almost any other factor.

**Fixed-size character splitting** is the naive approach and the worst one. It splits on character count with no respect for sentence boundaries, leaving chunks that begin and end mid-thought. The model receives incomplete context and produces incomplete answers.

**Sentence-aware sliding window chunking** is the correct approach:
1. Split on sentence boundaries using punctuation patterns
2. Accumulate sentences into a chunk until a soft size limit is reached
3. When the limit is reached, save the chunk and backtrack by a configurable overlap — typically the last few sentences — before starting the next chunk

The overlap is critical. Without it, a fact that sits at the boundary between two chunks will be split across both, fully present in neither. With overlap, it appears completely in at least one chunk.

**The chunking tradeoff:**

| Chunk size | Precision effect | Recall effect |
|---|---|---|
| Too small (< 100 chars) | High — tight focus on exact terms | Low — loses surrounding context, model gets incomplete information |
| Too large (> 1500 chars) | Low — relevant sentence drowns in noise | High — captures more context, but dilutes the signal |
| Optimal (400–700 chars) | Balanced | Balanced |

This tradeoff is the first encounter with the precision-recall tension that runs through every layer of a RAG pipeline. We will return to it in depth.

**See it live:** The demo exposes Chunk Size and Chunk Overlap sliders. Upload a document, adjust both, and re-ingest. Watch the chunk count change in the Ingested Corpus table. Then run the same query at different chunk sizes and compare which source cards appear — this is chunking's effect on retrieval precision made directly observable.

---

## Stage 2: Indexing — Building Structures That Enable Fast Retrieval

Once chunks exist, they are indexed in two parallel structures: a vector index for semantic search and an inverted index for keyword search.

### Vector indexing

Each chunk is passed through an embedding model — in our demo, `Xenova/all-MiniLM-L6-v2`, a 384-dimensional model running entirely in the browser via ONNX/WebAssembly. The model converts text into a dense floating-point vector that encodes its semantic meaning.

Vectors are normalized to unit length at generation time, which reduces cosine similarity to a simple dot product at query time — a critical performance optimisation at scale.

These vectors are stored in IndexedDB. At small scale (thousands of chunks), an exact scan is fast enough. At millions of chunks, an Approximate Nearest Neighbour (ANN) index — HNSW, IVF-PQ, or ScaNN — is required to maintain sub-10ms retrieval latency.

### BM25 inverted index

BM25 (Best Match 25, or Okapi BM25) is a probabilistic keyword retrieval function. It builds an inverted index: for each term in the corpus, a list of which chunks contain it and how many times.

At query time, BM25 scores each chunk using:

```
Score(q, d) = Σ IDF(t) × [tf(t,d) × (k₁+1)] / [tf(t,d) + k₁ × (1 - b + b × |d|/avgdl)]
```

Where:
- **tf(t,d)** — term frequency: how often term `t` appears in chunk `d`
- **IDF(t)** — inverse document frequency: `log(1 + (N - df + 0.5) / (df + 0.5))` — rare terms score higher than common terms
- **|d| / avgdl** — length normalisation: prevents long chunks from winning simply because they repeat terms more
- **k₁ = 1.2** — controls term frequency saturation. After a term appears ~3-4 times, additional occurrences add diminishing returns
- **b = 0.75** — controls how strongly length normalisation is applied

The inverted index in the demo is stored entirely in IndexedDB — no external search engine, no Elasticsearch, no Solr. One record per term, mapping chunk IDs to their term frequencies.

---

## Stage 3: Retrieval — The Heart of the Pipeline

This is where precision and recall become concrete engineering decisions, not abstract metrics.

### Precision and Recall in Information Retrieval

Before going further, these terms need to be precisely defined in the RAG context.

**Precision** — of the chunks you retrieved, what fraction are actually relevant?
```
Precision = Relevant chunks retrieved / Total chunks retrieved
```

If you retrieve 10 chunks and 7 are relevant to the query, precision = 0.7.

**Recall** — of all the relevant chunks that exist in the corpus, what fraction did you retrieve?
```
Recall = Relevant chunks retrieved / Total relevant chunks in corpus
```

If there are 20 relevant chunks in the corpus and you retrieved 7 of them, recall = 0.35.

**The fundamental tension:** Retrieving more chunks increases recall (you catch more relevant ones) but decreases precision (you also catch more irrelevant ones). Retrieving fewer chunks increases precision but decreases recall.

In RAG, this tension directly affects answer quality:
- **Low precision:** The LLM prompt is polluted with irrelevant context. The model either ignores it (wasted tokens) or incorporates it (wrong answer).
- **Low recall:** The relevant chunk that would have answered the question was never retrieved. The model either admits it doesn't know or hallucinates.

There is no universal optimum. The right balance depends on your domain, your chunk size, and your reranking strategy.

### Precision@K and Recall@K

In practice, retrieval systems are evaluated at a specific cutoff K — the number of results returned. The standard metrics become:

**Precision@K** — of the K chunks retrieved, what fraction are relevant?
```
P@K = Relevant chunks in top K / K
```

**Recall@K** — of all relevant chunks in the corpus, what fraction appear in the top K?
```
R@K = Relevant chunks in top K / Total relevant chunks
```

In RAG systems, K is typically your top-K parameter — the number of chunks injected into the LLM prompt. Setting K=4 means you are operating at P@4 and R@4. This is one of the most important knobs in your system.

### Mean Reciprocal Rank (MRR)

MRR measures how high the first relevant chunk appears in your ranking. If the most relevant chunk is ranked 1st, MRR = 1.0. If it's ranked 5th, MRR = 0.2.

```
MRR = (1/|Q|) × Σ (1 / rank of first relevant chunk for query q)
```

MRR matters because the LLM processes context in order. A highly relevant chunk buried at position 15 in a list of 15 is not as useful as the same chunk at position 1. Research on LLM attention patterns (the "lost in the middle" problem) shows models disproportionately weight context that appears at the beginning and end of the prompt. MRR optimisation — getting the most relevant chunk to the top — is a direct lever on answer quality.

### Normalized Discounted Cumulative Gain (NDCG)

NDCG goes further than MRR by handling graded relevance — where some chunks are highly relevant, some partially relevant, and some not relevant at all.

```
DCG@K = Σ (relevance_score_i / log₂(i+1))
NDCG@K = DCG@K / IDCG@K
```

Where IDCG is the DCG of the ideal ranking (most relevant chunks first). NDCG = 1.0 means your ranking is perfect. NDCG = 0.5 means you have significant room to improve.

**Why this matters for RAG:** A ranking that puts two highly relevant chunks in positions 1 and 2, then three weakly relevant chunks in positions 3-5, is measurably better than a ranking that puts one highly relevant chunk at position 3 and fills the top 2 with weakly relevant chunks. NDCG captures this distinction. MRR does not.

### Mean Average Precision (MAP)

MAP averages the precision at each position where a relevant chunk is found, across all queries.

```
AP(q) = (1/R) × Σ P@k × rel(k)   [where rel(k) = 1 if chunk at position k is relevant]
MAP = (1/|Q|) × Σ AP(q)
```

MAP penalises you both for retrieving irrelevant chunks and for retrieving relevant chunks late. It is the most complete single-number summary of ranking quality across a query set.

### Which metric to use?

| Metric | Best for | Blind to |
|---|---|---|
| Precision@K | Minimising noise in the prompt | Late relevant results, partial relevance |
| Recall@K | Ensuring relevant info is captured | Irrelevant results in top K |
| MRR | Optimising for first relevant hit | Order of subsequent relevant results |
| NDCG@K | Graded relevance, full ranking quality | Binary relevant/not-relevant setups |
| MAP | Complete ranking quality across queries | Graded relevance |

In practice: use **NDCG@10** as your primary retrieval quality metric during development. Use **Precision@K** to tune your final top-K prompt injection. Use **Recall@K** to validate that your retrieval is not missing critical documents.

---

## Stage 4: Hybrid Retrieval — Why Neither Vector Nor BM25 Alone Is Enough

Vector search and BM25 fail in complementary ways. This is not a coincidence — it follows from their fundamental designs.

**Vector search fails at:** exact term matching. Searching for "ISO 27001 Section 6.1.2" in a document corpus will return chunks about information security in general. The embedding model has learned that "ISO 27001" and "information security" are semantically related and surfaces both. But the user wanted the specific section, not general context.

**BM25 fails at:** semantic equivalence. "Heart attack" and "myocardial infarction" mean the same thing. BM25 sees two completely different term sets and matches neither to queries using the other. Vector search handles this naturally.

**Hybrid fusion:**
```
fusedScore = α × vectorScore + (1-α) × normalizedBM25Score
```

Normalising BM25 scores to [0,1] before fusion is non-optional. Raw BM25 scores are unbounded positive numbers. A score of 4.7 from BM25 and a cosine similarity of 0.73 from the vector search are not directly comparable until BM25 is normalised by the maximum BM25 score in the result set.

**Effect on precision and recall:**
- Hybrid retrieval consistently outperforms either approach alone on both precision and recall across mixed document types
- The optimal α varies by domain: exact-match-heavy domains (legal, financial, technical) → lower α (more BM25); conceptual domains (research, FAQs) → higher α (more vector)
- Tuning α changes the precision-recall tradeoff — leaning BM25 increases precision for exact queries, leaning vector increases recall for semantically varied queries

**See it live:** Enable Benchmark Mode in the demo to run Vector Only, BM25 Only, and Hybrid side by side on the same query. The retrieved source cards for each method show different chunks with different scores. On a query like "who is Vishal Mysore", the BM25 path retrieves chunks where those exact words appear; the vector path retrieves chunks about career and professional background; the hybrid path retrieves the most relevant combination of both.

---

## Stage 5: Reranking — Fixing the Order After Retrieval

First-stage retrieval is optimised for speed, not ranking quality. The bi-encoder model that powers vector search embeds the query and each chunk *independently* — it never actually reads them together. BM25 scores term overlap without any understanding of context.

Reranking is the second stage: take the top-30 candidates from fusion and re-score them with a model that reads query and chunk *jointly*.

### Syntactic proximity reranking

No model required. Measures how tightly query keywords cluster within each chunk.

For query `"machine learning pipeline"` and a chunk containing those three words:
- If they appear within 5 words of each other → high proximity score
- If they are spread across 300 words → low proximity score

```
proximityScore = termDensity × 0.6 + (1 / (1 + minSpan/5)) × 0.4
```

**Effect on precision:** High. Chunks where query terms cluster tightly are almost always more relevant than chunks where the same terms are scattered. This reranker raises precision without requiring a model download.

**Blind spot:** It cannot detect semantic relevance. A chunk that answers the question perfectly using synonyms scores zero if no query tokens are present.

### Neural cross-encoder reranking

The `ms-marco-MiniLM-L-6-v2` model was trained on the MS MARCO dataset — 8.8 million real search queries paired with human-judged relevant passages. It takes `[query, chunk]` as a concatenated input and outputs a relevance score.

Unlike the bi-encoder used for vector search, the cross-encoder has full attention over both query and chunk simultaneously. It can detect:
- Semantic equivalence across different word choices
- Partial relevance (the chunk addresses part of the question)
- Query intent (distinguishing "how to fix X" from "what is X")

**Effect on MRR and NDCG:** Neural reranking consistently improves both. In standard IR benchmarks, adding a cross-encoder reranker to a bi-encoder retrieval system improves NDCG@10 by 15-25% on average.

**The cost:** One model inference per candidate chunk. At 30 candidates, that is 30 sequential inference calls. In the demo running in a browser via ONNX/WebAssembly, this takes 150-400ms depending on the device. In a GPU-accelerated production backend, it takes 20-50ms total with batching.

**See it live:** Switch the Stage 2 Reranking dropdown to Neural. The Observability Trace shows the reranking step. Compare the order of source cards in the Retrieved Sources panel before and after — chunks that moved up were judged more relevant by the cross-encoder than the bi-encoder had rated them.

---

## Stage 6: The Precision-Recall Tradeoff Across the Full Pipeline

Every parameter in your RAG system shifts the precision-recall balance. Understanding which levers move which metric is the difference between systematic tuning and random experimentation.

| Parameter | Increase → | Effect on Precision | Effect on Recall |
|---|---|---|---|
| Chunk size | Larger | ↓ Lower (more noise per chunk) | ↑ Higher (more context per chunk) |
| Top-K | Higher K | ↓ Lower (more irrelevant chunks enter prompt) | ↑ Higher (more relevant chunks captured) |
| Vector weight α | Higher α | Domain-dependent | Domain-dependent |
| Reranking | Neural > Syntactic > None | ↑ Higher (better chunks ranked top) | Neutral (same pool) |
| Confidence threshold | Higher threshold | ↑ Higher (stricter gate) | ↓ Lower (more queries return no answer) |

**The key insight:** Reranking improves precision without affecting recall — it rearranges the same pool of candidates, pushing relevant chunks to the top without adding or removing any. This makes it one of the most valuable levers in the pipeline. It is a precision improvement with no recall cost.

Top-K, by contrast, is a direct precision-recall tradeoff. Every chunk you add beyond the most relevant ones reduces precision. Every chunk you remove risks losing a relevant one and reducing recall. The practical optimum for most use cases is K=4 to K=6 with good reranking, rather than K=10 to K=15 with poor reranking.

---

## Stage 7: Constrained Generation — Where Retrieval Meets the LLM

Retrieved chunks reach the LLM through the system prompt. How that prompt is constructed determines whether the LLM uses the retrieved context faithfully or supplements it with its own training knowledge.

This is where most RAG implementations have a silent flaw: they provide context but do not constrain the model to use only that context. The model, trained to be helpful, fills gaps with its training knowledge when the context is insufficient. This produces answers that are partially grounded and partially hallucinated — the worst possible combination, because they are difficult to detect and appear credible.

**The constrained generation contract:**
```
System: You are a citation-backed AI assistant. Answer using ONLY 
the provided Context sections. For every claim, include [Source N].
If the Context does not contain sufficient information, respond:
"The provided documents do not contain sufficient information to 
answer this question." Do NOT use training data to fill gaps.
Temperature: 0.0 or 0.1
```

**Temperature is a precision knob for generation.** Higher temperature = more creative responses = higher probability of drifting from provided context. For RAG, set temperature at 0.0 or 0.1. RAG is a precision task, not a creative one.

**Prompt position matters.** Research on LLM attention ("lost in the middle", Liu et al. 2023) shows models give disproportionate weight to content at the beginning and end of the context window. The most relevant chunk should be positioned first. This is another reason reranking matters — it puts the highest-confidence chunk where the model will attend to it most strongly.

---

## Stage 8: Faithfulness and Hallucination Detection

Constrained generation dramatically reduces hallucination. It does not eliminate it. Even with an explicit "do not use training data" instruction, models occasionally:

- Paraphrase a chunk in a way that subtly alters its meaning
- Combine information from two chunks in a way that neither alone supports
- Generate a precise-sounding number that does not appear in any retrieved chunk
- Produce an answer that is technically grounded but misleadingly incomplete

**Faithfulness** measures how well the generated answer is supported by the retrieved context:

```
Faithfulness = verified assertions / total assertions
```

Where assertions are extracted from the response — numbers, proper nouns, technical terms, dates, percentages — and each is checked for presence in the retrieved chunks.

**The faithfulness-completeness tradeoff:** A response that says nothing can never hallucinate. A response that is comprehensive is more useful but has more surface area for hallucination. The right operating point is not maximum faithfulness (which produces vacuous answers) — it is high faithfulness with a warning system that fires when the model drifts.

**Threshold design:**
- faithfulness > 0.9 → show response, no warning
- 0.8 ≤ faithfulness ≤ 0.9 → show response with soft warning
- faithfulness < 0.8 → show response with prominent hallucination alert listing unverified claims
- faithfulness < 0.5 → suppress response, return to retrieval step with relaxed parameters

**See it live:** The demo computes faithfulness after every query and displays it in the Evaluation Metrics panel. The Hallucination Alert banner fires when faithfulness drops below threshold, listing the specific assertions that could not be grounded in the retrieved context.

---

## Stage 9: Continuous Evaluation — The Metrics That Matter in Production

RAG systems degrade silently. New documents change the embedding distribution. A model update changes generation patterns. A spike in query diversity hits retrieval patterns the system was not tuned for. Without continuous evaluation, you find out from users, not from your monitoring system.

### The three core RAG evaluation metrics

**Context Relevance** — retrieval quality metric
```
contextRelevance = |queryTokens ∩ contextTokens| / |queryTokens|
```
If this is low, your retrieval is broken. The chunks reaching the LLM are not relevant to the question. No generation improvement will fix this — it is a retrieval problem.

**Faithfulness** — generation quality metric
```
faithfulness = verified assertions / total assertions
```
If this is low with good context relevance, your constrained generation is failing. The model is drifting from the provided context. Tighten the system prompt and reduce temperature.

**Answer Relevance** — end-to-end quality metric
```
answerRelevance = |queryTokens ∩ answerTokens| / |queryTokens|
```
High context relevance + low answer relevance = the model is ignoring the context. This often indicates a prompt structure problem where the context is too long and the model is attending only to the system instructions.

### Additional production metrics

**Retrieval diversity** — are your top-K chunks always coming from the same 2-3 documents? Low diversity indicates your embedding space is too clustered or your corpus lacks coverage on the query domain.

**Latency percentiles (p50/p95/p99)** — track per stage: embedding generation, ANN search, BM25 scan, fusion, reranking, LLM call, eval. p95 and p99 latency often reveal bottlenecks invisible in averages.

**Cache hit rate** — in production with real user traffic, expect 30-50% of queries to be near-duplicates. A cache hit rate below 20% suggests your cache invalidation logic is too aggressive or your query distribution is too diverse for caching to help.

**User rejection rate** — the ground truth. When users click "this answer is wrong", log the query, the retrieved chunks, the generated answer, and all eval scores. This data is your most valuable dataset for systematic improvement.

**See it live:** After every query, the demo displays Context Relevance, Faithfulness, Answer Relevance, and Latency — computed locally in the browser. Adjust chunk size, retrieval weights, and reranking method, and watch how the metrics respond. This is the fastest feedback loop available for developing intuition about what actually moves RAG quality.

---

## Stage 10: Caching, Memory, and Getting Smarter Over Time

A RAG pipeline that produces the same quality on day 365 as on day 1 is not learning from its operation. Production RAG systems should improve continuously.

### Caching

**Exact cache:** Hash `(query + retrieval config + model)` → cache the full result. TTL tied to source document freshness. When a cached query's source documents are updated, invalidate the cache entry for that query.

**Semantic cache:** For incoming queries, compute the embedding and check cosine similarity against cached query embeddings. Similarity > 0.95 → return cached result. This handles paraphrased repetitions without storing an entry per phrasing variant.

### Long-term memory from human feedback

When an expert corrects a wrong answer, that correction is structured knowledge. Store it:
```json
{
  "correction": "The termination clause applies only to fixed-term contracts.",
  "triggered_by": "What does the termination clause cover?",
  "tags": ["termination", "contract", "fixed-term"],
  "domain": "legal"
}
```

On future similar queries, retrieve relevant corrections and inject them at the top of the system prompt. The model now has institutional knowledge that was not present in the original documents. This is how RAG systems accumulate accuracy over time without retraining the model or re-embedding the corpus.

---

## The Hero: Putting It All Together

The difference between a RAG prototype and a production RAG system is not which LLM you use. It is the engineering discipline applied to every layer between the user's query and the model's response.

The hero-level RAG engineer understands:

- **Chunking is a precision-recall decision**, not a preprocessing step. Every size choice has measurable consequences on retrieval quality.
- **BM25 and vector search fail in complementary ways.** Neither is optional. Hybrid retrieval is not an advanced feature — it is the baseline.
- **Precision@K, Recall@K, MRR, NDCG, and MAP** are not academic metrics. They are the vocabulary for diagnosing and fixing retrieval problems systematically rather than by intuition.
- **Reranking improves precision without reducing recall.** It is the highest-ROI improvement available to a functioning retrieval system.
- **The system prompt is an engineering contract**, not a suggestion. Constrained generation with citations is what separates a trustworthy RAG system from a sophisticated autocomplete.
- **Faithfulness is measurable and must be measured continuously.** Silent degradation is the failure mode that costs the most.
- **Retrieval quality matters more than model quality.** A perfectly retrieved set of 5 chunks with a mid-tier model produces a better answer than a frontier model working from poorly retrieved context.

---

## Experience the Full Pipeline

The [Advanced Local RAG Demo](https://vishalmysore.github.io/advancedRagDemo/) implements every stage described in this article — running entirely in your browser, with zero backend, zero cloud costs, and zero data leaving your device.

**What you can observe:**

- **Chunking tradeoffs:** Adjust Chunk Size and Overlap sliders, re-ingest a document, and run the same query. Watch precision and recall shift as chunk boundaries change.
- **Hybrid retrieval:** Benchmark Mode runs Vector Only, BM25 Only, and Hybrid simultaneously on the same query. The difference in retrieved chunks is the precision-recall tradeoff made visible.
- **Reranking effect on MRR:** Switch between None, Syntactic, and Neural reranking and observe how the order of source cards changes. The chunk that moves to position 1 is what the LLM will weight most heavily.
- **Faithfulness measurement:** Every query produces a live Faithfulness score. Run queries on documents where the answer is absent — watch faithfulness drop and the Hallucination Alert fire.
- **Top-K precision tradeoff:** Adjust the Retrieve Top-K slider from 1 to 10 and observe how answer quality and faithfulness change as more (potentially noisier) chunks enter the prompt.
- **Caching in action:** Run a query twice. The second run returns a `CACHE HIT` in the Observability Trace and completes in milliseconds. Toggle the cache off to see full pipeline latency.

Every slider, dropdown, and toggle in the demo is a direct implementation of one of the decisions described in this article. There is no faster way to build an intuition for RAG engineering than watching the metrics respond in real time to your choices.

---

*Demo source: [github.com/vishalmysore/advancedRagDemo](https://github.com/vishalmysore/advancedRagDemo) — Built with Vite, Transformers.js (ONNX/WASM), Dexie/IndexedDB, and a hand-rolled BM25 engine.*
