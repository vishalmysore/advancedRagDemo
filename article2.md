# How to Design a RAG Pipeline for 10 Million Documents with Zero Hallucination

**Retrieval-Augmented Generation (RAG) at scale is one of the most demanding engineering challenges in production AI today.** The gap between a working prototype and a system that reliably handles millions of documents without hallucinating is not a small one — it spans architecture, infrastructure, evaluation, and operational discipline.

When teams set out to build RAG systems for enterprise search, internal knowledge bases, legal document analysis, healthcare records, or large-scale customer support platforms, they quickly discover that the naive approach — "grab some embeddings, do a similarity search, pass chunks to an LLM" — breaks down fast. It works at 1,000 documents. It does not work at 1 million. And it is entirely unsuitable at 10 million.

The reasons are not mysterious. They are specific, predictable engineering problems: retrieval latency, index maintenance cost, precision degradation at scale, hallucinations that compound through the generation step, lack of verifiability, and no way to know when the system is silently failing.

Building a RAG pipeline for 10 million documents that produces trustworthy, citation-backed, hallucination-resistant responses requires treating each layer of the stack as a first-class engineering concern. There are **10 critical areas** to get right — each one a deliberate design decision, each one a failure mode if skipped or underestimated. This article walks through all of them: the reasoning, the math, and the production tradeoffs.

---

## Why 10 Million Docs Changes Everything

At 1,000 documents, you can brute-force anything. Vector scan every chunk, prompt the model with all of it, call it done.

At 10 million documents, everything breaks:

- Brute-force vector search takes **minutes**, not milliseconds
- A single frontier model call cannot see even 0.001% of your corpus
- A bad retrieval step poisons everything downstream — the best model in the world cannot unfailucinate a wrong chunk
- Hallucinations compound: one wrong fact leads the model to generate plausible-sounding extensions of that fact

This is why **retrieval quality matters more than the frontier model itself** at scale. A perfectly retrieved set of 5 chunks with GPT-3.5 will outperform a hallucinating GPT-4 response built on bad retrieval every single time.

---

## Step 01 — Ingest + Normalize Docs

Before any retrieval can work, your data has to be clean and consistent.

**The problem:** 10 million documents come from everywhere — PDFs, Word files, HTML pages, scanned images, database exports, Markdown files, internal wikis. Each has different encodings, different noise patterns, different structure.

**What you do:**
- Strip all formatting artifacts (HTML tags, PDF control characters, footnote markers)
- Normalize Unicode (NFC normalization — `é` and `é` are not the same bytes)
- Remove non-printable characters and control sequences
- Standardize whitespace and newlines
- Detect and handle multi-language content separately
- Tag every document with metadata at ingest: source, date, author, domain, version

**Why it matters:** A chunk that contains `"Vishal Mysore"` (non-breaking space) will never BM25-match a query for `"Vishal Mysore"` (regular space). These silent failures destroy recall and you will never debug them unless normalization is enforced at ingest.

**At 10M scale:** Use a distributed ingestion pipeline (Kafka + Spark or Flink). Process documents in parallel, idempotently. Every document gets a content hash — re-ingest is a no-op if the content hasn't changed.

---

## Step 02 — Hybrid Retrieval (BM25 + Embeddings)

This is where most engineers make their first mistake: **they use only embeddings.**

Embeddings are powerful but they have a known failure mode — exact keyword matching. If a user asks "What did Clause 4.2.1 of the NDA say about termination?", a semantic embedding model will return chunks about *termination in general* rather than *Clause 4.2.1 specifically*. BM25 will nail it.

**BM25 (Okapi BM25):**
```
score(q, d) = Σ IDF(tᵢ) × [tf(tᵢ,d) × (k1+1)] / [tf(tᵢ,d) + k1×(1-b+b×|d|/avgdl)]
```
- `k1 = 1.2` (term frequency saturation)
- `b = 0.75` (length normalization)
- IDF rewards rare terms, penalizes common ones
- Length normalization prevents long chunks from dominating just because they repeat words more

**Vector embeddings:**
- `all-MiniLM-L6-v2` for speed (384 dimensions), or `text-embedding-3-large` for accuracy
- Mean pooling with L2 normalization
- Cosine similarity = dot product on normalized vectors

**Hybrid fusion:**
```
fusedScore = α × cosineSimilarity + (1-α) × normalizedBM25
```

The weight `α` is tunable per domain. Legal documents → lean BM25 (0.3 vector / 0.7 BM25). Conceptual knowledge bases → lean vector (0.7 vector / 0.3 BM25).

At 10M docs, run both retrieval paths in parallel. Each returns top-30 candidates. Union them, fuse scores, pass top-15 to the reranker.

---

## Step 03 — ANN + Reranking (Two-Stage)

**Stage 1: Approximate Nearest Neighbour (ANN)**

You cannot do exact cosine similarity over 10M × 384-dimensional vectors in real time. ANN indices trade a small amount of accuracy for massive speed gains.

Options ranked by production maturity:
- **HNSW** (Hierarchical Navigable Small World) — best recall/speed tradeoff, used in Pinecone, Weaviate, pgvector
- **IVF-PQ** (Inverted File + Product Quantization) — used in FAISS, lower memory footprint
- **ScaNN** — Google's implementation, best throughput at extreme scale

HNSW at 10M vectors returns top-100 candidates in **~10ms** with >95% recall@10 compared to exact search.

**Stage 2: Cross-Encoder Reranking**

The ANN stage is fast but imprecise — it scores query and chunk *independently*. A cross-encoder scores them *jointly*, reading both together:

```
CrossEncoder([query, chunk]) → relevance_score ∈ [0,1]
```

Models: `ms-marco-MiniLM-L-6-v2` (fast, 90MB), `ms-marco-MiniLM-L-12-v2` (more accurate).

Why does this matter? The bi-encoder (ANN) that retrieved your top-30 might have ranked chunk #17 as the most relevant. The cross-encoder reads the actual query text against chunk #17's actual content and realizes chunk #3 is far more relevant. This reordering is the difference between a good RAG and a great one.

At 10M scale, reranking only runs on top-15 to top-30 candidates — never the full corpus.

---

## Step 04 — Source Confidence Scoring

Every retrieved chunk must carry a **confidence score** before it touches the LLM prompt. This score becomes your hallucination defence mechanism.

**Confidence components:**
1. **Retrieval confidence** — normalized fusion score from Step 3 (0→1)
2. **Source freshness** — recency weight: documents older than 2 years get a decay penalty
3. **Source authority** — domain-specific trust scores (internal audit docs > random web pages)
4. **Cross-chunk agreement** — if 4 of your top-5 chunks say the same thing, confidence rises

**Compute a final weighted confidence:**
```
confidence = 0.5×retrievalScore + 0.2×freshnessScore + 0.2×authorityScore + 0.1×agreementScore
```

**Threshold gate:** If `confidence < 0.65` for ALL retrieved chunks, do not generate. Return "Insufficient information found in the knowledge base." This is not a failure — this is the system working correctly. A confident wrong answer is infinitely worse than an honest "I don't know."

---

## Step 05 — Constrained Generation

This is the architectural decision that separates zero-hallucination RAG from regular RAG.

**The rule:** The LLM system prompt must explicitly constrain the model to only use the provided context. No exceptions.

```
System: You are a citation-backed AI assistant. Answer using ONLY 
the provided Context sections below.

Rules:
1. Every claim you make must be supported by the provided Context.
2. Cite every assertion with [Source N] where N is the context section number.
3. If the Context does not contain the answer, respond with exactly:
   "The provided documents do not contain sufficient information to 
   answer this question."
4. Do NOT use any knowledge from your training data to fill gaps.
5. Do NOT speculate, extrapolate, or make inferences beyond what 
   the Context explicitly states.

Context:
---
[Source 1: document_name.pdf, Page 4]
<chunk text>
---
[Source 2: policy_v3.docx, Page 12]
<chunk text>
---
```

**Why this works:** Modern LLMs are instruction-following machines. Given explicit, unambiguous constraints with consequences defined (cite or admit ignorance), they comply far more reliably than when given vague prompts like "answer based on the documents."

**Temperature:** Set to `0.0` or `0.1` for RAG. High temperature = high creativity = high hallucination. You do not want creativity. You want faithfulness.

---

## Step 06 — Citation-Backed Responses

Every response must be verifiable. Not just "here's the answer" — but "here's the answer, it came from these exact chunks, which came from these exact documents, on these exact pages."

**Citation format in the response:**
```
Vishal Mysore joined the company in 2019 [Source 1] and led the 
cloud migration initiative [Source 3], which reduced infrastructure 
costs by 40% [Source 1, Source 2].
```

**What you store per response:**
- The exact chunk text used
- The source document ID and version
- The page number and character offset
- The retrieval score at the time of generation
- The timestamp of the document version (was it the latest at query time?)

**Why this is non-negotiable at enterprise scale:** Legal, compliance, and audit teams need to trace every AI-generated claim back to its source document. If you cannot do this, your RAG system is not enterprise-ready. Full stop.

Citations also enable a feedback loop — if a user disputes a claim, you know exactly which chunk generated it, which lets you fix the document, retrain embeddings for that document, and invalidate the cache for related queries.

---

## Step 07 — Hallucination Fallback Layer

Even with constrained generation and citations, hallucinations can slip through. You need an automated detection layer **before the response reaches the user.**

**Three-pass verification:**

**Pass 1 — Assertion extraction:**
Extract all factual claims from the response. Numbers, proper nouns, dates, percentages, named entities. Regex + NER (Named Entity Recognition).

**Pass 2 — Grounding check:**
For each extracted assertion, verify it appears in the retrieved context. Fuzzy string matching (not exact — the model paraphrases). If an assertion appears in the response but NOT in any retrieved chunk, flag it.

**Pass 3 — Confidence threshold:**
```
faithfulness = verified_assertions / total_assertions
```
If `faithfulness < 0.8` AND there are flagged assertions → show hallucination warning. Do NOT suppress the response — surface the warning to the user with the specific claims that could not be verified.

**Fallback action options (in order of severity):**
1. Show response with inline warning on unverified claims
2. Re-run with `temperature = 0.0` and stricter prompt
3. Return "cannot verify" if second pass also fails
4. Escalate to human review queue

**At 10M scale:** Run this as an async post-processor. Stream the response to the user, run verification in parallel, and show the hallucination banner if needed within 500ms of response completion.

---

## Step 08 — Continuous Evals

You cannot improve what you do not measure. RAG evals must run **continuously in production**, not just during offline testing.

**The three core RAG metrics:**

**Context Relevance** — Are the retrieved chunks actually relevant to the query?
```
contextRelevance = queryTokens ∩ contextTokens / |queryTokens|
```
Low context relevance = retrieval problem, not a model problem.

**Faithfulness** — Does the response stay grounded in the retrieved context?
```
faithfulness = verified_claims / total_claims
```
Low faithfulness = generation problem. Constrain the prompt harder.

**Answer Relevance** — Does the response actually answer the question?
```
answerRelevance = queryTokens ∩ answerTokens / |queryTokens|
```
Low answer relevance despite high context relevance = the model is ignoring the context.

**Beyond the three core metrics:**
- **Latency p50/p95/p99** — per retrieval stage, per model call
- **Cache hit rate** — should be >40% in production for common query patterns
- **Retrieval diversity** — are you always pulling from the same 5 documents?
- **User rejection rate** — how often do users click "this answer is wrong"?

**Pipeline:** Every query → eval scores computed → logged to time-series store → dashboarded → alerting on degradation. If faithfulness drops below 0.75 over a rolling 1-hour window, page someone.

---

## Step 09 — Caching + Memory Layer

At 10M documents and production traffic, you will see the same queries repeatedly. Recomputing the full pipeline for identical queries is wasteful and adds unnecessary latency.

**Two-level cache:**

**Level 1 — Query result cache (exact match):**
Hash `(query + retrieval_config + model)` → cache the full response with citations. TTL tied to document freshness — if any source document in the result has been updated, invalidate the cache entry.

**Level 2 — Embedding cache (semantic near-duplicate):**
Cache query embeddings. For incoming queries, check if a semantically similar query (cosine similarity > 0.97) has been answered before. Return the cached result with a "similar query matched" note.

**Memory layer (session + long-term):**

*Session memory:* Within a conversation, maintain context of what has been discussed. If the user asked about "Clause 4.2.1" three turns ago, they shouldn't have to repeat it. Store the conversation history and inject relevant prior turns into retrieval context.

*Long-term memory (HITL feedback):* When a human expert corrects a wrong answer, store that correction tagged with the query topic, source document, and domain keywords. On future similar queries, retrieve relevant corrections and prepend them to the system prompt:
```
[Retrieved expert correction from prior session]
Note: Previous answer on termination clauses was incorrect — 
Clause 4.2.1 applies only to fixed-term contracts, not at-will.
```

This is how your RAG system gets smarter over time without retraining the model.

---

## Step 10 — Observability Everywhere

At 10M docs and production load, something will go wrong. The only question is whether you find out from your monitoring system or from an angry enterprise customer.

**Instrument every layer:**

```
[INGEST LAYER]       Document parsed — 847 chunks generated in 2.3s
[VECTOR LAYER]       ANN search — 30 candidates in 8ms (HNSW index)
[BM25 LAYER]         Keyword search — 12 candidates in 3ms
[FUSION LAYER]       Hybrid merge — 38 unique candidates, top 15 selected
[RERANK LAYER]       Cross-encoder scored 15 chunks in 180ms
[CONFIDENCE LAYER]   Top chunk: 0.847, threshold: 0.65 — PASS
[GENERATION LAYER]   LLM call — 1240ms, 387 tokens generated
[EVAL LAYER]         Faithfulness: 0.91, Relevance: 0.84 — OK
[CACHE LAYER]        Result cached. Key: a3f9b2c1...
```

**What to trace per query:**
- Timing breakdown per stage (not just total latency)
- Which documents were retrieved and their scores
- Which chunks were used vs rejected by the reranker
- The exact system prompt sent to the model
- Raw model response before citation parsing
- Eval scores
- Cache hit/miss

**Infrastructure:** OpenTelemetry for distributed tracing, Prometheus + Grafana for metrics, structured JSON logs to Elasticsearch or Loki. Every trace must be queryable by document ID, query hash, user session, and time range.

**Why this matters:** When a hallucination slips through at 3am, you need to know within minutes: which document caused it, which retrieval step ranked it too highly, which eval metric failed to catch it, and how many users saw it. Without observability, you're guessing.

---

## The Takeaway That 99% Miss

Most engineers optimize in the wrong direction. They spend weeks evaluating GPT-4 vs Claude vs Gemini and minutes thinking about retrieval.

**The hard truth:**

> At 10M documents, **retrieval quality matters more than the frontier model itself.** A well-retrieved set of 5 faithful chunks with a mid-tier model will produce a better, more trustworthy answer than a frontier model hallucinating over poorly retrieved context.

Your retrieval pipeline is the foundation. The LLM is the finishing coat. You cannot paint over a cracked wall and expect it to hold.

The engineers who understand this — who obsess over BM25 index quality, fusion weights, reranker calibration, confidence thresholds, and citation grounding — are the ones building RAG systems that actually work in production. The rest are building impressive demos that fail in the first enterprise audit.

---

## See It Live — Advanced RAG Demo

Everything described in this article is **running in your browser right now** at the [Advanced Local RAG Demo](https://vishalmysore.github.io/advancedRagDemo/).

No server. No backend. 100% browser-native.

Here is what you can observe live as you use it:

**Step 02 in action — Hybrid Retrieval:**
Upload any PDF and watch the RAG Pipeline Observability Trace show both `VECTOR RETRIEVAL` and `BM25 KEYWORD RETRIEVAL` running in parallel, each returning scored candidates. Adjust the **Weights slider** (Vector vs Keyword) to see how changing `α` shifts which chunks surface.

**Step 03 in action — Two-Stage Reranking:**
Switch the **Stage 2 Reranking** dropdown between:
- *Syntactic (Query Proximity Window)* — watch the proximity score computation in the trace
- *Neural (Cross-Encoder MiniLM — 90MB)* — triggers a real ONNX model download and scores your chunks with a genuine ML cross-encoder, entirely in-browser via WebAssembly

**Step 04 in action — Confidence Scoring:**
Every source card in the Retrieved Sources panel shows its **Vector Cosine score**, **BM25 score**, and **Confidence %** — the exact three-component fusion score that gates whether that chunk enters the prompt.

**Step 05 + 06 in action — Constrained Generation + Citations:**
The generated answer uses `[Source N]` citation badges. Click any badge and it scrolls directly to the source chunk that backed that claim. The system prompt enforcing constrained generation is visible in the trace under `PROMPT ASSEMBLY`.

**Step 07 in action — Hallucination Detection:**
The **Hallucination Alert** banner fires automatically when faithfulness drops below 0.8. The banner tells you exactly which assertions in the response could not be grounded in the retrieved context.

**Step 08 in action — Continuous Evals:**
After every query, four metrics appear live: **Context Relevance**, **Faithfulness**, **Answer Relevance**, and **Latency** — computed locally in the browser without any eval API.

**Step 09 in action — Caching:**
Run the same query twice. The second run shows `CACHE HIT` in the trace and returns in milliseconds instead of seconds. Toggle **Use Retrieval Cache** off to force a fresh pipeline run and compare latencies.

**Step 10 in action — Observability:**
The entire **RAG Pipeline Observability Trace** panel is a live implementation of Step 10 — every layer logs what it did, how long it took, and what score it produced. This is your local version of a distributed trace.

**Benchmark Mode:**
Toggle **Compare Retrieval Methods** to run all three retrieval strategies simultaneously — Vector Only, BM25 Only, and Fused Hybrid — side by side on the same query. You will see immediately why neither pure vector nor pure keyword retrieval matches the hybrid approach on real documents.

---

The demo runs entirely on your device. Your documents never leave your browser. Your API key never touches our servers. It is the fastest way to develop an intuition for what these 10 steps actually feel like at the retrieval and generation boundary — before you go build it at 10 million documents.

---

*Built with Vite, Transformers.js (ONNX/WASM), Dexie/IndexedDB, and a hand-rolled BM25 engine. No backend. No vector database subscription. No cloud costs.*
