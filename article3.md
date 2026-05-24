# Top 10 Things to Take Care of When Building a RAG Pipeline

Building a Retrieval-Augmented Generation (RAG) pipeline looks straightforward on paper. You chunk your documents, embed them, store the vectors, retrieve the closest matches, and hand them to an LLM. Thirty lines of code and you have a working demo.

Then you take it to production and the real problems start.

After building and observing a fully browser-native RAG pipeline — one that runs vector search, BM25 keyword retrieval, hybrid fusion, reranking, hallucination detection, and continuous evaluation entirely in the browser without a backend — here are the 10 things that will make or break your RAG system, and exactly what to do about each one.

---

## 1. Chunking Strategy Is Not a Detail — It Is the Foundation

Most teams treat chunking as an afterthought. Pick a chunk size, split on newlines, move on. This is the single most common reason RAG systems produce bad answers.

**The problem with naive chunking:**
- Fixed-size character splits tear sentences in half. A chunk that ends mid-sentence gives the LLM an incomplete thought and the model either ignores it or hallucinates a completion.
- Chunks that are too large drown the relevant sentence in noise — the LLM's attention gets diluted across 800 words when only 40 of them answer the question.
- Chunks that are too small lose all context — a chunk containing only "The decision was approved." tells you nothing without the surrounding paragraph.

**What to do instead:**
- Split on sentence boundaries, not character counts. Use a target chunk size (e.g. 500 characters) as a soft limit, but never break mid-sentence.
- Add overlap between consecutive chunks — typically 10-15% of chunk size. This ensures that a fact sitting at the boundary of two chunks is fully represented in at least one of them.
- Preserve metadata per chunk: source document, page number, section heading. This context is invaluable for citation and debugging.

**See it live:** The [Advanced Local RAG Demo](https://vishalmysore.github.io/advancedRagDemo/) exposes **Chunk Size** and **Chunk Overlap** sliders directly in the UI. Upload a PDF, adjust the sliders, re-ingest, and watch the chunk count in the Ingested Corpus table change. Then run the same query at different chunk sizes and compare the retrieved sources — the difference in answer quality is immediate and visible.

---

## 2. Never Rely on Vector Search Alone

Vector embeddings capture semantic meaning. They are excellent at understanding that "myocardial infarction" and "heart attack" mean the same thing. They are poor at understanding that "Clause 4.2" and "Section 4.2" might refer to the same thing but "Article 4.2" in a different document does not.

Keyword search (BM25) captures exact term matches. It is excellent at finding documents that contain the precise phrase the user typed. It is poor at understanding synonyms or paraphrased questions.

**The solution is always hybrid.** Run both in parallel, fuse the scores:

```
fusedScore = α × vectorScore + (1-α) × normalizedBM25Score
```

Neither approach dominates universally. The right balance depends on your domain:
- **Legal, compliance, financial documents** → lean BM25. Exact clause references matter more than semantic proximity.
- **Conceptual knowledge bases, FAQs, support docs** → lean vector. Users paraphrase; exact terms vary.
- **Mixed corpora** → start at 50/50 and tune based on user feedback.

**See it live:** The demo runs both retrieval paths simultaneously and shows both scores on every retrieved source card — Vector Cosine and BM25 Score side by side. Use the **Weights slider** to shift the balance and observe which chunks rise and fall in ranking. Enable **Benchmark Mode** to run all three strategies — Vector Only, BM25 Only, and Hybrid — in parallel on the same query and compare results directly.

---

## 3. Reranking Is Not Optional at Any Scale

Your first-stage retrieval — whether ANN vector search or BM25 — is optimized for speed, not accuracy. It scores the query and each document chunk *independently*. It does not jointly reason about whether this specific query and this specific chunk are a good match.

A cross-encoder reranker does exactly that. It reads the query and the chunk together and outputs a single relevance score. It is slower (one model call per candidate chunk) but far more accurate.

**The standard two-stage pattern:**
1. Fast retrieval returns top 30 candidates in milliseconds
2. Reranker scores all 30, takes 150-300ms
3. Top K (typically 4-6) go into the LLM prompt

Skipping reranking means the chunks you feed to the LLM are ordered by a heuristic approximation of relevance, not actual relevance. The LLM will try to work with whatever it gets — and if the most relevant chunk is ranked 12th and the least relevant is ranked 1st, your answer quality suffers proportionally.

**Two reranking approaches worth knowing:**

*Syntactic proximity reranking:* No model required. Measures how closely together query keywords appear within each chunk. A chunk where "data" and "pipeline" appear within 3 words of each other scores higher than one where they appear 200 words apart. Fast, interpretable, zero cost.

*Neural cross-encoder reranking:* Uses a model like `ms-marco-MiniLM-L-6-v2` — trained specifically on query-passage relevance using the MS MARCO dataset. Understands paraphrasing, context, and intent in ways that no heuristic can.

**See it live:** Switch the **Stage 2 Reranking** dropdown between *None*, *Syntactic (Query Proximity Window)*, and *Neural (Cross-Encoder)*. The Observability Trace logs the reranking step explicitly, showing which chunks moved up or down and what their final confidence scores are. With Neural selected, the demo downloads and runs a real ONNX cross-encoder model entirely in your browser via WebAssembly — no server involved.

---

## 4. Source Confidence Scoring Must Gate Your Pipeline

Every chunk that enters your LLM prompt should carry a confidence score. Not as a decoration — as a gate.

If your retrieval system cannot find chunks above a minimum confidence threshold, **do not generate.** Return "The available documents do not contain sufficient information to answer this question." This is the system working correctly. A confident hallucinated answer is not better than an honest admission of insufficient information — it is far worse, especially in regulated domains.

**Confidence components to consider:**
- Retrieval score (fusion of vector + BM25)
- Source freshness (older documents get a decay penalty)
- Source authority (internal audited documents outrank random web imports)
- Cross-chunk agreement (multiple chunks saying the same thing raises confidence)

The specific weights depend on your domain. What matters is having the concept: retrieval has to earn the right to generate.

**See it live:** Every source card in the demo shows a **Confidence %** — the final weighted score after retrieval and reranking. Watch it change as you adjust chunk size, retrieval weights, and reranking method. The hallucination detection layer downstream uses this confidence as one of its inputs.

---

## 5. Your System Prompt Is an Engineering Artifact, Not an Afterthought

The system prompt is where you enforce the contract between retrieval and generation. It is the mechanism by which you prevent the LLM from supplementing retrieved context with its own training knowledge.

A weak system prompt:
```
Use the following documents to answer the question.
```

A production system prompt:
```
You are a citation-backed AI assistant. Answer using ONLY the 
provided Context sections. For every claim, include a [Source N] 
citation. If the Context does not contain the answer, respond with:
"The provided documents do not contain sufficient information to 
answer this question." Do NOT use knowledge from your training data.
Do NOT speculate beyond what the Context explicitly states.
```

The difference is specificity. Vague instructions produce vague compliance. Explicit rules — cite every claim, never use training data, use exact fallback phrasing when unsure — produce consistent, verifiable behavior.

**Temperature matters here too.** Set it to `0.0` or `0.1` for RAG. RAG is a precision task, not a creative one. Every degree of temperature you add increases the probability of the model drifting from the provided context.

**See it live:** The Observability Trace in the demo logs a `PROMPT ASSEMBLY` step for every query. This is where the retrieved chunks are formatted with `[Source N: filename (Page X)]` headers and injected into the constrained system prompt. The exact structure of this prompt is what drives the `[Source 1]` citation badges in the generated answer.

---

## 6. Citations Are Not a Feature — They Are a Requirement

A RAG system without verifiable citations is not a knowledge system — it is a text generator with a retrieval preprocessing step. Citations are what make the output auditable, trustworthy, and correctable.

**What citations enable:**
- Users can verify any claim against the source document
- Incorrect answers can be traced back to the exact chunk that caused them
- You can identify which documents are consistently producing bad outputs
- Compliance and audit teams have a paper trail for every AI-generated statement

**What good citations look like in practice:**
- Inline references in the response text: `The policy was updated in Q3 2023 [Source 2]`
- Each citation links to the exact source document, version, and page number
- The chunk text used is stored alongside the response — not just the document name

**See it live:** In the demo, generated answers contain clickable `[Source N]` badges. Click any badge and the page scrolls directly to the source card showing the exact chunk text, document name, page number, vector score, BM25 score, and confidence. This is the citation chain made tangible — from claim, to chunk, to document.

---

## 7. Build a Hallucination Detection Layer Before the Response Reaches Users

Constrained generation and good retrieval dramatically reduce hallucination. They do not eliminate it. LLMs can still:
- Paraphrase a chunk inaccurately
- Combine information from two chunks in a way that is not supported by either
- Generate a plausible-sounding number that appears nowhere in the context
- Synthesize an answer that is technically grounded but misleadingly incomplete

You need an automated detection pass that runs on every generated response before it is shown to the user.

**A practical three-pass approach:**

*Pass 1 — Assertion extraction:* Pull out all verifiable claims from the response. Numbers, dates, percentages, proper nouns, named entities, technical terms.

*Pass 2 — Grounding check:* For each extracted assertion, verify that it appears in the retrieved context. Fuzzy matching — the model paraphrases, so exact string match is too strict. Threshold: if the assertion is not findable with >70% similarity in any retrieved chunk, flag it.

*Pass 3 — Faithfulness score:*
```
faithfulness = verified_assertions / total_assertions
```
Below 0.8 with flagged assertions → surface a warning. Show the specific claims that could not be verified. Do not suppress the response — surfacing uncertainty is more honest and more useful than hiding it.

**See it live:** The demo computes faithfulness automatically on every query and displays the score in the **Evaluation Metrics** panel. When faithfulness drops below threshold, the **Hallucination Alert** banner appears above the generated answer, listing the specific unverified claims. This runs entirely in the browser — no external eval API.

---

## 8. Continuous Evaluation Is the Only Way to Know Your System Is Working

The most dangerous RAG failure mode is silent degradation. Your retrieval quality drops because new documents changed the index distribution. Your faithfulness score trends down because a recent model update changed generation behavior. Nobody notices because there are no evals running.

**The three metrics every RAG system must track continuously:**

**Context Relevance** — Are the retrieved chunks actually relevant to the query?
```
contextRelevance = queryTokens ∩ contextTokens / |queryTokens|
```
If this is low, your retrieval is broken. The LLM is working with the wrong information regardless of how good the model is.

**Faithfulness** — Does the generated response stay grounded in the retrieved context?
```
faithfulness = verified_claims / total_claims
```
If this is low, your generation constraints are too weak or your chunks are too noisy.

**Answer Relevance** — Does the response actually answer the question that was asked?
```
answerRelevance = queryTokens ∩ answerTokens / |queryTokens|
```
High context relevance but low answer relevance means the model is ignoring the retrieved context — a system prompt problem.

Track these per query, aggregate over rolling windows, alert on degradation. If faithfulness drops below 0.75 over a 1-hour window, something has changed and someone needs to look at it.

**See it live:** The demo computes and displays all three metrics live after every query — no API call, no external service. Watch how the scores change as you adjust chunk size, retrieval weights, and reranking method. This is the fastest feedback loop available for developing intuition about what actually moves these numbers.

---

## 9. Caching and Memory Are Force Multipliers

In any production RAG system with real user traffic, a significant percentage of queries are semantically identical or near-identical to queries that have already been answered. Recomputing the full pipeline — embedding generation, ANN search, BM25 scan, fusion, reranking, LLM call — for every one of these is wasteful and adds unnecessary latency.

**Two levels of caching to implement:**

*Exact query cache:* Hash the query + retrieval config + model identifier. If seen before and the source documents haven't changed since the cache was written, return the cached result. Latency drops from seconds to milliseconds.

*Semantic cache:* For incoming queries, check if a semantically similar query (cosine similarity > 0.95) has been answered recently. Return the cached result with a note. Handles paraphrased repetitions — "What is the refund policy?" and "How do I get a refund?" often have the same answer.

**Memory beyond caching:**

Session memory keeps track of what has been discussed in a conversation, so users don't have to repeat context across turns.

Long-term correction memory is more powerful: when a human expert flags an answer as wrong and provides the correct information, store that correction tagged with the query topic and domain. On future similar queries, retrieve and inject the correction into the system prompt. This is how your RAG system accumulates institutional knowledge over time without retraining.

**See it live:** Run a query in the demo, then run the exact same query again. The Observability Trace shows `CACHE HIT` on the second run and returns in milliseconds. Toggle **Use Retrieval Cache** off to force a fresh pipeline run and see the full latency comparison. The cache is stored in IndexedDB — it persists across page refreshes and survives until you explicitly clear it.

---

## 10. Observability Is Not Logging — It Is Understanding What Your Pipeline Is Doing

Most teams add logging as an afterthought. They log errors. They log the final response. They have no visibility into what happened in between — which chunks were retrieved, what scores they carried, why the reranker promoted chunk 12 above chunk 1, what the system prompt looked like, whether the eval passed.

When something goes wrong at 2am — and it will go wrong — you need to reconstruct the exact execution path of any query within minutes.

**What to instrument on every query:**
- Timing for every stage: embedding generation, vector search, BM25 search, fusion, reranking, LLM call, eval
- Which documents were retrieved and their scores at each stage
- The exact system prompt sent to the model
- The raw model response before any post-processing
- Eval scores: context relevance, faithfulness, answer relevance
- Cache hit or miss, and if miss, why (new query vs invalidated cache)
- Any hallucination flags and the specific claims that triggered them

The observability layer is also your primary debugging tool during development. Without it, RAG tuning is guesswork. With it, you can look at a bad answer and immediately see: the retrieval scores were low (retrieval problem), or the retrieval was good but faithfulness was 0.4 (generation constraint problem), or the chunk sizes are too small (chunking problem).

**See it live:** The **RAG Pipeline Observability Trace** panel in the demo is a working implementation of this principle. Every stage logs what it did, in what order, at what time, and with what outcome — color-coded by layer type. This is the fastest way to develop an intuition for what a well-instrumented RAG pipeline looks like before you build one.

---

## The One Thing That Ties All 10 Together

Each of these 10 areas is a distinct engineering concern. But they share a common thread: **RAG is a pipeline, not a model call.** The quality of the final answer is determined by the weakest link in the chain — bad chunking, missing reranking, no confidence gating, a vague system prompt, no evals — any one of these can neutralize the best LLM on the market.

The teams that build reliable RAG systems are not the ones with access to the most powerful models. They are the ones who obsess over each layer of the pipeline, instrument everything, measure continuously, and treat retrieval quality as seriously as generation quality.

---

## Explore the Full Pipeline Live

The [Advanced Local RAG Demo](https://vishalmysore.github.io/advancedRagDemo/) is a fully working implementation of all 10 principles described in this article — running entirely in your browser, with no backend, no cloud costs, and no data leaving your device.

**What you can do with it:**

- Upload any PDF or text file and watch the chunking, embedding, and BM25 indexing happen in real time
- Tune chunk size and overlap and observe how it changes retrieval quality on the same query
- Switch between Vector Only, BM25 Only, and Hybrid retrieval in Benchmark Mode and compare results side by side
- Toggle between Syntactic and Neural reranking and see how chunk ordering changes
- Watch the Observability Trace log every pipeline stage with timestamps and scores
- See Faithfulness, Context Relevance, and Answer Relevance computed live after every query
- Click citation badges in the answer to trace any claim back to its exact source chunk
- Observe the Hallucination Alert fire when generated claims cannot be grounded in retrieved context
- Test the caching layer by running the same query twice and comparing latencies

Every slider, toggle, and dropdown in the demo corresponds directly to one of the 10 engineering decisions described in this article. It is the fastest feedback loop available for building an intuition about what actually matters in a RAG pipeline — before you commit to a production architecture.

---

*The demo is built with Vite, Transformers.js (ONNX/WASM), Dexie/IndexedDB, and a hand-rolled BM25 engine. Source available at [github.com/vishalmysore/advancedRagDemo](https://github.com/vishalmysore/advancedRagDemo).*
