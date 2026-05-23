// ragEngine.js — Core orchestrator for the advanced browser-native RAG pipeline
import { db } from './db.js'
import { getEmbedding, cosineSimilarity, getCrossEncoderScore } from './embeddings.js'
import { indexChunkInTransaction, tokenize, searchBM25 } from './bm25.js'
import { evaluateAnswerFaithfulness, evaluateContextRelevance, evaluateAnswerRelevance } from './eval.js'
import { callLLM, setLLMConfig } from './llm.js'

/**
 * MD5-like string hashing helper to create unique cache keys.
 */
function hashString(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i)
    hash |= 0 // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36)
}

/**
 * Standard text normalisation: cleans multiple spaces, strip control chars, normalize Unicode.
 */
export function normalizeText(text) {
  if (!text) return ''
  return text
    .normalize('NFC')
    .replace(/[\r\n]+/g, '\n') // standardise newlines
    .replace(/[^\x20-\x7E\n]/g, '') // remove non-printable ASCII (optional, keeps readable chars)
    .replace(/\s+/g, ' ') // standardise whitespaces
    .trim()
}

/**
 * Splits text into sentences using regex boundary checks.
 */
function splitIntoSentences(text) {
  // Split on periods, exclamation marks, or question marks followed by spaces
  const sentences = text.match(/[^.!?]+[.!?]+(?:\s|$)/g) || [text]
  return sentences.map(s => s.trim()).filter(s => s.length > 0)
}

/**
 * Chunk documents into sliding windows, respecting sentence boundaries.
 */
export function chunkDocument(text, chunkSize = 500, chunkOverlap = 50) {
  const normalized = normalizeText(text)
  const sentences = splitIntoSentences(normalized)
  
  const chunks = []
  let currentChunkText = ''
  let currentChunkSentences = []

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i]
    
    if ((currentChunkText + ' ' + sentence).length > chunkSize && currentChunkText.length > 0) {
      // Save current chunk
      chunks.push(currentChunkText.trim())
      
      // Determine overlap. Backtrack sentences to satisfy overlap length
      let overlapText = ''
      const overlapSentences = []
      for (let j = currentChunkSentences.length - 1; j >= 0; j--) {
        const oSentence = currentChunkSentences[j]
        if ((overlapText + ' ' + oSentence).length <= chunkOverlap) {
          overlapText = oSentence + ' ' + overlapText
          overlapSentences.unshift(oSentence)
        } else {
          break
        }
      }
      
      currentChunkSentences = overlapSentences
      currentChunkText = overlapText
    }
    
    currentChunkText += (currentChunkText.length > 0 ? ' ' : '') + sentence
    currentChunkSentences.push(sentence)
  }

  if (currentChunkText.trim().length > 0) {
    chunks.push(currentChunkText.trim())
  }

  return chunks
}

/**
 * Process and ingest an uploaded document into the RAG database.
 * @param {string} docName 
 * @param {string} docType 
 * @param {Array<{ pageNumber: number, text: string }>} pages 
 * @param {object} config - Chunker settings: { chunkSize, chunkOverlap }
 * @param {Function} onProgress - Progress reporting callback (currentChunk, totalChunks)
 */
export async function ingestDocument(docName, docType, pages, config, onProgress) {
  const docId = 'doc_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5)
  
  // 1. First, preprocess and collect all chunks across pages
  const processedChunks = []
  let totalWords = 0

  for (const page of pages) {
    const textChunks = chunkDocument(page.text, config.chunkSize, config.chunkOverlap)
    for (let index = 0; index < textChunks.length; index++) {
      const text = textChunks[index]
      const tokenCount = text.split(/\s+/).length
      totalWords += tokenCount
      
      processedChunks.push({
        id: `${docId}_${processedChunks.length}`,
        documentId: docId,
        text,
        pageNumber: page.pageNumber,
        tokenCount
      })
    }
  }

  if (processedChunks.length === 0) {
    throw new Error('No text content could be extracted or chunked from this document.')
  }

  // 2. Generate local vector embeddings for chunks (can take a while)
  const totalChunks = processedChunks.length
  const chunksWithEmbeddings = []

  for (let i = 0; i < totalChunks; i++) {
    const chunk = processedChunks[i]
    if (onProgress) {
      onProgress(i + 1, totalChunks, 'generating_embeddings')
    }
    
    // Generate normalized vector embedding
    const embedding = await getEmbedding(chunk.text)
    chunksWithEmbeddings.push({
      ...chunk,
      embedding
    })
  }

  // 3. Write metadata to documents table and chunks/indexes to database inside a transaction
  await db.transaction('rw', [db.documents, db.chunks, db.bm25Index, db.cache], async () => {
    // Save document metadata
    await db.documents.put({
      id: docId,
      name: docName,
      type: docType,
      size: processedChunks.reduce((acc, c) => acc + c.text.length, 0),
      uploadDate: new Date().toISOString(),
      chunkCount: totalChunks,
      totalTokens: totalWords
    })

    // Save chunks
    for (const chunk of chunksWithEmbeddings) {
      await db.chunks.put(chunk)
      // Index in custom BM25 index within this transaction
      await indexChunkInTransaction(chunk.id, chunk.text)
    }

    // Clear RAG response cache since indices changed
    await db.cache.clear()
  })

  return docId
}

// ── Syntactic Reranker Helper ─────────────────────────────────────

/**
 * Calculates syntactic keyword proximity score.
 * Measures how close query keywords are to each other inside the chunk.
 * Chunks where keywords cluster tightly together rank higher.
 */
function calculateProximityScore(queryText, chunkText) {
  const queryTokens = tokenize(queryText)
  if (queryTokens.length <= 1) return 0 // Proximity needs multiple terms

  const textWords = chunkText.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/)
  const positions = {}
  
  // Find index positions of query tokens inside the chunk
  for (const token of queryTokens) {
    positions[token] = []
    for (let i = 0; i < textWords.length; i++) {
      if (textWords[i] === token) {
        positions[token].push(i)
      }
    }
  }

  // Calculate smallest window containing all matching terms
  const presentTokens = queryTokens.filter(t => positions[t].length > 0)
  if (presentTokens.length < 2) return 0 // Not enough terms present

  let minWindow = Infinity
  
  // Heuristic slider-window: find closest proximity
  // For small queries, we check distances between matching word indices
  const combinations = [[]]
  for (const token of presentTokens) {
    const currentPositions = positions[token]
    const nextCombinations = []
    for (const comb of combinations) {
      for (const pos of currentPositions) {
        nextCombinations.push([...comb, pos])
      }
    }
    // Limit exponential growth of combos
    if (nextCombinations.length > 100) break
    combinations.length = 0
    combinations.push(...nextCombinations)
  }

  for (const comb of combinations) {
    if (comb.length < 2) continue
    const max = Math.max(...comb)
    const min = Math.min(...comb)
    const span = max - min
    if (span < minWindow) minWindow = span
  }

  if (minWindow === Infinity) return 0
  
  // Normalize score: small window = high score, 1-word gap = 1.0, 50-word gap = ~0
  const termDensity = presentTokens.length / queryTokens.length
  const proximityBonus = 1 / (1 + minWindow / 5) // standard decay
  return termDensity * 0.6 + proximityBonus * 0.4
}

// ── Hybrid Retrieval & Orchestration ──────────────────────────────

/**
 * Orchestrate the complete Hybrid Retrieval, Reranking, Generation, and Eval flow.
 * Supports caching, logging, and benchmarking logs.
 */
export async function executeRAGQuery(query, config, traceCallback) {
  const {
    vectorWeight = 0.5,
    bm25Weight = 0.5,
    topK = 5,
    rerankType = 'proximity', // 'none' | 'proximity' | 'neural'
    useCache = true,
    llmProvider = 'mock',
    llmModel = 'mock-rag-agent',
    apiKey = '',
    proxyUrl = ''
  } = config

  // Synchronise global LLM settings right before calling provider APIs
  setLLMConfig({ provider: llmProvider, model: llmModel, apiKey, proxyUrl })

  const trace = {
    timings: {},
    steps: [],
    evals: {},
    cached: false
  }

  const logStep = (name, detail) => {
    trace.steps.push({ name, timestamp: Date.now(), detail })
    if (traceCallback) traceCallback(name, detail)
  }

  // 1. Check Cache Layer
  const cacheKey = hashString(JSON.stringify({ query, vectorWeight, bm25Weight, topK, rerankType, llmProvider, llmModel }))
  if (useCache) {
    const cachedResult = await db.cache.get(cacheKey)
    if (cachedResult) {
      logStep('Cache Hit', 'Found query results in IndexedDB cache.')
      trace.cached = true
      return {
        ...cachedResult,
        trace: { ...trace, cached: true }
      }
    }
  }

  // 2. Vector Search (Stage 1 - ANN)
  const startVector = Date.now()
  logStep('Embedding Generation', `Generating embedding vector for query: "${query}"`)
  const queryVector = await getEmbedding(query)
  trace.timings.embeddingGen = Date.now() - startVector

  logStep('Vector Retrieval', 'Scanning IndexedDB chunks for cosine similarity...')
  const allChunks = await db.chunks.toArray()
  const vectorResults = []
  
  for (const chunk of allChunks) {
    const sim = cosineSimilarity(queryVector, chunk.embedding)
    vectorResults.push({ chunkId: chunk.id, score: sim })
  }
  vectorResults.sort((a, b) => b.score - a.score)
  const topVector = vectorResults.slice(0, 30) // select top 30 for fusion
  logStep('Vector Results Ready', `Retrieved ${topVector.length} candidates. Top score: ${topVector[0]?.score?.toFixed(4) || 0}`)

  // 3. BM25 Search (Stage 1 - Keyword Index Scan)
  const startBm25 = Date.now()
  logStep('BM25 Keyword Retrieval', 'Querying inverted indexes in IndexedDB...')
  
  const bm25Results = await searchBM25(query, 30)
  trace.timings.bm25Search = Date.now() - startBm25
  logStep('BM25 Results Ready', `Retrieved ${bm25Results.length} candidates. Top score: ${bm25Results[0]?.score?.toFixed(4) || 0}`)

  // 4. Hybrid Score Fusion
  logStep('Hybrid Fusion', `Fusing results with weights: Vector (${vectorWeight}) + BM25 (${bm25Weight})`)
  
  // Normalize BM25 scores to [0,1] using standard scaling
  const maxBm25 = bm25Results[0]?.score || 1
  const bm25Map = new Map(bm25Results.map(r => [r.chunkId, r.score / maxBm25]))
  const vectorMap = new Map(vectorResults.map(r => [r.chunkId, r.score]))

  const allCandidateIds = new Set([...bm25Map.keys(), ...topVector.map(r => r.chunkId)])
  let fusedResults = []

  for (const cid of allCandidateIds) {
    const vScore = vectorMap.get(cid) || 0
    const bScore = bm25Map.get(cid) || 0
    
    // Linear Fusion
    const fusedScore = (vectorWeight * vScore) + (bm25Weight * bScore)
    fusedResults.push({
      chunkId: cid,
      score: fusedScore,
      vectorScore: vScore,
      bm25Score: bScore * maxBm25 // show raw BM25 in UI
    })
  }
  
  fusedResults.sort((a, b) => b.score - a.score)
  let candidateChunks = fusedResults.slice(0, 15) // Top 15 go to reranking

  // Fetch full chunk objects from DB
  const docMeta = await db.documents.toArray()
  const docMap = new Map(docMeta.map(d => [d.id, d.name]))

  for (const item of candidateChunks) {
    const rawChunk = await db.chunks.get(item.chunkId)
    item.text = rawChunk.text
    item.documentId = rawChunk.documentId
    item.docName = docMap.get(rawChunk.documentId) || 'Document'
    item.pageNumber = rawChunk.pageNumber
  }

  // 5. Stage 2 - Reranking
  const startRerank = Date.now()
  if (rerankType === 'proximity') {
    logStep('Syntactic Reranking', 'Reranking candidate chunks using Query-Term Proximity windowing...')
    for (const item of candidateChunks) {
      const prox = calculateProximityScore(query, item.text)
      // Fuse syntactic density with original hybrid score
      item.rerankedScore = item.score * 0.4 + prox * 0.6
      item.confidence = item.rerankedScore
    }
    candidateChunks.sort((a, b) => b.rerankedScore - a.rerankedScore)
  } else if (rerankType === 'neural') {
    logStep('Neural Reranking', 'Loading ONNX Cross-Encoder model (ms-marco-MiniLM-L-6-v2) for relevance scoring...')
    for (const item of candidateChunks) {
      const neuralScore = await getCrossEncoderScore(query, item.text)
      item.rerankedScore = neuralScore
      item.confidence = neuralScore
    }
    candidateChunks.sort((a, b) => b.rerankedScore - a.rerankedScore)
  } else {
    logStep('No Reranking', 'Skipping second-stage reranking. Using hybrid score.')
    for (const item of candidateChunks) {
      item.confidence = item.score // raw confidence
    }
  }
  trace.timings.reranking = Date.now() - startRerank
  
  // Select top-K chunks for prompt context
  const retrievedChunks = candidateChunks.slice(0, topK)
  logStep('Retrieval Completed', `Selected top ${retrievedChunks.length} chunks. Context confidence: ${(retrievedChunks[0]?.confidence * 100).toFixed(1)}%`)

  // 6. Constrained Prompt Assembly
  logStep('Prompt Assembly', 'Building context injection rules (constrained generation)...')
  const contextStr = retrievedChunks.map((c, i) => `[Source ${i+1}: ${c.docName} (Page ${c.pageNumber})]\n${c.text}`).join('\n---\n')
  
  const systemPrompt = `You are a high-accuracy, citation-backed AI assistant. Answer the user's query relying ONLY on the provided context sections.
  
Rules:
1. Ground every claim you make.
2. For every assertion or detail you mention, include a numerical citation referencing which Source it came from, in the format "[Source X]".
3. Do NOT make any claims that are not explicitly present in the provided Context.
4. If the context does not contain the information needed to answer the query, reply with: "I am sorry, but the provided documents do not contain the information required to answer this query." Do not attempt to synthesize an answer from external knowledge.

Context:
---
${contextStr}
---`

  const messages = [
    { role: 'user', content: query }
  ]

  // 7. LLM Call
  const startLLM = Date.now()
  logStep('LLM Call Initiated', `Sending prompt to LLM (${llmProvider}/${llmModel})...`)
  const llmResult = await callLLM(messages, systemPrompt, retrievedChunks)
  trace.timings.llmGeneration = Date.now() - startLLM
  logStep('LLM Response Received', `Completed in ${trace.timings.llmGeneration}ms. Extracting response...`)

  // 8. Continuous Evaluation & Hallucination Check
  logStep('Evaluation Engine', 'Evaluating generated answer relevance, context overlap, and checking for hallucinations...')
  const startEval = Date.now()
  
  const contextRelevance = evaluateContextRelevance(query, contextStr)
  const answerRelevance = evaluateAnswerRelevance(query, llmResult.text)
  const { score: faithfulness, mismatches } = evaluateAnswerFaithfulness(llmResult.text, contextStr)
  
  trace.timings.evaluations = Date.now() - startEval
  trace.evals = {
    contextRelevance,
    answerRelevance,
    faithfulness,
    mismatches
  }

  // Hallucination Warning Threshold check
  const hallucinationFlag = faithfulness < 0.8 && mismatches.length > 0
  if (hallucinationFlag) {
    logStep('Hallucination Alert', `Detected possible hallucinations: ${mismatches.join('; ')}`)
  } else {
    logStep('Evaluations OK', 'Groundedness verification succeeded.')
  }

  // Calculate overall query context confidence
  const avgConfidence = retrievedChunks.reduce((acc, c) => acc + c.confidence, 0) / (retrievedChunks.length || 1)

  const finalOutput = {
    query,
    answer: llmResult.text,
    sources: retrievedChunks,
    metrics: {
      latencyMs: Object.values(trace.timings).reduce((a, b) => a + b, 0),
      confidence: avgConfidence,
      ...trace.evals
    },
    hallucinationWarning: hallucinationFlag,
    mismatches: mismatches
  }

  // 9. Save to Cache
  if (useCache && !hallucinationFlag) {
    await db.cache.put({
      queryKey: cacheKey,
      query,
      answer: finalOutput.answer,
      sources: finalOutput.sources,
      metrics: finalOutput.metrics,
      hallucinationWarning: false,
      mismatches: [],
      timestamp: Date.now()
    })
  }

  return {
    ...finalOutput,
    trace: { ...trace, timings: { ...trace.timings, total: Date.now() - startVector } }
  }
}
