// bm25.js — Custom BM25 Indexer & Retriever for browser-native keyword search
import { db } from './db.js'

import { tokenize, STOP_WORDS } from './text.js'
export { tokenize }

/**
 * Incrementally index chunk text into the bm25Index Dexie table.
 * Designed to run within a Dexie transaction for speed and safety.
 * @param {string} chunkId 
 * @param {string} text 
 * @param {object} transaction - Active Dexie transaction
 */
export async function indexChunkInTransaction(chunkId, text, transaction) {
  const tokens = tokenize(text)
  if (tokens.length === 0) return

  // Count term frequencies within this chunk
  const termCounts = {}
  for (const token of tokens) {
    termCounts[token] = (termCounts[token] || 0) + 1
  }

  // Batch update terms in IndexedDB
  const uniqueTerms = Object.keys(termCounts)
  
  // Fetch existing index records for these terms
  const existingRecords = await db.bm25Index.where('term').anyOf(uniqueTerms).toArray()
  const recordMap = new Map(existingRecords.map(r => [r.term, r]))

  for (const term of uniqueTerms) {
    const freq = termCounts[term]
    let record = recordMap.get(term)
    
    if (!record) {
      record = { term, docCounts: {} }
    }
    
    record.docCounts[chunkId] = freq
    await db.bm25Index.put(record)
  }
}

/**
 * Perform BM25 search for a given query over all chunks in Dexie.
 * @param {string} queryText 
 * @param {number} limit 
 * @returns {Promise<Array<{ chunkId: string, score: number }>>}
 */
export async function searchBM25(queryText, limit = 20) {
  const queryTokens = tokenize(queryText)
  if (queryTokens.length === 0) return []

  // 1. Fetch total document (chunk) count and average length
  const totalChunks = await db.chunks.count()
  if (totalChunks === 0) return []

  // To calculate avgdl, we fetch all chunks' tokenCount
  // We can optimize this by storing stats elsewhere or fetching tokenCounts in bulk
  const allChunksMeta = await db.chunks.toArray()
  let totalLength = 0
  const chunkLengths = {}
  for (const chunk of allChunksMeta) {
    const len = chunk.tokenCount || tokenize(chunk.text).length || 1
    chunkLengths[chunk.id] = len
    totalLength += len
  }
  const avgdl = totalLength / totalChunks

  // 2. Fetch inverted index records for query terms
  const indexRecords = await db.bm25Index.where('term').anyOf(queryTokens).toArray()
  
  // 3. BM25 calculation parameters
  const k1 = 1.2
  const b = 0.75
  
  const chunkScores = {} // Map of chunkId -> BM25 score

  // Calculate scores
  for (const record of indexRecords) {
    const term = record.term
    const docCounts = record.docCounts || {}
    const df = Object.keys(docCounts).length // Document frequency of term

    if (df === 0) continue

    // IDF calculation with positive-only scaling
    const idf = Math.log(1 + (totalChunks - df + 0.5) / (df + 0.5))

    for (const chunkId in docCounts) {
      if (!chunkLengths[chunkId]) continue // Skip if chunk was deleted

      const tf = docCounts[chunkId]
      const docLen = chunkLengths[chunkId]

      // Standard BM25 formula
      const score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / avgdl))))

      chunkScores[chunkId] = (chunkScores[chunkId] || 0) + score
    }
  }

  // 4. Format and sort results
  const results = Object.keys(chunkScores).map(chunkId => ({
    chunkId,
    score: chunkScores[chunkId]
  }))

  results.sort((a, b) => b.score - a.score)
  return results.slice(0, limit)
}
