// bm25.js — Custom BM25 Indexer & Retriever for browser-native keyword search
import { db } from './db.js'

// Simple list of common English stopwords
const STOP_WORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
  'can', 'cant', 'cannot', 'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during',
  'each', 'few', 'for', 'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 
  'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 
  'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 
  'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 
  'out', 'over', 'own', 'same', 'shan' , 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 
  'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 
  'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 
  'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 
  'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 
  'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
])

/**
 * Clean and split text into lowercase words, removing stopwords and punctuation.
 * @param {string} text 
 * @returns {string[]} List of tokens
 */
export function tokenize(text) {
  if (!text) return []
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '') // remove punctuation except hyphens
    .split(/\s+/)
    .filter(token => token.length > 1 && !STOP_WORDS.has(token))
}

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
