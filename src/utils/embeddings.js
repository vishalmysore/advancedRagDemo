// embeddings.js — Transformers.js wrapper for local embedding generation and reranking
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'

// Configure env to run in pure browser environment
env.allowLocalModels = false; // Fetch models from Hugging Face hub
env.useBrowserCache = true;   // Cache files in browser's Cache API

let embeddingPipeline = null
let rerankerPipeline = null

// Helper to standardise vector normalization (for cosine similarity)
function dotProduct(a, b) {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i]
  return dot
}

function magnitude(a) {
  let sum = 0
  for (let i = 0; i < a.length; i++) sum += a[i] * a[i]
  return Math.sqrt(sum)
}

export function normalizeVector(vector) {
  const mag = magnitude(vector)
  if (mag === 0) return vector
  return vector.map(val => val / mag)
}

/**
 * Load the embedding pipeline and report download progress.
 * @param {Function} progressCallback - Called with { status, progress, file }
 */
export async function loadEmbeddingModel(progressCallback) {
  if (embeddingPipeline) return embeddingPipeline

  // Configure progress listener
  const onProgress = (event) => {
    if (event.status === 'progress' && progressCallback) {
      progressCallback(event)
    }
  }

  embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    progress_callback: onProgress
  })

  return embeddingPipeline
}

/**
 * Generate embedding for a single text chunk.
 * @param {string} text
 * @returns {Promise<number[]>} Float array of normalized embedding vector
 */
export async function getEmbedding(text) {
  const pipe = await loadEmbeddingModel()
  
  // Clean text to avoid empty content causing pipeline failure
  const cleanText = text.trim() || " "
  
  const output = await pipe(cleanText, {
    pooling: 'mean',
    normalize: true
  })

  // Convert Float32Array to standard array
  return Array.from(output.data)
}

/**
 * Load the Cross-Encoder Reranker pipeline.
 * @param {Function} progressCallback - Called with download progress events
 */
export async function loadRerankerModel(progressCallback) {
  if (rerankerPipeline) return rerankerPipeline

  const onProgress = (event) => {
    if (event.status === 'progress' && progressCallback) {
      progressCallback(event)
    }
  }

  rerankerPipeline = await pipeline('text-classification', 'Xenova/ms-marco-MiniLM-L-6-v2', {
    progress_callback: onProgress
  })

  return rerankerPipeline
}

/**
 * Rerank a document chunk against a query using the Cross-Encoder.
 * @param {string} query
 * @param {string} chunkText
 * @returns {Promise<number>} Relevance score between 0 and 1
 */
export async function getCrossEncoderScore(query, chunkText) {
  const pipe = await loadRerankerModel()
  const result = await pipe(query, {
    text_pair: chunkText
  })
  
  // ms-marco-MiniLM-L-6-v2 outputs a classification. We get the score for LABEL_1 (relevant)
  // Usually result is array: [{ label: 'LABEL_1', score: 0.92 }] or [{ label: 'LABEL_0', score: 0.08 }, { label: 'LABEL_1', score: 0.92 }]
  const label1Obj = result.find(r => r.label === 'LABEL_1')
  if (label1Obj) return label1Obj.score

  // Fallback: check first element
  if (result && result[0]) {
    // If it labeled LABEL_0 (irrelevant), score is 1 - LABEL_0 score
    if (result[0].label === 'LABEL_0') return 1 - result[0].score
    return result[0].score
  }

  return 0
}

/**
 * Computes cosine similarity between two vectors.
 * If vectors are normalized, it is simply the dot product.
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0
  return dotProduct(vecA, vecB)
}
