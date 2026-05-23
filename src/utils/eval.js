// eval.js — Browser-native evaluation metrics for RAG
import { tokenize } from './bm25.js'

/**
 * Calculates Context Relevance (overlap of query tokens in the retrieved context).
 * @param {string} query 
 * @param {string} context 
 * @returns {number} Score between 0 and 1
 */
export function evaluateContextRelevance(query, context) {
  const queryTokens = tokenize(query)
  if (queryTokens.length === 0) return 1

  const contextLower = context.toLowerCase()
  let matches = 0

  for (const token of queryTokens) {
    if (contextLower.includes(token)) {
      matches++
    }
  }

  return matches / queryTokens.length
}

/**
 * Evaluates Answer Faithfulness (groundedness).
 * Extracts assertions/numbers/key-nouns from the answer and verifies if they exist in the context.
 * Useful for catching hallucinated dates, numbers, or terms.
 * @param {string} answer 
 * @param {string} context 
 * @returns {{ score: number, mismatches: string[] }}
 */
export function evaluateAnswerFaithfulness(answer, context) {
  if (!answer) return { score: 1, mismatches: [] }
  if (!context) return { score: 0, mismatches: ['Entire context is empty'] }

  // 1. Extract alphanumeric tokens, numbers, and potential entities
  const contextLower = context.toLowerCase()
  
  // Find all numbers (like 123, 4.5, 2026) in the answer
  const answerNumbers = answer.match(/\b\d+(?:\.\d+)?\b/g) || []
  
  // Find all unique capitalized words in the answer (often entities, names, etc.)
  // (Excluding the very first word of sentences for simplicity)
  const sentences = answer.split(/[.!?]+/)
  const answerEntities = new Set()
  for (const sentence of sentences) {
    const words = sentence.trim().split(/\s+/)
    // Skip first word, inspect subsequent ones
    for (let i = 1; i < words.length; i++) {
      const word = words[i].replace(/[^\w]/g, '')
      if (word && word[0] === word[0].toUpperCase() && word[0] !== word[0].toLowerCase()) {
        answerEntities.add(word.toLowerCase())
      }
    }
  }

  // Find all nouns / technical terms (tokens of length > 4 that aren't stopwords)
  const answerTerms = tokenize(answer).filter(t => t.length > 4)

  const mismatches = []
  let totalChecks = 0
  let matchedChecks = 0

  // Check numbers
  const uniqueNumbers = [...new Set(answerNumbers)]
  for (const num of uniqueNumbers) {
    totalChecks++
    // We search for the number as a word boundary in context
    const numRegex = new RegExp(`\\b${num.replace('.', '\\.')}\\b`)
    if (numRegex.test(context)) {
      matchedChecks++
    } else {
      mismatches.push(`Number "${num}" not found in sources`)
    }
  }

  // Check entities
  for (const entity of answerEntities) {
    totalChecks++
    if (contextLower.includes(entity)) {
      matchedChecks++
    } else {
      mismatches.push(`Entity "${entity}" not found in sources`)
    }
  }

  // Check key technical terms
  const uniqueTerms = [...new Set(answerTerms)].slice(0, 15) // limit to top 15 key terms
  for (const term of uniqueTerms) {
    // skip if already checked under entities
    if (answerEntities.has(term)) continue

    totalChecks++
    if (contextLower.includes(term)) {
      matchedChecks++
    } else {
      mismatches.push(`Key term "${term}" not found in sources`)
    }
  }

  if (totalChecks === 0) return { score: 1, mismatches: [] }

  const score = matchedChecks / totalChecks
  return { score, mismatches }
}

/**
 * Evaluates Answer Relevance (overlap between user query and generated response).
 * @param {string} query 
 * @param {string} answer 
 * @returns {number} Score between 0 and 1
 */
export function evaluateAnswerRelevance(query, answer) {
  const queryTokens = tokenize(query)
  if (queryTokens.length === 0) return 1

  const answerLower = answer.toLowerCase()
  let matches = 0

  for (const token of queryTokens) {
    if (answerLower.includes(token)) {
      matches++
    }
  }

  return matches / queryTokens.length
}
