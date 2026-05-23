// db.js — IndexedDB wrapper using Dexie.js
import Dexie from 'https://cdn.jsdelivr.net/npm/dexie@4.0.8/+esm'

export const db = new Dexie('AdvancedRAGDB')

// Define schema:
// - documents: stores original metadata about files
// - chunks: stores text pieces, embeddings, and page info. We index documentId.
// - bm25Index: inverted index mapping term to an object of chunk frequencies
// - cache: caches queries and full RAG responses to avoid repeated LLM calls
// - chatHistory: stores multi-turn conversation messages
db.version(1).stores({
  documents: 'id, name, type, size, uploadDate',
  chunks: 'id, documentId',
  bm25Index: 'term',
  cache: 'queryKey, timestamp',
  chatHistory: '++id, sessionId, timestamp'
})

// Database helper functions
export async function clearAllData() {
  await db.transaction('rw', [db.documents, db.chunks, db.bm25Index, db.cache, db.chatHistory], async () => {
    await db.documents.clear()
    await db.chunks.clear()
    await db.bm25Index.clear()
    await db.cache.clear()
    await db.chatHistory.clear()
  })
}

export async function deleteDocument(docId) {
  await db.transaction('rw', [db.documents, db.chunks, db.bm25Index, db.cache], async () => {
    // 1. Delete document metadata
    await db.documents.delete(docId)

    // 2. Delete all chunks belonging to this document
    await db.chunks.where('documentId').equals(docId).delete()

    // 3. Remove this document's chunks from the BM25 index
    // We scan terms and filter out matching chunk IDs
    await db.bm25Index.toCollection().modify((record) => {
      if (record.docCounts) {
        let changed = false
        for (const chunkId in record.docCounts) {
          if (chunkId.startsWith(docId + '_')) {
            delete record.docCounts[chunkId]
            changed = true
          }
        }
        if (changed) {
          // If no chunks left for this term, we can delete or keep empty.
          // Dexie modify deletes properties if we modify the record.
        }
      }
    })

    // Remove terms with empty docCounts to keep index clean
    const emptyTerms = await db.bm25Index.filter(r => Object.keys(r.docCounts || {}).length === 0).toArray()
    for (const termRecord of emptyTerms) {
      await db.bm25Index.delete(termRecord.term)
    }

    // 4. Clear cache because index changes
    await db.cache.clear()
  })
}

export async function getDocumentStats() {
  const docCount = await db.documents.count()
  const chunkCount = await db.chunks.count()
  const docs = await db.documents.toArray()
  return { docCount, chunkCount, docs }
}
