// llm.js — WebLLM client: runs LLM inference locally in the browser via WebGPU.
// No cloud API keys or CORS proxies required.

export const WEBLLM_MODELS = [
  { id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',  name: 'Llama 3.2 1B  (~0.9 GB) — fastest' },
  { id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',  name: 'Qwen 2.5 1.5B (~1.1 GB) — fast' },
  { id: 'gemma-2-2b-it-q4f16_1-MLC',           name: 'Gemma 2 2B   (~1.5 GB) — balanced' },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',  name: 'Llama 3.2 3B  (~2.0 GB) — good' },
  { id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',  name: 'Phi-3.5 Mini  (~2.2 GB) — best quality' },
]

// ── Worker singleton ──────────────────────────────────────────────

let _worker      = null
let _status      = 'idle'   // idle | loading | ready | error
let _modelId     = null
let _genCounter  = 0

// Pending promise callbacks keyed by type
let _loadResolve = null
let _loadReject  = null
let _genResolve  = null
let _genReject   = null

// Progress callback set by loadModel caller
let _onProgress  = null

function _ensureWorker() {
  if (_worker) return
  _worker = new Worker(new URL('../worker.js', import.meta.url), { type: 'module' })
  _worker.onmessage = _handleWorkerMessage
  _worker.onerror   = (e) => {
    _status = 'error'
    const msg = e.message ?? 'Worker crashed'
    if (_loadReject)  { _loadReject(new Error(msg));  _loadResolve = _loadReject = null }
    if (_genReject)   { _genReject(new Error(msg));   _genResolve  = _genReject  = null }
  }
}

function _handleWorkerMessage(e) {
  const msg = e.data

  switch (msg.status) {
    case 'device_detected':
      _onProgress?.({ type: 'device', device: msg.device })
      break

    case 'phase':
      _onProgress?.({ type: 'phase', phase: msg.phase, note: msg.note })
      break

    case 'downloading':
      _onProgress?.({ type: 'downloading', file: msg.file, progress: msg.progress })
      break

    case 'ready':
      _status  = 'ready'
      _modelId = msg.modelId
      _onProgress?.({ type: 'ready', modelId: msg.modelId })
      if (_loadResolve) { _loadResolve(msg.modelId); _loadResolve = _loadReject = null }
      break

    case 'success':
      if (_genResolve) {
        _genResolve({ text: msg.generatedText, latencyMs: Math.round(msg.elapsedMs), tokensPerSec: msg.tokensPerSec })
        _genResolve = _genReject = null
      }
      break

    case 'error':
      _status = _status === 'loading' ? 'error' : _status
      const err = new Error(msg.error)
      _onProgress?.({ type: 'error', error: msg.error })
      if (_loadReject)  { _status = 'error'; _loadReject(err);  _loadResolve = _loadReject = null }
      if (_genReject)   { _genReject(err);   _genResolve  = _genReject  = null }
      break

    case 'cancelled':
    case 'disposed':
      _status  = 'idle'
      _modelId = null
      break
  }
}

// ── Public API ────────────────────────────────────────────────────

export function getModelStatus() {
  return { status: _status, modelId: _modelId }
}

/**
 * Load a WebLLM model into the worker.
 * @param {string} modelId — from WEBLLM_MODELS
 * @param {Function} onProgress — called with {type, ...} progress events
 * @returns {Promise<string>} resolves with modelId when ready
 */
export function loadModel(modelId, onProgress) {
  _ensureWorker()
  _status     = 'loading'
  _onProgress = onProgress ?? null
  _genCounter++

  return new Promise((resolve, reject) => {
    _loadResolve = resolve
    _loadReject  = reject
    _worker.postMessage({ action: 'load', modelId, gen: _genCounter })
  })
}

/**
 * callLLM — Send a RAG prompt to the loaded WebLLM model.
 * Signature matches the original cloud-provider callLLM so ragEngine.js needs no changes.
 *
 * @param {Array<{role: string, content: string}>} messages
 * @param {string} systemPrompt
 * @param {Array} _retrievedChunks — unused (kept for API compat)
 * @returns {Promise<{text: string, rawResponse: object, latencyMs: number}>}
 */
export async function callLLM(messages, systemPrompt, _retrievedChunks = []) {
  if (_status !== 'ready' || !_worker) {
    throw new Error('No model loaded. Load a WebLLM model first using the model selector.')
  }

  _genCounter++
  return new Promise((resolve, reject) => {
    _genResolve = ({ text, latencyMs, tokensPerSec }) => {
      resolve({ text, rawResponse: { webllm: true, tokensPerSec }, latencyMs })
    }
    _genReject = reject
    _worker.postMessage({ action: 'generate', messages, systemPrompt, gen: _genCounter })
  })
}

/**
 * setLLMConfig — kept for API compat with ragEngine.js (no-op in WebLLM mode).
 */
export function setLLMConfig(_cfg) {}

/**
 * getLLMConfig — returns a minimal config object for compat with ragEngine.js.
 */
export function getLLMConfig() {
  return { provider: 'webllm', model: _modelId ?? 'none', apiKey: '', proxyUrl: '' }
}

export function cancelLoad() {
  _worker?.postMessage({ action: 'cancel' })
}
