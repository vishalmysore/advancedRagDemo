// main.js — Main UI logic and RAG orchestrator binder
import { db, deleteDocument, getDocumentStats, clearAllData } from './utils/db.js'
import { parsePdf, parseTxt } from './utils/pdfParser.js'
import { loadEmbeddingModel, loadRerankerModel } from './utils/embeddings.js'
import { ingestDocument, executeRAGQuery } from './utils/ragEngine.js'
import { PROVIDERS, setLLMConfig, getLLMConfig, testConnection } from './utils/llm.js'
import { precisionAtK, recallAtK, mrr, ndcgAtK, averagePrecision, precisionRecallCurve, renderPRCurve } from './utils/irMetrics.js'


// ── State management ──────────────────────────────────────────────
let isProcessingFile = false
let isExecutingQuery = false
let lastRetrievedSources = []       // holds sources from last query for P&R labeling
let relevanceLabels = []            // parallel array: true=relevant, false=not relevant, null=unlabeled

// ── DOM Elements ──────────────────────────────────────────────────
// Config Banner
const providerSelect = document.getElementById('providerSelect')
const modelSelect = document.getElementById('modelSelect')
const apiKeyInput = document.getElementById('apiKeyInput')
const proxyInput = document.getElementById('proxyInput')
const resetProxyBtn = document.getElementById('resetProxyBtn')
const testConnectionBtn = document.getElementById('testConnectionBtn')
const testResult = document.getElementById('testResult')
const dbStatsBadge = document.getElementById('dbStatsBadge')
const clearDbBtn = document.getElementById('clearDbBtn')

// File Ingestion
const dropZone = document.getElementById('dropZone')
const fileInput = document.getElementById('fileInput')
const docTableBody = document.getElementById('docTableBody')

// Parameters
const sliderChunkSize = document.getElementById('sliderChunkSize')
const valChunkSize = document.getElementById('valChunkSize')
const sliderChunkOverlap = document.getElementById('sliderChunkOverlap')
const valChunkOverlap = document.getElementById('valChunkOverlap')
const sliderWeights = document.getElementById('sliderWeights')
const valWeights = document.getElementById('valWeights')
const rerankSelect = document.getElementById('rerankSelect')
const sliderTopK = document.getElementById('sliderTopK')
const valTopK = document.getElementById('valTopK')
const checkUseCache = document.getElementById('checkUseCache')

// Playground & Trace
const queryInput = document.getElementById('queryInput')
const benchmarkToggle = document.getElementById('benchmarkToggle')
const submitQueryBtn = document.getElementById('submitQueryBtn')
const traceContainer = document.getElementById('traceContainer')

// Wasm Progress Banner
const modelProgressBanner = document.getElementById('modelProgressBanner')
const progressTitle = document.getElementById('progressTitle')
const progressPercent = document.getElementById('progressPercent')
const progressBarFill = document.getElementById('progressBarFill')
const progressFileName = document.getElementById('progressFileName')

// Standard RAG Output Panels
const standardResponseDesk = document.getElementById('standardResponseDesk')
const waitingResponse = document.getElementById('waitingResponse')
const responseContainer = document.getElementById('responseContainer')
const hallucinationBanner = document.getElementById('hallucinationBanner')
const hallucinationReason = document.getElementById('hallucinationReason')
const ragAnswer = document.getElementById('ragAnswer')
const evalContextRelevance = document.getElementById('evalContextRelevance')
const evalFaithfulness = document.getElementById('evalFaithfulness')
const evalAnswerRelevance = document.getElementById('evalAnswerRelevance')
const evalLatency = document.getElementById('evalLatency')
const sourcesContainer = document.getElementById('sourcesContainer')

// Precision & Recall Tab
const tabResponse        = document.getElementById('tabResponse')
const tabPrecisionRecall = document.getElementById('tabPrecisionRecall')
const prPanel            = document.getElementById('prPanel')
const prChunkList        = document.getElementById('prChunkList')
const prCurveContainer   = document.getElementById('prCurveContainer')
const prP1               = document.getElementById('prP1')
const prP3               = document.getElementById('prP3')
const prPK               = document.getElementById('prPK')
const prRK               = document.getElementById('prRK')
const prMRR              = document.getElementById('prMRR')
const prNDCG             = document.getElementById('prNDCG')
const prAP               = document.getElementById('prAP')

// Benchmark Panel Output
const benchmarkResponseDesk = document.getElementById('benchmarkResponseDesk')
const benchVectorBody = document.getElementById('benchVectorBody')
const benchBm25Body = document.getElementById('benchBm25Body')
const benchHybridBody = document.getElementById('benchHybridBody')

// ── Bootstrapping ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initProvidersDropdown()
  loadLLMConfigFromStorage()
  bindParameterSliders()
  bindEventHandlers()
  bindTabHandlers()
  await refreshDocumentList()
  appendLog('system', 'Observability Log Initialised. Ready to process files.', 'info-step')
})

// ── Tab Switching ─────────────────────────────────────────────────
function bindTabHandlers() {
  tabResponse.addEventListener('click', () => switchRightTab('response'))
  tabPrecisionRecall.addEventListener('click', () => switchRightTab('pr'))
}

function switchRightTab(tab) {
  if (tab === 'response') {
    tabResponse.classList.add('active')
    tabPrecisionRecall.classList.remove('active')
    prPanel.classList.add('hidden')
    // Show whichever response desk is currently active
    standardResponseDesk.style.display = ''
    benchmarkResponseDesk.style.display = ''
  } else {
    tabPrecisionRecall.classList.add('active')
    tabResponse.classList.remove('active')
    prPanel.classList.remove('hidden')
    standardResponseDesk.style.display = 'none'
    benchmarkResponseDesk.style.display = 'none'
    // Render current sources if available
    renderPRChunks()
  }
}

// ── Precision & Recall Panel ──────────────────────────────────────

function renderPRChunks() {
  if (lastRetrievedSources.length === 0) {
    prChunkList.innerHTML = '<div class="pr-empty">Run a query first, then label the retrieved chunks here.</div>'
    updatePRMetrics()
    return
  }

  prChunkList.innerHTML = ''
  lastRetrievedSources.forEach((src, idx) => {
    const label = relevanceLabels[idx]   // true | false | null

    const card = document.createElement('div')
    card.className = 'pr-chunk-card' +
      (label === true ? ' labeled-relevant' : label === false ? ' labeled-irrelevant' : '')
    card.id = `pr-card-${idx}`

    card.innerHTML = `
      <div class="pr-chunk-rank">#${idx + 1}</div>
      <div class="pr-chunk-body">
        <div class="pr-chunk-doc">${src.docName} · Page ${src.pageNumber}</div>
        <div class="pr-chunk-text">${src.text}</div>
        <div class="pr-chunk-scores">
          Vec: ${src.vectorScore?.toFixed(3) || '0.000'} &nbsp;|&nbsp;
          BM25: ${src.bm25Score?.toFixed(2) || '0.00'} &nbsp;|&nbsp;
          Conf: ${Math.round((src.confidence || 0) * 100)}%
        </div>
      </div>
      <div class="pr-chunk-actions">
        <button class="pr-btn ${label === true ? 'rel-active' : ''}" data-idx="${idx}" data-val="true">✓ Relevant</button>
        <button class="pr-btn ${label === false ? 'irrel-active' : ''}" data-idx="${idx}" data-val="false">✗ Not Relevant</button>
      </div>
    `

    // Bind label buttons
    card.querySelectorAll('.pr-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const i = parseInt(btn.dataset.idx)
        const val = btn.dataset.val === 'true'
        // Toggle off if already labeled same way
        relevanceLabels[i] = relevanceLabels[i] === val ? null : val
        renderPRChunks()
        updatePRMetrics()
      })
    })

    prChunkList.appendChild(card)
  })

  updatePRMetrics()
}

function updatePRMetrics() {
  const K = lastRetrievedSources.length
  // Build binary relevance array — unlabeled treated as not relevant for metric calc
  const rel = relevanceLabels.slice(0, K).map(v => v === true)
  const labeled = relevanceLabels.slice(0, K).filter(v => v !== null).length

  if (labeled === 0) {
    // No labels yet — show dashes
    ;[prP1, prP3, prPK, prRK, prMRR, prNDCG, prAP].forEach(el => { el.textContent = '—' })
    renderPRCurve(prCurveContainer, [], 0)
    return
  }

  const fmt = v => `${(v * 100).toFixed(1)}%`

  prP1.textContent   = fmt(precisionAtK(rel, 1))
  prP3.textContent   = fmt(precisionAtK(rel, Math.min(3, K)))
  prPK.textContent   = fmt(precisionAtK(rel, K))
  prRK.textContent   = fmt(recallAtK(rel, K))
  prMRR.textContent  = fmt(mrr(rel))
  prNDCG.textContent = fmt(ndcgAtK(rel, K))

  const ap = averagePrecision(rel)
  prAP.textContent   = fmt(ap)

  const curve = precisionRecallCurve(rel)
  renderPRCurve(prCurveContainer, curve, ap)
}

// ── Observability Trace Logger ────────────────────────────────────
function appendLog(stepName, message, typeClass = 'info-step') {
  const now = new Date()
  const timeStr = `${now.toTimeString().split(' ')[0]}.${String(now.getMilliseconds()).padStart(3, '0')}`

  // If container currently shows placeholder, clear it
  const emptyPlaceholder = traceContainer.querySelector('.trace-empty')
  if (emptyPlaceholder) {
    traceContainer.innerHTML = ''
  }

  const row = document.createElement('div')
  row.className = `trace-entry ${typeClass}`
  
  const timeCol = document.createElement('div')
  timeCol.className = 'trace-time'
  timeCol.textContent = timeStr
  
  const labelCol = document.createElement('div')
  labelCol.className = 'trace-label'
  labelCol.textContent = stepName.toUpperCase()

  const msgCol = document.createElement('div')
  msgCol.className = 'trace-msg'
  msgCol.textContent = message

  row.appendChild(timeCol)
  row.appendChild(labelCol)
  row.appendChild(msgCol)
  
  traceContainer.appendChild(row)
  traceContainer.scrollTop = traceContainer.scrollHeight
}

// ── Wasm Model Loading Progress Binder ──────────────────────────
function handleModelDownloadProgress(event) {
  if (event.status === 'initiate') {
    modelProgressBanner.classList.remove('hidden')
    progressTitle.textContent = `Downloading ${event.model ? 'Embedding Model' : 'Model Files'}...`
    progressFileName.textContent = `File: ${event.file}`
    progressPercent.textContent = '0%'
    progressBarFill.style.width = '0%'
  } else if (event.status === 'progress') {
    modelProgressBanner.classList.remove('hidden')
    const pct = Math.round(event.progress || 0)
    progressPercent.textContent = `${pct}%`
    progressBarFill.style.width = `${pct}%`
    progressFileName.textContent = `Downloading: ${event.file}`
  } else if (event.status === 'done' || event.status === 'ready') {
    // delay hiding slightly to let user see 100% completion
    setTimeout(() => {
      modelProgressBanner.classList.add('hidden')
    }, 1000)
  }
}

// ── LLM Provider Configuration Dropdowns ────────────────────────
function initProvidersDropdown() {
  providerSelect.innerHTML = ''
  PROVIDERS.forEach(p => {
    const opt = document.createElement('option')
    opt.value = p.id
    opt.textContent = `${p.icon} ${p.name}`
    providerSelect.appendChild(opt)
  })

  providerSelect.addEventListener('change', () => {
    updateModelsDropdown()
    saveLLMConfigToStorage()
  })

  modelSelect.addEventListener('change', saveLLMConfigToStorage)
  apiKeyInput.addEventListener('input', saveLLMConfigToStorage)
  proxyInput.addEventListener('input', saveLLMConfigToStorage)

  updateModelsDropdown()
}

function updateModelsDropdown() {
  const pId = providerSelect.value
  const provider = PROVIDERS.find(p => p.id === pId)
  modelSelect.innerHTML = ''

  if (provider) {
    provider.models.forEach(m => {
      const opt = document.createElement('option')
      opt.value = m.id
      opt.textContent = m.name
      modelSelect.appendChild(opt)
    })
    apiKeyInput.placeholder = provider.keyPlaceholder
    
    // Toggle api-key fields based on Provider
    if (pId === 'mock') {
      apiKeyInput.disabled = true
      apiKeyInput.value = ''
    } else {
      apiKeyInput.disabled = false
    }
  }
}

function loadLLMConfigFromStorage() {
  const stored = localStorage.getItem('rag_llm_config')
  if (stored) {
    try {
      const parsed = JSON.parse(stored)
      providerSelect.value = parsed.provider || 'mock'
      updateModelsDropdown()
      modelSelect.value = parsed.model || 'mock-rag-agent'
      // apiKey is NOT restored from storage (session only — matches harnessEngineeringDemo)
      apiKeyInput.value = ''
      proxyInput.value = parsed.proxyUrl || getLLMConfig().proxyUrl
    } catch (e) {
      console.error('Error parsing stored LLM config', e)
    }
  } else {
    proxyInput.value = getLLMConfig().proxyUrl
  }
  applyConfigChange()
}

function saveLLMConfigToStorage() {
  // Do NOT save apiKey to localStorage (security — session only, matches harnessEngineeringDemo pattern).
  // Only persist non-sensitive preferences: provider, model, proxyUrl.
  const cfg = {
    provider: providerSelect.value,
    model: modelSelect.value,
    proxyUrl: proxyInput.value
  }
  localStorage.setItem('rag_llm_config', JSON.stringify(cfg))
  applyConfigChange()
}

function applyConfigChange() {
  setLLMConfig({
    provider: providerSelect.value,
    model: modelSelect.value,
    apiKey: apiKeyInput.value,
    proxyUrl: proxyInput.value
  })
}

// ── Parameters and Sliders ──────────────────────────────────────
function bindParameterSliders() {
  // Chunk size
  sliderChunkSize.addEventListener('input', () => {
    valChunkSize.textContent = sliderChunkSize.value
  })
  
  // Chunk overlap
  sliderChunkOverlap.addEventListener('input', () => {
    valChunkOverlap.textContent = sliderChunkOverlap.value
  })

  // Hybrid Weights
  sliderWeights.addEventListener('input', () => {
    const vecPct = sliderWeights.value
    const kwPct = 100 - vecPct
    valWeights.textContent = `${vecPct}% / ${kwPct}%`
  })

  // Top-K
  sliderTopK.addEventListener('input', () => {
    valTopK.textContent = sliderTopK.value
  })
}

// ── Ingest Document File Handling ───────────────────────────────
async function handleFileSelected(file) {
  if (isProcessingFile) return
  isProcessingFile = true
  
  appendLog('ingest', `Ingesting file: ${file.name} (${(file.size/1024).toFixed(1)} KB)...`, 'info-step')
  
  try {
    // 1. Warm up embedding model first (and show download progress if needed)
    appendLog('model', 'Initialising local embedding pipeline (all-MiniLM-L6-v2)...', 'info-step')
    await loadEmbeddingModel(handleModelDownloadProgress)
    
    // 2. Parse text page-by-page
    let pages = []
    if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
      appendLog('parse', 'Extracting pages from PDF file locally...', 'info-step')
      pages = await parsePdf(file, (curr, total) => {
        appendLog('parse', `Reading PDF page ${curr}/${total}...`, 'info-step')
      })
    } else {
      appendLog('parse', 'Reading text file locally...', 'info-step')
      const text = await parseTxt(file)
      pages = [{ pageNumber: 1, text }]
    }

    // 3. Process chunking & generate embeddings
    const config = {
      chunkSize: parseInt(sliderChunkSize.value),
      chunkOverlap: parseInt(sliderChunkOverlap.value)
    }

    appendLog('rag', `Chunking document and computing float embeddings locally...`, 'info-step')
    const docId = await ingestDocument(file.name, file.type || 'text/plain', pages, config, (curr, total) => {
      appendLog('embed', `Computing embedding vectors: chunk ${curr}/${total}...`, 'vector-step')
    })

    appendLog('success', `Document "${file.name}" ingested successfully! ID: ${docId}`, 'eval-step')
    
  } catch (err) {
    console.error(err)
    appendLog('error', `Failed to ingest file: ${err.message}`, 'error-step')
    alert(`File Ingestion Failed: ${err.message}`)
  } finally {
    isProcessingFile = false
    modelProgressBanner.classList.add('hidden')
    await refreshDocumentList()
  }
}

// ── Refresh Document List from DB ────────────────────────────────
async function refreshDocumentList() {
  const stats = await getDocumentStats()
  
  // Update cache badge
  dbStatsBadge.textContent = `${stats.docCount} docs / ${stats.chunkCount} chunks`

  docTableBody.innerHTML = ''
  if (stats.docs.length === 0) {
    docTableBody.innerHTML = '<tr class="empty-row"><td colspan="3">No documents ingested.</td></tr>'
    return
  }

  stats.docs.forEach(doc => {
    const tr = document.createElement('tr')
    
    const nameTd = document.createElement('td')
    nameTd.textContent = doc.name
    nameTd.title = doc.name

    const chunksTd = document.createElement('td')
    chunksTd.textContent = doc.chunkCount

    const actionTd = document.createElement('td')
    const btn = document.createElement('button')
    btn.className = 'btn-delete-doc'
    btn.innerHTML = '🗑'
    btn.title = 'Delete document'
    btn.addEventListener('click', async () => {
      if (confirm(`Are you sure you want to delete ${doc.name}?`)) {
        appendLog('delete', `Deleting document: ${doc.name}...`, 'info-step')
        await deleteDocument(doc.id)
        appendLog('delete', `Deleted. Cache cleared.`, 'info-step')
        await refreshDocumentList()
      }
    })
    actionTd.appendChild(btn)

    tr.appendChild(nameTd)
    tr.appendChild(chunksTd)
    tr.appendChild(actionTd)
    docTableBody.appendChild(tr)
  })
}

// ── Bind UI Event Listeners ──────────────────────────────────────
function bindEventHandlers() {
  // Test connection button
  testConnectionBtn.addEventListener('click', async () => {
    // Always sync config from UI first (matches harnessEngineeringDemo pattern)
    applyConfigChange()
    const cfg = getLLMConfig()
    if (cfg.provider !== 'mock' && !cfg.apiKey) {
      testResult.classList.remove('hidden')
      testResult.className = 'test-result test-result-fail'
      testResult.textContent = 'Connection failed: No API key entered. Please enter your API key above.'
      return
    }

    testConnectionBtn.disabled = true
    testResult.classList.remove('hidden')
    testResult.className = 'test-result test-result-ok'
    testResult.textContent = 'Verifying connection...'

    try {
      const res = await testConnection()
      testResult.textContent = `Success! Response: "${res.text}" (Latency: ${res.latencyMs}ms)`
    } catch (e) {
      testResult.className = 'test-result test-result-fail'
      testResult.textContent = `Connection failed: ${e.message}`
    } finally {
      testConnectionBtn.disabled = false
    }
  })

  // Reset proxy button
  resetProxyBtn.addEventListener('click', () => {
    proxyInput.value = 'https://quantumstudio.visrow.workers.dev/'
    saveLLMConfigToStorage()
  })

  // Clear DB
  clearDbBtn.addEventListener('click', async () => {
    if (confirm('Are you sure you want to delete all documents, chunks, and cached runs? This cannot be undone.')) {
      appendLog('system', 'Wiping IndexedDB database...', 'alert-step')
      await clearAllData()
      appendLog('system', 'Database cleared successfully.', 'eval-step')
      await refreshDocumentList()
    }
  })

  // Drag and Drop
  dropZone.addEventListener('click', () => fileInput.click())
  
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0])
    }
  })

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault()
    dropZone.classList.add('dragover')
  })

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover')
  })

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault()
    dropZone.classList.remove('dragover')
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0])
    }
  })

  // Submit Query button
  submitQueryBtn.addEventListener('click', handleQuerySubmission)
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleQuerySubmission()
    }
  })
}

// ── Handle Query Submissions ──────────────────────────────────────
async function handleQuerySubmission() {
  const query = queryInput.value.trim()
  if (!query) return
  if (isExecutingQuery) return

  // Always sync config from UI before executing (matches harnessEngineeringDemo pattern)
  applyConfigChange()
  const cfg = getLLMConfig()
  if (cfg.provider !== 'mock' && !cfg.apiKey) {
    alert('Please enter your API key in the configuration bar, or switch to Mock AI to test without one.')
    return
  }

  // Verify we have documents indexed
  const stats = await getDocumentStats()
  if (stats.chunkCount === 0) {
    alert('Please upload/ingest at least one PDF or text file before querying.')
    return
  }

  isExecutingQuery = true
  submitQueryBtn.disabled = true

  const isBenchmark = benchmarkToggle.checked
  const rerankType = rerankSelect.value

  const commonConfig = {
    topK: parseInt(sliderTopK.value),
    rerankType,
    useCache: checkUseCache.checked,
    llmProvider: providerSelect.value,
    llmModel: modelSelect.value,
    apiKey: apiKeyInput.value,
    proxyUrl: proxyInput.value
  }

  // Pre-load Cross Encoder model if selected to prevent timing logs from overlapping downloads
  if (rerankType === 'neural') {
    appendLog('model', 'Pre-loading local Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...', 'info-step')
    await loadRerankerModel(handleModelDownloadProgress)
  }

  if (isBenchmark) {
    // ── Benchmark side-by-side mode ──
    standardResponseDesk.classList.add('hidden')
    benchmarkResponseDesk.classList.remove('hidden')
    
    // Reset columns
    benchVectorBody.innerHTML = '<div class="bench-waiting">Generating Vector Only response...</div>'
    benchBm25Body.innerHTML = '<div class="bench-waiting">Generating BM25 Only response...</div>'
    benchHybridBody.innerHTML = '<div class="bench-waiting">Generating Hybrid response...</div>'

    appendLog('benchmark', 'Starting side-by-side parallel retrievals (Vector vs Keyword vs Hybrid)...', 'fusion-step')

    try {
      // 1. Vector Only query run
      const runVector = async () => {
        const conf = { ...commonConfig, vectorWeight: 1.0, bm25Weight: 0.0, useCache: false }
        const res = await executeRAGQuery(query, conf)
        renderBenchmarkResult('vector', res, benchVectorBody)
      }

      // 2. Keyword Only query run
      const runKeyword = async () => {
        const conf = { ...commonConfig, vectorWeight: 0.0, bm25Weight: 1.0, useCache: false }
        const res = await executeRAGQuery(query, conf)
        renderBenchmarkResult('bm25', res, benchBm25Body)
      }

      // 3. Fused Hybrid query run
      const runHybrid = async () => {
        const vecPct = parseInt(sliderWeights.value) / 100
        const kwPct = 1 - vecPct
        const conf = { ...commonConfig, vectorWeight: vecPct, bm25Weight: kwPct }
        const res = await executeRAGQuery(query, conf, (name, detail) => {
          appendLog(name, detail, 'fusion-step')
        })
        renderBenchmarkResult('hybrid', res, benchHybridBody)
      }

      // Run parallel
      await Promise.all([runVector(), runKeyword(), runHybrid()])
      appendLog('benchmark', 'All comparisons loaded.', 'eval-step')

    } catch (err) {
      appendLog('error', `Benchmark run failed: ${err.message}`, 'error-step')
    }

  } else {
    // ── Standard RAG run mode ──
    benchmarkResponseDesk.classList.add('hidden')
    standardResponseDesk.classList.remove('hidden')

    waitingResponse.classList.remove('hidden')
    responseContainer.classList.add('hidden')
    hallucinationBanner.classList.add('hidden')

    appendLog('query', `Received query: "${query}"`, 'info-step')

    const vecPct = parseInt(sliderWeights.value) / 100
    const kwPct = 1 - vecPct
    const config = {
      ...commonConfig,
      vectorWeight: vecPct,
      bm25Weight: kwPct
    }

    try {
      const result = await executeRAGQuery(query, config, (name, detail) => {
        let type = 'info-step'
        if (name.includes('Vector')) type = 'vector-step'
        if (name.includes('BM25')) type = 'bm25-step'
        if (name.includes('Fusion')) type = 'fusion-step'
        if (name.includes('Evaluation') || name.includes('OK')) type = 'eval-step'
        if (name.includes('Alert')) type = 'alert-step'
        appendLog(name, detail, type)
      })

      // 1. Render Generated Answer (with citations)
      renderAnswerText(result.answer)
      
      // 2. Render Evals
      evalContextRelevance.textContent = `${Math.round(result.metrics.contextRelevance * 100)}%`
      evalFaithfulness.textContent = `${Math.round(result.metrics.faithfulness * 100)}%`
      evalAnswerRelevance.textContent = `${Math.round(result.metrics.answerRelevance * 100)}%`
      evalLatency.textContent = `${result.metrics.latencyMs || result.trace.timings.total}ms`

      // Apply rating classes for Faithfulness
      evalFaithfulness.className = 'metric-val'
      if (result.metrics.faithfulness < 0.6) {
        evalFaithfulness.classList.add('low-score')
      } else if (result.metrics.faithfulness < 0.8) {
        evalFaithfulness.classList.add('mid-score')
      }

      // 3. Hallucination Alert display
      if (result.hallucinationWarning) {
        hallucinationBanner.classList.remove('hidden')
        hallucinationReason.textContent = `Warning: Assertions mismatch. Key items not in sources: ${result.mismatches.join('; ')}`
      } else {
        hallucinationBanner.classList.add('hidden')
      }

      // 4. Render retrieved sources
      renderSources(result.sources)

      // 5. Store sources for Precision & Recall tab
      lastRetrievedSources = result.sources || []
      relevanceLabels = new Array(lastRetrievedSources.length).fill(null)
      // If P&R tab is active, refresh it immediately
      if (!prPanel.classList.contains('hidden')) renderPRChunks()

      waitingResponse.classList.add('hidden')
      responseContainer.classList.remove('hidden')

    } catch (err) {
      console.error(err)
      appendLog('error', `RAG query failed: ${err.message}`, 'error-step')
      waitingResponse.innerHTML = `<div class="error-msg">⚠️ RAG Execution Failed<p style="font-size:12px; margin-top:6px">${err.message}</p></div>`
    }
  }

  isExecutingQuery = false
  submitQueryBtn.disabled = false
  await refreshDocumentList()
}

// ── Render Generated Answer Text & Interactive Citations ────────
function renderAnswerText(text) {
  // Instruct model to respond with [Source 1] citations.
  // We match [Source X] or [Source X, Page Y] or simply [X] (if LLM generates footnotes like [1])
  // Standardise match format
  let processedHTML = text
    .replace(/\[Source\s*(\d+)\]/gi, (match, g1) => {
      return `<span class="citation-badge" data-source-index="${g1}">Source ${g1}</span>`
    })
    .replace(/\[(\d+)\]/g, (match, g1) => {
      return `<span class="citation-badge" data-source-index="${g1}">${g1}</span>`
    })
    
  ragAnswer.innerHTML = processedHTML

  // Add click listeners to badges
  const badges = ragAnswer.querySelectorAll('.citation-badge')
  badges.forEach(b => {
    b.addEventListener('click', () => {
      const idx = parseInt(b.getAttribute('data-source-index')) - 1
      const sourceCard = document.getElementById(`source-card-${idx}`)
      
      if (sourceCard) {
        sourceCard.scrollIntoView({ behavior: 'smooth', block: 'center' })
        
        // Flash visual highlight
        sourceCard.classList.add('highlighted')
        setTimeout(() => {
          sourceCard.classList.remove('highlighted')
        }, 1800)
      } else {
        console.warn(`Source card matching index ${idx} not found in DOM.`)
      }
    })
  })
}

// ── Render Sources Cards list ─────────────────────────────────────
function renderSources(sources) {
  sourcesContainer.innerHTML = ''
  if (sources.length === 0) {
    sourcesContainer.innerHTML = '<div class="empty-sources">No context chunks matched the retrieval query.</div>'
    return
  }

  sources.forEach((source, idx) => {
    const card = document.createElement('div')
    card.className = 'source-card'
    card.id = `source-card-${idx}`

    const header = document.createElement('div')
    header.className = 'source-header'
    
    const title = document.createElement('div')
    title.className = 'source-title'
    title.textContent = `[${idx + 1}] ${source.docName}`

    const badge = document.createElement('div')
    badge.className = 'source-badge-id'
    badge.textContent = `Page ${source.pageNumber}`

    header.appendChild(title)
    header.appendChild(badge)

    const text = document.createElement('div')
    text.className = 'source-text'
    text.textContent = source.text

    const footer = document.createElement('div')
    footer.className = 'source-footer'

    // Scores
    const vecScore = document.createElement('div')
    vecScore.className = 'source-score-item'
    vecScore.innerHTML = `Vector Cos: <span>${source.vectorScore?.toFixed(3) || '0.000'}</span>`

    const bmScore = document.createElement('div')
    bmScore.className = 'source-score-item'
    bmScore.innerHTML = `BM25 Score: <span>${source.bm25Score?.toFixed(2) || '0.00'}</span>`

    const mergeScore = document.createElement('div')
    mergeScore.className = 'source-score-item'
    mergeScore.innerHTML = `Confidence: <span>${Math.round(source.confidence * 100)}%</span>`

    footer.appendChild(vecScore)
    footer.appendChild(bmScore)
    footer.appendChild(mergeScore)

    card.appendChild(header)
    card.appendChild(text)
    card.appendChild(footer)
    
    sourcesContainer.appendChild(card)
  })
}

// ── Render Benchmark Result Column ──────────────────────────────
function renderBenchmarkResult(type, result, targetElement) {
  targetElement.innerHTML = ''
  
  const card = document.createElement('div')
  card.className = 'bench-res-card'

  const answerTitle = document.createElement('label')
  answerTitle.className = 'field-label'
  answerTitle.textContent = 'Generated Response'

  const answer = document.createElement('p')
  answer.style.fontSize = '12.5px'
  answer.style.lineHeight = '1.5'
  answer.style.color = 'var(--text)'
  answer.textContent = result.answer

  card.appendChild(answerTitle)
  card.appendChild(answer)

  // Add Hallucination alert if present
  if (result.hallucinationWarning) {
    const alertDiv = document.createElement('div')
    alertDiv.className = 'hallucination-alert'
    alertDiv.style.padding = '6px 10px'
    alertDiv.style.marginTop = '10px'
    alertDiv.innerHTML = `<span style="font-size:14px">⚠️</span> <span style="font-size:10px; color:#ffb8b8">Possible Hallucination Warning!</span>`
    card.appendChild(alertDiv)
  }

  // Chunks Title
  const chunksTitle = document.createElement('label')
  chunksTitle.className = 'field-label'
  chunksTitle.style.marginTop = '12px'
  chunksTitle.textContent = 'Top Chunks Used'
  card.appendChild(chunksTitle)

  // Lists top chunks
  if (result.sources.length === 0) {
    const empty = document.createElement('div')
    empty.style.fontSize = '11px'
    empty.style.color = 'var(--text-muted)'
    empty.textContent = 'No matching chunks found.'
    card.appendChild(empty)
  } else {
    result.sources.slice(0, 3).forEach((src, idx) => {
      const srcDiv = document.createElement('div')
      srcDiv.className = 'bench-source-item'
      // Color coded border based on search type
      if (type === 'vector') srcDiv.style.borderLeftColor = 'var(--vector-color)'
      if (type === 'bm25') srcDiv.style.borderLeftColor = 'var(--bm25-color)'
      if (type === 'hybrid') srcDiv.style.borderLeftColor = 'var(--hybrid-color)'

      srcDiv.textContent = `[${idx+1}] ${src.docName} (Page ${src.pageNumber}) - Conf: ${Math.round(src.confidence*100)}%`
      card.appendChild(srcDiv)
    })
  }

  // Metrics footer
  const metrics = document.createElement('div')
  metrics.className = 'bench-metrics'

  const latencySpan = document.createElement('span')
  latencySpan.innerHTML = `Time: <strong>${result.metrics.latencyMs}ms</strong>`

  const faithSpan = document.createElement('span')
  faithSpan.innerHTML = `Faith: <strong>${Math.round(result.metrics.faithfulness * 100)}%</strong>`

  const relevSpan = document.createElement('span')
  relevSpan.innerHTML = `Relevance: <strong>${Math.round(result.metrics.contextRelevance * 100)}%</strong>`

  metrics.appendChild(latencySpan)
  metrics.appendChild(faithSpan)
  metrics.appendChild(relevSpan)
  
  card.appendChild(metrics)
  targetElement.appendChild(card)
}
