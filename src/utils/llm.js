// llm.js — Multi-provider LLM API client with proxy routing
// Supported: OpenAI, Anthropic, Google Gemini, NVIDIA NIM, and Mock AI

const DEFAULT_PROXY = 'https://quantumstudio.visrow.workers.dev/'

export const PROVIDERS = [
  {
    id: 'openai',
    name: 'OpenAI',
    icon: '🤖',
    keyPlaceholder: 'sk-…',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    models: [
      { id: 'gpt-4o',      name: 'GPT-4o (recommended)' },
      { id: 'gpt-4o-mini', name: 'GPT-4o Mini (fast)' },
      { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
    ],
    format: 'openai',
  },
  {
    id: 'gemini',
    name: 'Google Gemini',
    icon: '✨',
    keyPlaceholder: 'AIza…',
    endpoint: 'https://generativelanguage.googleapis.com',
    models: [
      { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash (recommended)' },
      { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash' },
      { id: 'gemini-1.5-pro',   name: 'Gemini 1.5 Pro' },
    ],
    format: 'openai', // handles Gemini API key via URL query parameter
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    icon: '🧬',
    keyPlaceholder: 'sk-ant-…',
    endpoint: 'https://api.anthropic.com/v1/messages',
    models: [
      { id: 'claude-3-5-sonnet-20241022',   name: 'Claude 3.5 Sonnet' },
      { id: 'claude-3-5-haiku-20241022',    name: 'Claude 3.5 Haiku' },
    ],
    format: 'anthropic',
  },
  {
    id: 'nvidia',
    name: 'NVIDIA NIM',
    icon: '🟢',
    keyPlaceholder: 'nvapi-…',
    endpoint: 'https://integrate.api.nvidia.com/v1/chat/completions',
    models: [
      { id: 'meta/llama-3.1-70b-instruct',            name: 'Llama 3.1 70B Instruct' },
      { id: 'nvidia/llama-3.1-nemotron-70b-instruct', name: 'Llama 3.1 Nemotron 70B' },
      { id: 'nvidia/nemotron-nano-12b-v2-vl',         name: 'Nano 12B V2 (lightweight)' },
    ],
    format: 'openai',
  },
  {
    id: 'mock',
    name: 'Mock AI',
    icon: '🧪',
    keyPlaceholder: 'No key needed',
    endpoint: 'mock',
    models: [
      { id: 'mock-rag-agent', name: 'Mock RAG Responder (Local)' },
    ],
    format: 'mock',
  },
]

let _config = {
  provider: 'mock',
  apiKey: '',
  model: 'mock-rag-agent',
  proxyUrl: _loadProxyUrl(),
}

function _loadProxyUrl() {
  try { return localStorage.getItem('rag_proxy_url') || DEFAULT_PROXY } catch { return DEFAULT_PROXY }
}

export function setLLMConfig(cfg) {
  _config = { ..._config, ...cfg }
  if (cfg.proxyUrl !== undefined) {
    try { localStorage.setItem('rag_proxy_url', cfg.proxyUrl || DEFAULT_PROXY) } catch { /* ignore */ }
  }
}

export function getLLMConfig() {
  return { ..._config, proxyUrl: _config.proxyUrl || DEFAULT_PROXY }
}

export function getProviderDef(providerId) {
  return PROVIDERS.find(p => p.id === (providerId || _config.provider))
}

/**
 * Call the selected LLM provider.
 * @param {Array<{role: string, content: string}>} messages
 * @param {string} systemPrompt
 * @param {Array<object>} retrievedChunks - Pass chunks here to support Mock AI grounding
 * @returns {Promise<{text: string, rawResponse: object, latencyMs: number}>}
 */
export async function callLLM(messages, systemPrompt, retrievedChunks = []) {
  const { provider, apiKey, model, proxyUrl } = getLLMConfig()
  const providerDef = getProviderDef(provider)
  if (!providerDef) throw new Error(`Unknown provider: ${provider}`)
  
  if (provider !== 'mock' && !apiKey) {
    throw new Error('API Key is missing. Please enter your API key in the configuration bar.')
  }

  const start = Date.now()
  const proxy = proxyUrl || DEFAULT_PROXY

  // Handle Mock mode locally
  if (provider === 'mock') {
    await new Promise(r => setTimeout(r, 800)) // simulate network delay
    const userQuery = messages[messages.length - 1]?.content || ''
    const generatedText = generateMockRagResponse(userQuery, retrievedChunks)
    return {
      text: generatedText,
      rawResponse: { mock: true, sourcesUsed: retrievedChunks.length },
      latencyMs: Date.now() - start
    }
  }

  if (providerDef.format === 'anthropic') {
    const data = await _callAnthropic(messages, systemPrompt, model, apiKey, proxy)
    return {
      text: data.content?.[0]?.text || '',
      rawResponse: data,
      latencyMs: Date.now() - start
    }
  }

  // OpenAI format (including NVIDIA and Gemini compatibility)
  const data = await _callOpenAIFormat(messages, systemPrompt, model, apiKey, proxy, providerDef)
  
  let text = ''
  if (provider === 'gemini') {
    text = data.candidates?.[0]?.content?.parts?.[0]?.text || ''
  } else {
    text = data.choices?.[0]?.message?.content || ''
  }

  return {
    text,
    rawResponse: data,
    latencyMs: Date.now() - start
  }
}

// ── Call Implementations ──────────────────────────────────────────

async function _callOpenAIFormat(messages, systemPrompt, model, apiKey, proxy, providerDef) {
  const targetUrl = providerDef.id === 'gemini'
    ? `${providerDef.endpoint}/v1beta/models/${model}:generateContent?key=${apiKey}`
    : providerDef.endpoint

  const headers = {
    'Content-Type': 'application/json',
    'x-target-url': targetUrl,
  }
  if (providerDef.id !== 'gemini') {
    headers['Authorization'] = `Bearer ${apiKey}`
  }

  let body
  if (providerDef.id === 'gemini') {
    // Convert messages to Gemini API format
    const contents = messages.map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }]
    }))
    // Prepend system prompt
    body = {
      contents,
      systemInstruction: { parts: [{ text: systemPrompt }] },
      generationConfig: { temperature: 0.1 }
    }
  } else {
    body = {
      model,
      temperature: 0.1,
      messages: [{ role: 'system', content: systemPrompt }, ...messages],
    }
  }

  const res = await fetch(proxy, { method: 'POST', headers, body: JSON.stringify(body) })
  const data = await res.json()
  if (!res.ok) throw new Error(data?.error?.message || `${providerDef.name} error ${res.status}`)
  return data
}

async function _callAnthropic(messages, systemPrompt, model, apiKey, proxy) {
  const headers = {
    'Content-Type': 'application/json',
    'x-target-url': 'https://api.anthropic.com/v1/messages',
    'x-api-key': apiKey,
    'anthropic-version': '2023-06-01',
  }
  // Convert 'system' message out of messages array for Anthropic
  const anthropicMessages = messages.filter(m => m.role !== 'system')

  const body = { 
    model, 
    system: systemPrompt, 
    messages: anthropicMessages, 
    max_tokens: 2048, 
    temperature: 0.1 
  }

  const res = await fetch(proxy, { method: 'POST', headers, body: JSON.stringify(body) })
  const data = await res.json()
  if (!res.ok) throw new Error(data?.error?.message || `Anthropic error ${res.status}`)
  return data
}

// ── Mock RAG Synthesizer ──────────────────────────────────────────

function generateMockRagResponse(query, chunks) {
  if (chunks.length === 0) {
    return `[Mock AI Response - No Context Found]
I'm running locally in your browser. I searched the index but couldn't find any relevant text chunks matching your query: "${query}".

Please upload a relevant PDF or text file first, or adjust your chunk size/retrieval parameters.`
  }

  // Synthesize an answer directly grounding in the top chunk texts
  const topChunk = chunks[0]
  const topText = topChunk.text
  const docName = topChunk.docName || 'Uploaded File'
  const pageNum = topChunk.pageNumber ? `, Page ${topChunk.pageNumber}` : ''
  
  // Extract a few sentences to make the answer look natural
  const sentences = topText.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 5)
  const keyFact = sentences[0] || topText.substring(0, 100)
  const secondaryFact = sentences[1] || 'This document contains essential concepts regarding your query.'

  return `[Mock AI Response - Local Browser-Native RAG]

Based on the retrieved context from **${docName}**${pageNum} (which matched your query with a confidence score of **${(topChunk.confidence * 100).toFixed(1)}%**):

"${keyFact}." [1]

Furthermore, the document indicates that "${secondaryFact}." [1]

*Note: Since you are using **Mock AI**, this response is synthesized locally using the actual text from your top matching document chunk to demonstrate the RAG pipeline. To generate high-quality, fully reasoned responses, add an API Key and select OpenAI, Gemini, or Anthropic.*`
}

/**
 * testConnection — Verify the API key and proxy by sending a fast verification call.
 */
export async function testConnection() {
  const { provider, apiKey, model, proxyUrl } = getLLMConfig()
  const providerDef = getProviderDef(provider)
  if (!providerDef) throw new Error(`Unknown provider: ${provider}`)

  if (provider === 'mock') {
    await new Promise(r => setTimeout(r, 400))
    return { text: 'OK (mock)', latencyMs: 400 }
  }

  if (!apiKey) throw new Error('No API key configured.')

  const proxy = proxyUrl || DEFAULT_PROXY
  const start = Date.now()

  if (providerDef.format === 'anthropic') {
    const res = await fetch(proxy, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-target-url': providerDef.endpoint,
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model,
        max_tokens: 10,
        messages: [{ role: 'user', content: 'Say OK' }],
      }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data?.error?.message || `Anthropic error ${res.status}`)
    return { text: data.content?.[0]?.text || 'OK', latencyMs: Date.now() - start }
  }

  // OpenAI format (including Gemini/Nvidia)
  let targetUrl = providerDef.endpoint
  if (provider === 'gemini') {
    targetUrl = `${providerDef.endpoint}/v1beta/models/${model}:generateContent?key=${apiKey}`
  }

  const headers = {
    'Content-Type': 'application/json',
    'x-target-url': targetUrl,
  }
  if (provider !== 'gemini') headers['Authorization'] = `Bearer ${apiKey}`

  let body
  if (provider === 'gemini') {
    body = { contents: [{ parts: [{ text: 'Say OK' }] }], generationConfig: { maxOutputTokens: 10 } }
  } else {
    body = { model, max_tokens: 10, messages: [{ role: 'user', content: 'Say OK' }] }
  }

  const res = await fetch(proxy, { method: 'POST', headers, body: JSON.stringify(body) })
  const data = await res.json()
  if (!res.ok) throw new Error(data?.error?.message || `${providerDef.name} error ${res.status}`)

  let text = ''
  if (provider === 'gemini') {
    text = data.candidates?.[0]?.content?.parts?.[0]?.text || 'OK'
  } else {
    text = data.choices?.[0]?.message?.content || 'OK'
  }
  return { text: text.trim(), latencyMs: Date.now() - start }
}
