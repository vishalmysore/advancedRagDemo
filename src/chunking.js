// chunking.js — UI for the Chunking Lab page
import { runLab, STRATEGIES } from './utils/chunkLab.js'
import { SAMPLE_DOC, SAMPLE_QE } from './utils/labCorpus.js'

const $ = id => document.getElementById(id)
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]))
const pct = v => Math.round(v * 100) + '%'
const BUDGETS = [40, 60, 80, 100, 120, 150, 180, 220, 260, 300, 350, 400]

const state = {
  L: 450, budget: 120, split: 'test', mode: 'flat', headers: false, resolveRefs: true,
  doc: SAMPLE_DOC, qe: SAMPLE_QE, q: 0, graphKey: 'struct',
}
let lab = null

// URL params let the article link straight to a configuration (and make screenshots reproducible)
const params = new URLSearchParams(location.search)
for (const k of ['L', 'budget', 'q']) if (params.has(k)) state[k] = +params.get(k)
for (const k of ['split', 'mode', 'graphKey']) if (params.has(k)) state[k] = params.get(k)
if (params.has('headers')) state.headers = params.get('headers') === '1'

function run() {
  lab = runLab(state.doc, state.qe, { L: state.L, budget: state.budget, evalSplit: state.split, headers: state.headers, resolveRefs: state.resolveRefs })
  if (state.q >= lab.evSet.length) state.q = 0
  renderScoreboard(); renderCurve(); renderMap(); renderInspector(); renderGraph()
}

// ── Scoreboard ────────────────────────────────────────────────────────────────
function bar(v, color, best) {
  return `<div class="cell-bar"><div class="cell-fill" style="width:${Math.round(v * 100)}%;background:${color}"></div><span class="${best ? 'best' : ''}">${pct(v)}</span></div>`
}
function renderScoreboard() {
  const m = state.mode === 'flat' ? 'Flat' : 'Graph'
  const R = lab.results
  const bestC = Math.max(...STRATEGIES.map(s => R[s.key]['complete' + m]))
  const rows = STRATEGIES.map(s => {
    const r = R[s.key]
    return `<tr>
      <td><div class="strat"><span class="dot" style="background:${s.color}"></span><div><div class="sname">${s.name}</div><div class="stag">${s.tag}</div></div></div></td>
      <td class="num">${r.count}</td>
      <td class="num">${r.avgChars}</td>
      <td>${bar(r.cleanBoundaries, '#3b4b61')}</td>
      <td>${bar(r.evidenceIntact, '#3b4b61')}</td>
      <td>${bar(r.xrefResolved, '#3b4b61')}</td>
      <td>${bar(r['partial' + m], s.color + '88')}</td>
      <td>${bar(r['complete' + m], s.color, r['complete' + m] === bestC)}</td>
    </tr>`
  }).join('')
  $('scoreTable').innerHTML = `<thead><tr>
      <th>Strategy</th><th>Chunks</th><th>Avg chars</th>
      <th title="Share of chunk starts that land exactly on a provision heading">Clean boundaries</th>
      <th title="Share of evidence sentences that sit wholly inside at least one chunk">Evidence intact</th>
      <th title="Share of 'Section x.y' references whose target text is in the same chunk">X-refs resolved</th>
      <th title="Context contains at least one evidence sentence">Partial recall</th>
      <th title="Context contains every evidence sentence the question needs">Complete recall</th>
    </tr></thead><tbody>${rows}</tbody>`
  const parts = STRATEGIES.map(s => R[s.key]['partial' + m]), comps = STRATEGIES.map(s => R[s.key]['complete' + m])
  const split = { test: 'held-out', train: 'QES-train', all: 'all' }[state.split]
  $('sbNote').textContent = `${lab.evSet.length} ${split} questions · ${state.budget}-word context · ${state.mode === 'flat' ? 'flat BM25 top-k' : 'chunk graph (SpectRAG-lite)'}`
  $('callout').innerHTML = `Every strategy retrieves <b>some</b> of the evidence for ${pct(Math.min(...parts))}–${pct(Math.max(...parts))} of questions, but <b>all</b> of it for only ${pct(Math.min(...comps))}–${pct(Math.max(...comps))}. Partial recall flatters; the gap between the two columns is the multi-fact failure that "The Bottleneck Is Retrieval, Not Reading" describes.`
}

// ── Budget curve ──────────────────────────────────────────────────────────────
function renderCurve() {
  const m = state.mode === 'flat' ? 'Flat' : 'Graph'
  const series = STRATEGIES.map(s => ({ s, pts: [] }))
  for (const B of BUDGETS) {
    const r = runLab(state.doc, state.qe, { L: state.L, budget: B, evalSplit: state.split, headers: state.headers, resolveRefs: state.resolveRefs })
    series.forEach(x => x.pts.push([B, r.results[x.s.key]['complete' + m]]))
  }
  const W = 600, H = 300, pl = 44, pr = 150, pt = 14, pb = 36
  const X = b => pl + (b - BUDGETS[0]) / (BUDGETS[BUDGETS.length - 1] - BUDGETS[0]) * (W - pl - pr)
  const Y = v => pt + (1 - v) * (H - pt - pb)
  let g = ''
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line x1="${pl}" x2="${W - pr}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${pl - 8}" y="${Y(v) + 4}" class="axis" text-anchor="end">${v * 100}%</text>`
  for (const b of [40, 100, 160, 220, 300, 400]) g += `<text x="${X(b)}" y="${H - pb + 16}" class="axis" text-anchor="middle">${b}</text>`
  g += `<text x="${(pl + W - pr) / 2}" y="${H - 4}" class="axis" text-anchor="middle">context budget (words)</text>`
  g += `<line x1="${X(state.budget)}" x2="${X(state.budget)}" y1="${pt}" y2="${H - pb}" class="marker"/>`
  // end labels, nudged apart
  const ends = series.map(x => ({ x, y: Y(x.pts[x.pts.length - 1][1]) })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13
  for (const { s, pts } of series) {
    g += `<polyline fill="none" stroke="${s.color}" stroke-width="2.2" points="${pts.map(([b, v]) => `${X(b)},${Y(v)}`).join(' ')}"/>`
    g += pts.map(([b, v]) => `<circle cx="${X(b)}" cy="${Y(v)}" r="2.6" fill="${s.color}"><title>${s.name} · ${b} words · ${pct(v)}</title></circle>`).join('')
  }
  for (const { x, y } of ends) g += `<text x="${W - pr + 8}" y="${y + 4}" class="lbl" fill="${x.s.color}">${x.s.name}</text>`
  $('curve').innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart">${g}</svg>`
}

// ── Boundary map ──────────────────────────────────────────────────────────────
function renderMap() {
  const { doc } = lab
  const N = doc.C.length
  const W = 1100, lane = 30, top = 34, labelW = 150
  const X = p => labelW + p / N * (W - labelW - 10)
  const q = lab.evSet[state.q]
  let g = ''
  // section ticks
  doc.provisions.forEach(p => {
    const major = /^\d+\.1$/.test(p.id)
    g += `<line x1="${X(p.start)}" x2="${X(p.start)}" y1="${major ? 12 : 22}" y2="${top + STRATEGIES.length * (lane + 14)}" class="${major ? 'tick-major' : 'tick'}"/>`
    if (major) g += `<text x="${X(p.start) + 3}" y="20" class="axis">§${p.id.split('.')[0]}</text>`
  })
  // evidence bands
  q.ev.forEach(e => { if (e.s >= 0) g += `<rect x="${X(e.s)}" y="${top - 6}" width="${Math.max(3, X(e.e) - X(e.s))}" height="${STRATEGIES.length * (lane + 14)}" class="ev-band"/>` })
  STRATEGIES.forEach((s, k) => {
    const r = lab.results[s.key]
    const y0 = top + k * (lane + 14)
    const picked = new Set(r.perQ[state.q][state.mode].picked)
    g += `<text x="0" y="${y0 + lane / 2 + 4}" class="lbl" fill="${s.color}">${s.name}</text>`
    const sorted = r.chunks.map((c, i) => ({ c, i })).sort((a, b) => a.c.start - b.c.start)
    let lastEnd = -1, row = 0
    sorted.forEach(({ c, i }, n) => {
      row = c.start < lastEnd ? 1 - row : 0
      lastEnd = Math.max(lastEnd, c.end)
      const h = lane / 2 - 1, y = y0 + row * (h + 2)
      const op = n % 2 ? 0.55 : 0.9
      g += `<rect x="${X(c.start)}" y="${y}" width="${Math.max(1.5, X(c.end) - X(c.start) - 1)}" height="${h}" rx="2" fill="${s.color}" fill-opacity="${op}" class="${picked.has(i) ? 'picked' : ''}"><title>chunk ${i} · ${c.text.length} chars</title></rect>`
    })
    // scissors where an evidence sentence is cut by every chunk
    q.ev.forEach(e => {
      if (e.s < 0) return
      if (!r.chunks.some(c => c.start <= e.s && c.end >= e.e)) g += `<text x="${X((e.s + e.e) / 2)}" y="${y0 + lane + 10}" class="cut" text-anchor="middle">✂ split</text>`
    })
  })
  const H = top + STRATEGIES.length * (lane + 14) + 4
  $('map').innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart map">${g}</svg>`
}

// ── Inspector ─────────────────────────────────────────────────────────────────
function highlight(text, evs) {
  let h = esc(text)
  for (const e of evs) { const t = esc(e.text); if (h.includes(t)) h = h.split(t).join(`<mark>${t}</mark>`) }
  return h.replace(/\n/g, '<br>')
}
function renderInspector() {
  $('qSelect').innerHTML = lab.evSet.map((q, i) => `<option value="${i}" ${i === state.q ? 'selected' : ''}>${esc(q.q)}  (${q.ev.length} evidence)</option>`).join('')
  const q = lab.evSet[state.q]
  $('inspGrid').innerHTML = STRATEGIES.map(s => {
    const r = lab.results[s.key]
    const p = r.perQ[state.q][state.mode]
    const status = p.complete ? ['✓ complete', 'ok'] : p.partial ? ['½ partial', 'part'] : ['✗ missed', 'miss']
    const missing = q.ev.filter((e, i) => !p.found[i])
    return `<div class="insp-card" style="--c:${s.color}">
      <div class="insp-head"><span class="sname">${s.name}</span><span class="badge ${status[1]}">${status[0]}</span></div>
      <div class="insp-meta">${p.picked.length} chunks · ${p.used}/${state.budget} words</div>
      <div class="insp-body">${p.picked.map(i => `<div class="ctx"><div class="ctx-id">#${i}${p.seeds.includes(i) ? ' · seed' : (state.mode === 'graph' ? ' · via graph' : '')}</div>${highlight(r.chunks[i].text, q.ev)}</div>`).join('') || '<div class="muted">nothing retrieved</div>'}</div>
      ${missing.length ? `<div class="insp-miss"><b>Missing:</b> ${missing.map(e => esc(e.text)).join('<br>')}</div>` : ''}
    </div>`
  }).join('')
}

// ── Chunk graph (arc diagram) ─────────────────────────────────────────────────
function renderGraph() {
  $('graphStrategy').innerHTML = STRATEGIES.map(s => `<option value="${s.key}" ${s.key === state.graphKey ? 'selected' : ''}>${s.name}</option>`).join('')
  const s = STRATEGIES.find(x => x.key === state.graphKey)
  const r = lab.results[s.key]
  const q = lab.evSet[state.q]
  const res = r.perQ[state.q].graph
  const W = 560, H = 300, mid = 170, pad = 18
  const n = r.chunks.length
  const order = r.chunks.map((c, i) => i).sort((a, b) => r.chunks[a].start - r.chunks[b].start)
  const pos = new Array(n)
  order.forEach((i, k) => { pos[i] = pad + (n === 1 ? 0.5 : k / (n - 1)) * (W - 2 * pad) })
  const hasEv = i => q.ev.some(e => r.chunks[i].text.replace(/\s+/g, ' ').includes(e.text))
  let g = `<line x1="${pad}" x2="${W - pad}" y1="${mid}" y2="${mid}" class="spine"/>`
  for (const e of r.graph.edges) {
    if (e.type === 'adjacent') continue
    const x1 = pos[e.a], x2 = pos[e.b]
    const up = e.type === 'xref' || e.type === 'section'
    const h = Math.min(up ? 300 : 230, 18 + Math.abs(x2 - x1) * 0.55)
    const cy = up ? mid - h : mid + h * 0.55
    g += `<path d="M ${x1} ${mid} Q ${(x1 + x2) / 2} ${cy} ${x2} ${mid}" class="arc arc-${e.type}" style="stroke-width:${e.type === 'lexical' ? Math.min(2.4, e.w) : 1.6}"/>`
  }
  const picked = new Set(res.picked), seeds = new Set(res.seeds)
  const maxS = Math.max(...res.scores, 1e-9)
  r.chunks.forEach((c, i) => {
    const rad = 3 + 6 * (res.scores[i] / maxS)
    const cls = seeds.has(i) && picked.has(i) ? 'n-seed' : picked.has(i) ? 'n-exp' : seeds.has(i) ? 'n-seedx' : 'n-off'
    g += `<circle cx="${pos[i]}" cy="${mid}" r="${rad}" class="node ${cls}"><title>chunk ${i} · score ${res.scores[i].toFixed(2)}</title></circle>`
    if (hasEv(i)) g += `<text x="${pos[i]}" y="${mid + rad + 13}" class="ev-star" text-anchor="middle">★</text>`
  })
  $('graph').innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart">${g}</svg>`
  $('graphLegend').innerHTML = `<span><i class="lg n-seed"></i>seed (BM25 top-3), kept</span><span><i class="lg n-exp"></i>added by propagation</span><span><i class="lg n-off"></i>not in context</span><span><b class="ev-star">★</b> holds evidence</span><span><i class="ln xref"></i>cross-ref</span><span><i class="ln section"></i>same provision</span><span><i class="ln lexical"></i>lexical</span>`
}

// ── Papers ────────────────────────────────────────────────────────────────────
const PAPERS = [
  ['QESChunker', '2610.36557', 'Oct 27', 'Chunking as supervised optimisation over question–evidence pairs: exact DP when chunks don\'t overlap, greedy submodular when they do.', 'QES · exact DP and QES · overlap greedy'],
  ['StrucChunk', '2610.26313', 'Oct 19', 'Provision-boundary chunks, heading path, cross-references resolved before embedding, BM25 + dense fused with RRF.', 'Structure-aware chunker + the boundary / x-ref metrics'],
  ['SpectRAG', '2610.36991', 'Oct 27', 'Weighted chunk graph (adjacency, sections, entities, citations), relevance propagation, token-budgeted context assembly.', 'Chunk graph mode (personalised PageRank, budgeted fill)'],
  ['The Bottleneck Is Retrieval, Not Reading', '2610.26951', 'Oct 18', 'Retrieval surfaces some but not all chunks multi-fact questions need; chunking drops document-level context.', 'Partial vs complete recall columns'],
  ['Pre-retrieval Query Clustering for Adaptive Top-k', '2609.13489', 'Sep 11', 'Pick retrieval depth per query cluster before retrieving; +36% F1, −14% tokens on easy clusters.', 'Budget curve (how much depth each chunker needs)'],
  ['Parser, Chunking, and Embedding Interactions', '2609.31660', 'Sep 15', '54-configuration factorial study: parser and chunker choices interact; no single retriever wins everywhere.', 'Why every knob here is exposed instead of fixed'],
]
function renderPapers() {
  $('paperGrid').innerHTML = PAPERS.map(([t, id, d, what, here]) => `<a class="paper" href="https://arxiv.org/abs/${id}" target="_blank" rel="noopener">
    <div class="paper-top"><span class="paper-title">${t}</span><span class="paper-id">arXiv ${id} · ${d} 2026</span></div>
    <div class="paper-what">${what}</div><div class="paper-here">In this lab → ${here}</div></a>`).join('')
}

// ── Wiring ────────────────────────────────────────────────────────────────────
function syncControls() {
  $('sL').value = state.L; $('vL').textContent = state.L + ' chars'
  $('sB').value = state.budget; $('vB').textContent = state.budget + ' words'
  for (const [seg, key] of [['segSplit', 'split'], ['segMode', 'mode']])
    $(seg).querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === state[key]))
  $('cHeaders').checked = state.headers; $('cRefs').checked = state.resolveRefs
}
$('sL').addEventListener('input', e => { state.L = +e.target.value; syncControls(); run() })
$('sB').addEventListener('input', e => { state.budget = +e.target.value; syncControls(); run() })
for (const [seg, key] of [['segSplit', 'split'], ['segMode', 'mode']])
  $(seg).addEventListener('click', e => { if (e.target.dataset.v) { state[key] = e.target.dataset.v; syncControls(); run() } })
$('cHeaders').addEventListener('change', e => { state.headers = e.target.checked; run() })
$('cRefs').addEventListener('change', e => { state.resolveRefs = e.target.checked; run() })
$('qSelect').addEventListener('change', e => { state.q = +e.target.value; renderMap(); renderInspector(); renderGraph() })
$('graphStrategy').addEventListener('change', e => { state.graphKey = e.target.value; renderGraph() })
$('btnCorpus').addEventListener('click', () => $('corpusPanel').classList.toggle('hidden'))
const fillCorpus = () => { $('taDoc').value = state.doc; $('taQE').value = JSON.stringify(state.qe, null, 1) }
$('btnReset').addEventListener('click', () => { state.doc = SAMPLE_DOC; state.qe = SAMPLE_QE; fillCorpus(); run(); $('corpusMsg').textContent = 'Sample restored.' })
$('btnApply').addEventListener('click', () => {
  try {
    const qe = JSON.parse($('taQE').value)
    if (!Array.isArray(qe) || !qe.length) throw new Error('Q/E must be a non-empty array')
    state.doc = $('taDoc').value; state.qe = qe; run()
    $('corpusMsg').textContent = lab.missing.length ? `⚠ ${lab.missing.length} evidence string(s) not found verbatim in the corpus.` : 'All evidence strings located.'
  } catch (err) { $('corpusMsg').textContent = '⚠ ' + err.message }
})

fillCorpus(); renderPapers(); syncControls(); run()
