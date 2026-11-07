// advisor.js — UI for the Strategy Advisor tab of the Chunking Lab page
import { CATEGORIES, STRATEGY_COLORS, ADVISOR_SAMPLES, profileDocument, recommend, previewChunks } from './utils/chunkAdvisor.js'

const $ = id => document.getElementById(id)
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]))
const pct = v => Math.round(v * 100) + '%'

const PAPER_NAMES = {
  '2610.26313': 'StrucChunk', '2610.36557': 'QESChunker', '2610.36991': 'SpectRAG',
  '2610.26951': 'The Bottleneck Is Retrieval, Not Reading', '2609.13489': 'Adaptive Top-k',
  '2609.31660': 'Parser, Chunking & Embedding Interactions',
}
const SAMPLE_LABELS = { contract: 'Contract', transcript: 'Meeting transcript', email: 'Email thread', faq: 'FAQ', code: 'Source code', table: 'CSV catalogue' }

const params = new URLSearchParams(location.search)
const st = {
  tab: params.get('tab') === 'advisor' ? 'advisor' : 'lab',
  cat: CATEGORIES.some(c => c.key === params.get('cat')) ? params.get('cat') : 'legal',
  budget: 'medium', retriever: 'hybrid', qe: 'no',
  profile: null, text: '',
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function showTab(tab) {
  st.tab = tab
  $('tabbar').querySelectorAll('button').forEach(b => { b.classList.toggle('on', b.dataset.tab === tab); b.setAttribute('aria-selected', b.dataset.tab === tab) })
  $('tabLab').classList.toggle('hidden', tab !== 'lab')
  $('tabAdvisor').classList.toggle('hidden', tab !== 'advisor')
  const u = new URL(location.href)
  if (tab === 'advisor') u.searchParams.set('tab', 'advisor'); else u.searchParams.delete('tab')
  history.replaceState(null, '', u)
}
$('tabbar').addEventListener('click', e => { const b = e.target.closest('button'); if (b) showTab(b.dataset.tab) })

// ── Category grid ─────────────────────────────────────────────────────────────
function renderCats() {
  const detected = st.profile ? st.profile.best : null
  $('catGrid').innerHTML = CATEGORIES.map(c => `<button class="cat${c.key === st.cat ? ' on' : ''}" data-k="${c.key}" style="--c:${STRATEGY_COLORS[c.strategy]}">
    <span class="cat-icon">${c.icon}</span>
    <span class="cat-name">${c.name}${c.key === detected ? ' <span class="cat-detected">detected</span>' : ''}</span>
    <span class="cat-ex">${c.examples}</span></button>`).join('')
}
$('catGrid').addEventListener('click', e => {
  const b = e.target.closest('.cat'); if (!b) return
  st.cat = b.dataset.k
  renderCats(); renderRec(); renderPreview()
})

// ── Constraints ───────────────────────────────────────────────────────────────
for (const [seg, key] of [['segBudget', 'budget'], ['segRetriever', 'retriever'], ['segQE', 'qe']]) {
  $(seg).addEventListener('click', e => {
    const v = e.target.dataset.v; if (!v) return
    st[key] = v
    $(seg).querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v))
    renderRec(); renderPreview()
  })
}

const current = () => recommend(st.cat, { budget: st.budget, retriever: st.retriever, qe: st.qe === 'yes' }, st.profile)

// ── Recommendation ────────────────────────────────────────────────────────────
function renderRec() {
  const r = current()
  const color = STRATEGY_COLORS[r.strategy]
  $('recPanel').innerHTML = `
    <div class="panel-header"><span class="panel-icon">2</span><span>Recommendation for ${r.cat.icon} ${esc(r.cat.name)}</span></div>
    <div class="rec" style="--c:${color}">
      <div class="rec-main">
        <div class="rec-label">Use</div>
        <div class="rec-strategy">${esc(r.strategyName)}</div>
        <table class="rec-params">${r.params.map(([k, v]) => `<tr><th>${k}</th><td>${esc(v)}</td></tr>`).join('')}</table>
      </div>
      <div class="rec-side">
        <h4>Why</h4><ul>${r.cat.why.map(w => `<li>${esc(w)}</li>`).join('')}</ul>
        <h4>For your setup</h4><ul class="rec-notes">${r.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>
        <h4>Watch out for</h4><ul class="rec-pitfalls">${r.cat.pitfalls.map(p => `<li>${esc(p)}</li>`).join('')}</ul>
        <div class="rec-papers">${r.cat.papers.map(id => `<a href="https://arxiv.org/abs/${id}" target="_blank" rel="noopener">${PAPER_NAMES[id]} · arXiv ${id}</a>`).join('')}</div>
      </div>
    </div>`
}

// ── Detection ─────────────────────────────────────────────────────────────────
$('sampleBtns').innerHTML = Object.keys(ADVISOR_SAMPLES).map(k => `<button class="chip" data-s="${k}">${SAMPLE_LABELS[k]}</button>`).join('')
$('sampleBtns').addEventListener('click', e => {
  const k = e.target.dataset.s; if (!k) return
  $('taAdvisor').value = ADVISOR_SAMPLES[k]
  detect()
})
$('btnDetect').addEventListener('click', detect)

function detect() {
  const text = $('taAdvisor').value
  if (text.trim().length < 80) { $('detectMsg').textContent = '⚠ Paste at least a few lines of text.'; return }
  st.text = text
  st.profile = profileDocument(text)
  st.cat = st.profile.best
  const top = st.profile.ranked[0]
  $('detectMsg').textContent = top.score > 0.4 ? `Detected: ${CATEGORIES.find(c => c.key === st.cat).name} (${pct(top.share)} of the evidence).` : 'No strong signal; defaulting to long-form prose.'
  renderProfile(); renderCats(); renderRec(); renderPreview()
}

function renderProfile() {
  const s = st.profile.signals
  const chips = [
    ['words', s.words], ['markdown headings', s.mdHeadings], ['numbered clauses', s.numberedHeadings],
    ['cross-references', s.crossRefs], ['speaker turns', s.speakerTurns], ['email headers', s.emailHeaders],
    ['quoted lines', s.quoted], ['Q&A lines', s.qaPairs], ['code lines', s.codeLines], ['table rows', s.tableLines],
    ['avg words / sentence', Math.round(s.avgSentenceWords)],
  ].filter(([k, v]) => v || k === 'words')
  const bars = st.profile.ranked.slice(0, 4).filter(x => x.score > 0).map(x => {
    const c = CATEGORIES.find(c => c.key === x.key)
    return `<div class="pbar"><span class="pbar-name">${c.icon} ${c.name}</span><div class="cell-bar"><div class="cell-fill" style="width:${pct(x.share)};background:${STRATEGY_COLORS[c.strategy]}"></div><span>${pct(x.share)}</span></div></div>`
  }).join('')
  $('profileOut').innerHTML = `<div class="profile"><div><h4>Signals found</h4><div class="sig">${chips.map(([k, v]) => `<span class="chip static"><b>${v}</b> ${k}</span>`).join('')}</div></div>
    <div><h4>Category scores</h4>${bars}</div></div>`
}

// ── Preview ───────────────────────────────────────────────────────────────────
const USED_NAMES = { struct: 'structure-aware', sentence: 'sentence windows', record: 'record / block split', fixed: 'fixed windows', qesov: 'sentence windows (QES needs Q/E pairs to run)' }

function renderPreview() {
  if (!st.text) { $('previewPanel').classList.add('hidden'); return }
  const r = current()
  const pv = previewChunks(st.text, r.strategy, r.labSize, r.overlap || 0)
  const tooBig = pv.maxWords > r.budgetWords / 2
  $('previewPanel').classList.remove('hidden')
  $('previewPanel').innerHTML = `
    <div class="panel-header"><span class="panel-icon">3</span><span>Your document, chunked with this recipe</span><span class="hdr-note">${USED_NAMES[pv.used]} · ~${r.labSize}-char target</span></div>
    <div class="pv-stats">
      <div><b>${pv.count}</b><span>chunks</span></div>
      <div><b>${pv.avgChars}</b><span>avg chars</span></div>
      <div class="${tooBig ? 'warn' : ''}"><b>${pv.maxWords}</b><span>largest chunk (words)</span></div>
      ${pv.headingStartShare !== null ? `<div><b>${pct(pv.headingStartShare)}</b><span>start at a heading</span></div>` : ''}
      ${pv.used === 'struct' ? `<div><b>${pv.resolvedRefs}</b><span>cross-refs resolved</span></div>` : ''}
    </div>
    ${tooBig ? `<p class="callout warn-callout">The largest chunk is more than half the ${r.budgetWords}-word budget, so at most one such chunk fits per question. Split it further or raise the budget.</p>` : ''}
    <div class="pv-chunks">${pv.chunks.slice(0, 8).map((c, i) => `<div class="ctx"><div class="ctx-id">#${i} · ${c.text.length} chars · ${c.words} words</div>${esc(c.text).replace(/\n/g, '<br>')}</div>`).join('')}
    ${pv.count > 8 ? `<div class="muted">…and ${pv.count - 8} more</div>` : ''}</div>
    <div class="corpus-actions"><span class="muted">To score strategies on this document, add question–evidence pairs in the Lab.</span><button class="btn btn-ghost" id="btnToLab">Open in Lab →</button></div>`
  $('btnToLab').addEventListener('click', () => {
    $('taDoc').value = st.text
    $('corpusPanel').classList.remove('hidden')
    $('corpusMsg').textContent = 'Document loaded from the Advisor. Replace the Q/E pairs with questions about it, then Re-run.'
    showTab('lab')
    $('corpusPanel').scrollIntoView({ behavior: 'smooth' })
  })
}

renderCats(); renderRec(); showTab(st.tab)
