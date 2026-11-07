// prodLab.js — Production Problems in RAG: a deterministic simulation of keeping a vector
// index in sync with documents that keep changing. Nine failure modes, each run twice:
// once without its fix and once with it. No model download, no server: embeddings are
// feature-hashed bag-of-words vectors, so two "models" with different seeds really do
// live in incompatible vector spaces, just like two real embedding models.

import { tokenize } from './text.js'
import { HANDBOOK, KB, QUESTIONS, cloneDoc } from './prodCorpus.js'

// ── primitives ────────────────────────────────────────────────────────────────
export function fnv1a(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) }
  return (h >>> 0).toString(16).padStart(8, '0')
}
export function rng(seed) { // mulberry32
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 }
}

export const MODELS = { v1: { id: 'embed-v1', seed: 'v1|' }, v2: { id: 'embed-v2', seed: 'v2|' } }
const DIM = 1024
export function embed(text, model = MODELS.v1) {
  const v = new Float32Array(DIM)
  for (const tok of tokenize(text)) {
    const h = parseInt(fnv1a(model.seed + tok), 16)
    v[h % DIM] += (h & 0x10000) ? 1 : -1
  }
  let n = 0; for (let i = 0; i < DIM; i++) n += v[i] * v[i]
  n = Math.sqrt(n) || 1; for (let i = 0; i < DIM; i++) v[i] /= n
  return v
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }

// ── documents and chunkers ────────────────────────────────────────────────────
export function render(doc) {
  return doc.sections.map(s => `## ${s.title}\n\n${s.paras.join('\n\n')}`).join('\n\n')
}
// Fixed-size character windows: the default in many tutorials.
export function fixedChunks(doc, size = 400) {
  const text = render(doc), out = []
  const starts = []; let off = 0
  doc.sections.forEach((s, i) => { const t = `## ${s.title}\n\n${s.paras.join('\n\n')}`; starts.push([off, s.id]); off += t.length + 2 })
  const secAt = p => { let id = starts[0][1]; for (const [o, sid] of starts) if (o <= p) id = sid; return id }
  for (let p = 0; p < text.length; p += size) out.push({ text: text.slice(p, p + size), section: secAt(p), pos: p })
  return out
}
// Stable chunks: one chunk per paragraph, anchored to the heading path. A local edit stays local.
export function stableChunks(doc) {
  const out = []
  doc.sections.forEach(s => s.paras.forEach((p, i) => out.push({ text: `${doc.title} › ${s.title}\n${p}`, section: s.id, pos: i })))
  return out
}

// ── a tiny vector store ───────────────────────────────────────────────────────
export class Store {
  constructor() { this.rows = new Map(); this.active = new Map(); this.seq = 0; this.embedCalls = 0 }
  add(row) { const id = ++this.seq; this.rows.set(id, { visible: true, ...row, id }); return id }
  embedRow(chunk, docId, extra = {}, model = MODELS.v1) {
    this.embedCalls++
    return this.add({ docId, text: chunk.text, section: chunk.section, hash: fnv1a(chunk.text), vec: embed(chunk.text, model), model: model.id, ...extra })
  }
  deleteWhere(fn) { let n = 0; for (const [id, r] of this.rows) if (fn(r)) { this.rows.delete(id); n++ } return n }
  count(fn = () => true) { let n = 0; for (const r of this.rows.values()) if (fn(r)) n++; return n }
  search(query, { k = 5, model = MODELS.v1, filter = () => true } = {}) {
    const q = embed(query, model), hits = []
    for (const r of this.rows.values()) if (r.visible && filter(r)) hits.push({ r, s: dot(q, r.vec) })
    hits.sort((a, b) => b.s - a.s)
    return hits.slice(0, k)
  }
}

export function indexAll(docs, { chunker = stableChunks, model = MODELS.v1, version = 1 } = {}) {
  const st = new Store()
  for (const d of docs) { for (const c of chunker(d)) st.embedRow(c, d.id, { version }, model); st.active.set(d.id, version) }
  return st
}
const corpus = () => [cloneDoc(HANDBOOK), ...KB.map(cloneDoc)]
const isActive = st => r => st.active.get(r.docId) === r.version

// Deterministic edits used throughout
export const EDIT_SENTENCE = { section: 2, para: 0, text: ' Requests for more than ten consecutive days also need sign-off from your director.' } // §3 Leave
function editOneSentence(doc) { const d = cloneDoc(doc); d.sections[EDIT_SENTENCE.section].paras[EDIT_SENTENCE.para] += EDIT_SENTENCE.text; return d }
// v13 policy change touching four paragraphs in four sections
export const POLICY_EDITS = [
  { s: 0, p: 2, from: 'up to 2 days per week', to: 'up to 3 days per week' },
  { s: 2, p: 0, from: '15 days of paid annual leave', to: '20 days of paid annual leave' },
  { s: 3, p: 0, from: 'up to 2 days per week', to: 'up to 3 days per week' },
  { s: 7, p: 1, from: 'Annual leave is 15 days per year', to: 'Annual leave is 20 days per year' },
]
function applyPolicy(doc) { const d = cloneDoc(doc); for (const e of POLICY_EDITS) d.sections[e.s].paras[e.p] = d.sections[e.s].paras[e.p].replace(e.from, e.to); return d }

const PROJ_VECTORS = 20_000_000, EMBEDS_PER_SEC = 2000
const fmtDur = s => s < 1 ? '<1 s' : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`

// ── Rung 1: every edit triggers a full re-index → store document IDs ─────────────
export function rung1() {
  const docs = corpus(), total = docs.reduce((n, d) => n + fixedChunks(d).length, 0)
  const st = indexAll(docs, { chunker: fixedChunks })
  const edited = editOneSentence(docs[0])
  // with doc IDs: delete this doc's vectors, re-chunk, re-embed only this doc
  st.embedCalls = 0
  const deleted = st.deleteWhere(r => r.docId === 'handbook')
  for (const c of fixedChunks(edited)) st.embedRow(c, 'handbook', { version: 1 })
  return {
    total, docs: docs.length,
    without: { embeds: total, deleted: total, projected: fmtDur(PROJ_VECTORS / EMBEDS_PER_SEC), projectedEmbeds: PROJ_VECTORS },
    with: { embeds: st.embedCalls, deleted, projected: fmtDur(st.embedCalls / EMBEDS_PER_SEC), projectedEmbeds: st.embedCalls },
  }
}

// diff helper: which new chunks have a hash that existed before?
function hashDiff(oldChunks, newChunks) {
  const pool = new Map()
  for (const c of oldChunks) { const h = fnv1a(c.text); pool.set(h, (pool.get(h) || 0) + 1) }
  return newChunks.map(c => { const h = fnv1a(c.text), n = pool.get(h) || 0; if (n) pool.set(h, n - 1); return { ...c, hash: h, reused: n > 0 } })
}

// ── Rung 2: a tiny edit re-embeds the whole document → content hashing ─────────
export function rung2() {
  const before = fixedChunks(HANDBOOK), after = fixedChunks(editOneSentence(HANDBOOK))
  const strip = hashDiff(before, after)
  const changed = strip.filter(c => !c.reused).length
  const firstChange = strip.findIndex(c => !c.reused)
  return {
    chunks: after.length,
    without: { embeds: after.length, strip: strip.map(c => ({ ...c, reused: false })) },
    with: { embeds: changed, strip, firstChange, cascade: strip.length - firstChange },
  }
}

// ── Rung 3: boundaries shift and every later hash changes → stable chunking ─────
export function rung3() {
  const r2 = rung2()
  const before = stableChunks(HANDBOOK), after = stableChunks(editOneSentence(HANDBOOK))
  const strip = hashDiff(before, after)
  return {
    without: { embeds: r2.with.embeds, strip: r2.with.strip, chunks: r2.chunks, label: 'Fixed 400-char windows + hashing' },
    with: { embeds: strip.filter(c => !c.reused).length, strip, chunks: strip.length, label: 'Paragraph chunks under heading path + hashing' },
  }
}

// helper: answer-ish view of hits
const hitView = h => ({ docId: h.r.docId, section: h.r.section, version: h.r.version, model: h.r.model, text: h.r.text, score: +h.s.toFixed(3), tag: h.r.tag })

// ── Rung 4: crash mid-update mixes v12 and v13 → document versioning ────────────
export function rung4({ crashAt = 0.5 } = {}) {
  const v12 = HANDBOOK, v13 = applyPolicy(HANDBOOK)
  const oldC = stableChunks(v12), newC = stableChunks(v13)
  const changedNew = hashDiff(oldC, newC).filter(c => !c.reused)
  const goneOld = hashDiff(newC, oldC).filter(c => !c.reused)
  const Q = 'how many days of paid annual leave per year'

  // Without versioning: in-place delete + insert per changed paragraph, crash halfway
  const a = indexAll([cloneDoc(v12), ...KB], { version: 12 })
  for (const r of a.rows.values()) if (r.docId === 'handbook') r.tag = 'v12'
  const ops = changedNew.map((c, i) => ({ del: goneOld[i], ins: c }))
  const done = Math.floor(ops.length * crashAt)
  ops.slice(0, done).forEach(op => { a.deleteWhere(r => r.docId === 'handbook' && r.hash === fnv1a(op.del.text)); a.embedRow(op.ins, 'handbook', { version: 12, tag: 'v13' }) })
  const hitsA = a.search(Q, { k: 4, filter: r => r.docId === 'handbook' }).map(hitView)

  // With versioning: build v13 next to v12, crash before activation, reads stay on v12
  const b = indexAll([cloneDoc(v12), ...KB], { version: 12 })
  for (const r of b.rows.values()) if (r.docId === 'handbook') r.tag = 'v12'
  const reuse = new Map([...b.rows.values()].filter(r => r.docId === 'handbook').map(r => [r.hash, r]))
  let written = 0
  const plan = hashDiff(oldC, newC)
  for (const c of plan) {
    if (written >= Math.floor(plan.length * crashAt)) break
    const old = c.reused && reuse.get(c.hash)
    if (old) b.add({ ...old, id: undefined, version: 13, tag: 'v13' }); else b.embedRow(c, 'handbook', { version: 13, tag: 'v13' })
    written++
  }
  const hitsB = b.search(Q, { k: 4, filter: r => r.docId === 'handbook' && isActive(b)(r) }).map(hitView)

  const facts = hits => [...new Set(hits.map(h => (h.text.match(/(\d+) days of paid annual leave|Annual leave is (\d+) days/) || []).slice(1).find(Boolean)).filter(Boolean))]
  return {
    query: Q, edits: POLICY_EDITS.length, crashAt,
    without: { applied: done, of: ops.length, hits: hitsA, facts: facts(hitsA), sections: summarizeVersions(a) },
    with: { written, of: plan.length, hits: hitsB, facts: facts(hitsB), active: 12, pending: 13 },
  }
}
function summarizeVersions(st) {
  const m = {}
  for (const r of st.rows.values()) if (r.docId === 'handbook') { (m[r.section] ||= new Set()).add(r.tag) }
  return Object.fromEntries(Object.entries(m).map(([k, v]) => [k, [...v]]))
}

// ── Rung 5: new version activated before every write lands → atomic switch ──────
// Writes go to an async upsert queue (as with most managed vector DBs). We search while
// the queue is `progress` drained.
export function rung5({ progress = 0.45 } = {}) {
  const v13 = applyPolicy(HANDBOOK)
  const Q = 'how many days per week can I work from home remotely'
  function deploy(atomic) {
    const st = indexAll([cloneDoc(HANDBOOK), ...KB], { version: 12 })
    const newC = stableChunks(v13)
    const r = rng(7), ids = []
    for (const c of newC) ids.push(st.embedRow(c, 'handbook', { version: 13 }))
    for (const id of ids) st.rows.get(id).visible = false
    // drain the queue out of order, like batched indexing does
    const order = ids.map(id => [r(), id]).sort((x, y) => x[0] - y[0]).map(x => x[1])
    const landed = Math.floor(order.length * progress)
    order.slice(0, landed).forEach(id => { st.rows.get(id).visible = true })
    const allLanded = landed === order.length
    if (!atomic || allLanded) st.active.set('handbook', 13) // non-atomic: flip as soon as writes are queued
    const act = st.active.get('handbook')
    const visibleActive = [...st.rows.values()].filter(x => x.docId === 'handbook' && x.version === act && x.visible)
    const coverage = newC.map((c, i) => ({ section: c.section, para: c.pos, present: visibleActive.some(x => x.section === c.section && x.text.split('\n')[1] === c.text.split('\n')[1]) || (act === 12) }))
    const hits = st.search(Q, { k: 3, filter: x => (x.docId !== 'handbook') || x.version === act }).map(hitView)
    return { activeVersion: act, landed, of: order.length, coverage, coveragePct: coverage.filter(c => c.present).length / coverage.length, hits }
  }
  return { query: Q, progress, without: deploy(false), with: deploy(true) }
}

// ── Rung 6: old versions pile up → garbage collection ─────────────────────────
export function rung6({ edits = 300, every = 10, keep = 1 } = {}) {
  const r = rng(42)
  function run(gc) {
    const st = indexAll([cloneDoc(HANDBOOK), ...KB], { version: 1 })
    let doc = cloneDoc(HANDBOOK), v = 1
    const series = [[0, st.rows.size, 0]]
    for (let e = 1; e <= edits; e++) {
      const s = Math.floor(r() * doc.sections.length), p = Math.floor(r() * doc.sections[s].paras.length)
      doc = cloneDoc(doc); doc.sections[s].paras[p] = doc.sections[s].paras[p].replace(/\s*\(rev \d+\)$/, '') + ` (rev ${e})`
      const prev = new Map([...st.rows.values()].filter(x => x.docId === 'handbook' && x.version === v).map(x => [x.hash, x]))
      v++
      for (const c of stableChunks(doc)) { const h = fnv1a(c.text), old = prev.get(h); if (old) st.add({ ...old, id: undefined, version: v }); else st.embedRow(c, 'handbook', { version: v }) }
      st.active.set('handbook', v)
      if (gc) st.deleteWhere(x => x.docId === 'handbook' && x.version < v - keep)
      if (e % every === 0) series.push([e, st.rows.size, st.count(x => !isActive(st)(x))])
    }
    const total = st.rows.size, stale = st.count(x => !isActive(st)(x))
    return { series, total, stale, stalePct: stale / total }
  }
  return { edits, keep, without: run(false), with: run(true) }
}

// ── Rung 7: embedding model upgrade mixes vector spaces → embedding versioning ──
export function recallAt(st, { k = 5, model = MODELS.v1, filter = () => true } = {}) {
  let hit = 0
  for (const q of QUESTIONS) if (st.search(q.q, { k, model, filter }).some(h => h.r.docId === q.docId)) hit++
  return hit / QUESTIONS.length
}
export function rung7({ editedShare = 0.25 } = {}) {
  const docs = corpus()
  const base = indexAll(docs, { model: MODELS.v1 })
  const before = recallAt(base, { model: MODELS.v1 })
  const r = rng(99), edited = new Set(docs.filter(() => r() < editedShare).map(d => d.id))

  // Without: pipeline switched to v2; edited docs re-embedded with v2; queries now use v2
  const a = indexAll(docs, { model: MODELS.v1 })
  for (const d of docs) if (edited.has(d.id)) { a.deleteWhere(x => x.docId === d.id); for (const c of stableChunks(d)) a.embedRow(c, d.id, { version: 1 }, MODELS.v2) }
  const mixedV2 = recallAt(a, { model: MODELS.v2 })
  const mixedV1 = recallAt(a, { model: MODELS.v1 })

  // With: model id on every vector; serving stays pinned to v1 while a shadow v2 index is
  // built deliberately; edits are dual-written; cut over when the shadow index is complete.
  const b = indexAll(docs, { model: MODELS.v1 })
  const during = recallAt(b, { model: MODELS.v1, filter: x => x.model === MODELS.v1.id })
  for (const d of docs) for (const c of stableChunks(d)) b.embedRow(c, d.id, { version: 1 }, MODELS.v2)
  const after = recallAt(b, { model: MODELS.v2, filter: x => x.model === MODELS.v2.id })
  b.deleteWhere(x => x.model === MODELS.v1.id)

  const mix = st => ({ v1: st.count(x => x.model === MODELS.v1.id), v2: st.count(x => x.model === MODELS.v2.id) })
  return {
    questions: QUESTIONS.length, editedDocs: edited.size, before,
    without: { recall: mixedV2, recallOldQueries: mixedV1, mix: mix(a) },
    with: { recallDuring: during, recall: after, mix: mix(b), reembedded: b.rows.size },
  }
}

const refreshYears = texts => { for (const t of texts) { const m = t.match(/every (\d) years/); if (m) return m[1] } }

// ── Rung 8: the update event never arrives → outbox + reconciliation sweep ──────
export function rung8({ events = 40, drop = [6, 23] } = {}) {
  const Q = 'how often are laptops refreshed'
  function run(reconcile) {
    const r = rng(5)
    const source = new Map(corpus().map(d => [d.id, d]))
    const st = indexAll([...source.values()], { version: 1 })
    const indexed = new Map([...source.keys()].map(id => [id, fnv1a(render(source.get(id)))]))
    const log = []
    const ids = [...source.keys()]
    for (let e = 0; e < events; e++) {
      const id = e === drop[0] ? 'handbook' : ids[1 + Math.floor(r() * (ids.length - 1))]
      let d = cloneDoc(source.get(id))
      if (id === 'handbook') d.sections[4].paras[0] = d.sections[4].paras[0].replace('every 3 years', 'every 4 years')
      else d.sections[0].paras[0] += ` Updated ${e + 1}.`
      source.set(id, d)
      const dropped = drop.includes(e)
      if (!dropped) { st.deleteWhere(x => x.docId === id); for (const c of stableChunks(d)) st.embedRow(c, id, { version: 1 }); indexed.set(id, fnv1a(render(d))) }
      log.push({ e: e + 1, docId: id, dropped })
    }
    let drift = [], repaired = []
    if (reconcile) {
      drift = [...source.keys()].filter(id => indexed.get(id) !== fnv1a(render(source.get(id))))
      for (const id of drift) { st.deleteWhere(x => x.docId === id); for (const c of stableChunks(source.get(id))) st.embedRow(c, id, { version: 1 }); indexed.set(id, fnv1a(render(source.get(id)))); repaired.push(id) }
    }
    const stale = [...source.keys()].filter(id => indexed.get(id) !== fnv1a(render(source.get(id))))
    const hits = st.search(Q, { k: 3, filter: x => x.docId === 'handbook' }).map(hitView)
    return { log, stale, drift, repaired, hits, answer: refreshYears(hits.map(h => h.text)) }
  }
  return { query: Q, truth: '4', events, without: run(false), with: run(true) }
}

// ── Rung 9: the answer cache keeps serving the old answer → version-aware cache ──
export function rung9() {
  const Q = 'how often are laptops refreshed'
  function run(versionAware) {
    const st = indexAll([cloneDoc(HANDBOOK), ...KB], { version: 1 })
    const cache = new Map(), log = []
    const ask = () => {
      const key = Q.toLowerCase().trim(), hit = cache.get(key)
      if (hit && (!versionAware || Object.entries(hit.sources).every(([d, v]) => st.active.get(d) === v))) { log.push({ step: 'ask', cached: true, answer: hit.answer }); return hit.answer }
      if (hit) log.push({ step: 'invalidate', reason: 'source version changed' })
      const hs = st.search(Q, { k: 3, filter: x => x.docId === 'handbook' && isActive(st)(x) })
      const answer = refreshYears(hs.map(h => h.r.text))
      cache.set(key, { answer, sources: { handbook: hs[0].r.version } })
      log.push({ step: 'ask', cached: false, answer }); return answer
    }
    ask()
    // publish v2 with the new refresh cycle; index is updated correctly
    const d = cloneDoc(HANDBOOK); d.sections[4].paras[0] = d.sections[4].paras[0].replace('every 3 years', 'every 4 years')
    for (const c of stableChunks(d)) st.embedRow(c, 'handbook', { version: 2 })
    st.active.set('handbook', 2); st.deleteWhere(x => x.docId === 'handbook' && x.version === 1)
    log.push({ step: 'publish', version: 2 })
    const answer = ask()
    return { log, answer }
  }
  return { query: Q, truth: '4', without: run(false), with: run(true) }
}

// ── The ladder ────────────────────────────────────────────────────────────────
export const RUNGS = [
  { n: 1, key: 'docIds', problem: 'One edit re-embeds the whole corpus', fix: 'Store a document ID on every chunk', run: rung1 },
  { n: 2, key: 'hashing', problem: 'A one-sentence edit re-embeds the whole document', fix: 'Content-hash every chunk', run: rung2 },
  { n: 3, key: 'stable', problem: 'The edit shifts every later chunk boundary', fix: 'Stable, structure-anchored chunking', run: rung3 },
  { n: 4, key: 'versioning', problem: 'A crash leaves v12 and v13 mixed together', fix: 'Write each update as a new document version', run: rung4 },
  { n: 5, key: 'atomic', problem: 'Search during a deploy returns half a document', fix: 'Atomic version switch after every write lands', run: rung5 },
  { n: 6, key: 'gc', problem: 'Old versions pile up in the index', fix: 'Garbage-collect superseded versions', run: rung6 },
  { n: 7, key: 'embedVersion', problem: 'A model upgrade mixes two vector spaces', fix: 'Track the embedding model on every vector', run: rung7 },
  { n: 8, key: 'reconcile', problem: 'The update event never reaches the indexer', fix: 'Transactional outbox + reconciliation sweep', run: rung8 },
  { n: 9, key: 'cache', problem: 'The answer cache serves yesterday’s answer', fix: 'Key cached answers to source versions', run: rung9 },
]

let _cache = null
export function runAll() {
  if (_cache) return _cache
  const t0 = performance.now()
  const out = {}
  for (const r of RUNGS) out[r.n] = r.run()
  out.ms = Math.round(performance.now() - t0)
  return (_cache = out)
}

// ── Chaos week: a scripted week of production, graded against the fixes you enabled ──
export const REQUIRES = { atomic: ['versioning'], gc: ['versioning'], stable: ['hashing'], hashing: ['docIds'] }
export function chaosWeek(fixes) {
  const R = runAll(), f = k => !!fixes[k]
  const edits = 100
  const r1 = R[1], r3 = R[3]
  const perEdit = !f('docIds') ? r1.without.embeds : !f('hashing') ? r1.with.embeds : !f('stable') ? r3.without.embeds : r3.with.embeds
  const incidents = [
    { day: 'Mon', what: `${edits} document edits`, ok: perEdit <= r3.with.embeds * 2,
      impact: `${(perEdit * edits).toLocaleString()} embeddings for the week (${perEdit.toLocaleString()} per edit)`, rungs: [1, 2, 3] },
    { day: 'Tue', what: 'Indexer crashes mid-update', ok: f('versioning'),
      impact: f('versioning') ? 'Readers stayed on the last complete version' : `Index answers both “${R[4].without.facts.join(' days” and “')} days” of leave`, rungs: [4] },
    { day: 'Wed', what: 'User searches during a deploy', ok: f('versioning') && f('atomic'),
      impact: f('versioning') && f('atomic') ? '100% of the handbook searchable throughout' : `Only ${Math.round(R[5].without.coveragePct * 100)}% of the handbook was searchable`, rungs: [5] },
    { day: 'Thu', what: '300 edits accumulate', ok: !f('versioning') || f('gc'),
      impact: !f('versioning') ? 'No versions kept, nothing to collect' : f('gc') ? `Index stays at ${R[6].with.total.toLocaleString()} vectors` : `${Math.round(R[6].without.stalePct * 100)}% of ${R[6].without.total.toLocaleString()} vectors are stale`, rungs: [6] },
    { day: 'Fri', what: 'Embedding model upgraded', ok: f('embedVersion'),
      impact: f('embedVersion') ? `Recall@5 held at ${Math.round(R[7].with.recall * 100)}%` : `Recall@5 fell from ${Math.round(R[7].before * 100)}% to ${Math.round(R[7].without.recall * 100)}%`, rungs: [7] },
    { day: 'Sat', what: 'Two update events lost', ok: f('reconcile'),
      impact: f('reconcile') ? `Sweep found and repaired ${R[8].with.repaired.length} stale documents` : `${R[8].without.stale.length} documents silently stale; laptops “every ${R[8].without.answer} years”`, rungs: [8] },
    { day: 'Sun', what: 'Popular question asked again', ok: f('cache'),
      impact: f('cache') ? 'Cache entry invalidated; fresh answer served' : `Cache served the old answer (“every ${R[9].without.answer} years”)`, rungs: [9] },
  ]
  const passed = incidents.filter(i => i.ok).length
  return { incidents, passed, grade: ['F', 'F', 'E', 'D', 'C', 'C', 'B', 'A'][passed] }
}
