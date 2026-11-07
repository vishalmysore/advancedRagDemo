// production.js — UI for "Production Problems in RAG and How to Solve Them"
import { runAll, RUNGS, chaosWeek, REQUIRES } from './utils/prodLab.js'
import { KB } from './utils/prodCorpus.js'

const $ = id => document.getElementById(id)
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]))
const pct = v => Math.round(v * 100) + '%'
const num = v => v.toLocaleString()

const R = runAll()
const params = new URLSearchParams(location.search)
let current = Math.min(9, Math.max(1, +(params.get('rung') || 1)))

// ── copy for each rung ─────────────────────────────────────────────────────────
const STORY = {
  1: { story: 'Your index started as a loop: chunk every document, embed every chunk, insert. When one paragraph changes, the simplest reaction is to run the loop again. At a few thousand vectors nobody notices. At twenty million it is hours of GPU time and a re-index window for one edit.',
       badTitle: 'Rebuild the whole index', goodTitle: 'Store a document ID on every chunk',
       code: `// every vector carries its source document
upsert({ id, vector, metadata: { docId: 'handbook', ... } })

// on change: replace only that document's vectors
await index.delete({ filter: { docId } })
for (const c of chunk(latest))
  await upsert(embed(c), { docId })`,
       next: 'The handbook is long. Someone added one sentence, and you just re-embedded every chunk of it.' },
  2: { story: 'Deleting and re-embedding one document is better than rebuilding everything, but a long document is still hundreds of chunks. The natural next step is a content hash per chunk: re-chunk the new version, compare hashes, embed only what changed. Here the edit is a single sentence added to section 3. Watch what the hashes do with fixed 400-character windows.',
       badTitle: 'Re-embed the whole document', goodTitle: 'Hash every chunk, embed only new hashes',
       code: `const old = new Set(await index.hashesFor(docId))
for (const c of chunk(latest)) {
  const h = sha256(c.text)
  if (!old.has(h)) await upsert(embed(c.text), { docId, hash: h })
}
// then delete vectors whose hash is no longer present`,
       next: 'Hashing helped far less than expected: one inserted sentence moved every chunk boundary after it.' },
  3: { story: 'Fixed-size windows are positional. Insert one sentence early in the document and every later window starts that many characters further on, so every later hash is new even though the text barely changed. The fix is to cut along boundaries that survive edits: headings, sections and paragraphs. Then an edit inside one paragraph changes exactly one chunk.',
       badTitle: 'Fixed windows + hashing', goodTitle: 'Structure-anchored chunks + hashing',
       code: `// one chunk per paragraph, prefixed with its heading path
for (const s of doc.sections)
  for (const p of s.paragraphs)
    chunks.push({
      text: \`\${doc.title} › \${s.title}\\n\${p}\`,
      section: s.id })
// long paragraphs: split on sentences, never on offsets`,
       next: 'Now updates are small and fast. Then one crashes halfway through.' },
  4: { story: 'Version 13 of the handbook changes the leave allowance from 15 to 20 days and remote work from 2 to 3 days, in four paragraphs across four sections. The indexer updates in place and crashes after two of the four. Nothing errors at query time: the index simply holds half of each version, and retrieval happily returns both.',
       badTitle: 'Update vectors in place', goodTitle: 'Write v13 next to v12; read only the active version',
       code: `// writers never touch the live version
for (const c of chunks)
  await upsert(embed(c), { docId, version: 13, hash: c.hash })
await versions.set(docId, { active: 12, building: 13 })

// readers filter on the active version
search(q, { filter: { docId, version: versions.active(docId) } })`,
       next: 'Versioning protects you from crashes, as long as you do not activate the new version too early.' },
  5: { story: 'Most vector databases acknowledge an upsert before it is searchable: writes are queued, batched and indexed asynchronously. If the indexer marks v13 active as soon as it has sent the writes, a user who searches during the deploy reads v13 while only part of it has landed. Here the search runs when 45% of the queue has drained.',
       badTitle: 'Flip active as soon as writes are sent', goodTitle: 'Flip only after every write is confirmed',
       code: `await upsertAll(v13Chunks)                 // queued, not yet visible
await waitUntil(() =>
  index.count({ docId, version: 13 }) === v13Chunks.length)
await versions.activate(docId, 13)         // one atomic pointer write
// readers never see a partially built version`,
       next: 'Every edit now leaves a complete old version behind. Nobody deletes them.' },
  6: { story: 'With versioning, every update writes a full new version of the document. Without a clean-up job, superseded versions stay in the index forever: they cost storage, slow down filtered search and become candidates for every bug that forgets the version filter. The simulation applies 300 small edits to the handbook.',
       badTitle: 'Keep every version', goodTitle: 'Collect superseded versions after activation',
       code: `// after a successful switch, keep one version for rollback
await versions.activate(docId, v)
queue.add('gc', { docId, olderThan: v - 1 })   // async, idempotent

worker('gc', ({ docId, olderThan }) =>
  index.delete({ filter: { docId, version: { $lt: olderThan } } }))`,
       next: 'The index is lean and consistent. Then the team upgrades the embedding model.' },
  7: { story: 'A new embedding model is not a better version of the old one: it is a different vector space. Here the pipeline switches to embed-v2 while 25 edited documents get re-embedded and the rest keep their embed-v1 vectors. Queries are now embedded with v2, so they can only find the documents that happen to be in v2 space.',
       badTitle: 'Swap the model in place', goodTitle: 'Tag the model, re-index deliberately, cut over',
       code: `upsert(vec, { docId, version, model: 'embed-v2', dim: 1024 })

// serving is pinned to one model
const m = active.model
search(embedWith(m, q), { filter: { model: m } })

await reindexAll({ model: 'embed-v2' })  // shadow, dual writes
await setActiveModel('embed-v2')         // cut over when done`,
       next: 'Every layer is correct now, as long as it hears about every change.' },
  8: { story: 'Updates reach the indexer as events. Events get lost: a broker restarts, a consumer throws, a deploy drops a queue. In this run 2 of 40 update events never arrive, and one of them is the change of the laptop refresh cycle from 3 to 4 years. Nothing fails. The index is just quietly wrong.',
       badTitle: 'Trust the event stream', goodTitle: 'Outbox on write + scheduled reconciliation',
       code: `// 1. outbox: event saved in the same transaction as the doc
await db.tx(async t => {
  await t.saveDoc(doc)
  await t.outbox.add({ docId: doc.id, hash: doc.hash })
})

// 2. nightly reconciliation: source hash vs indexed hash
for (const d of await db.docs())
  if (d.hash !== await index.sourceHash(d.id)) await reindex(d.id)`,
       next: 'The index is now correct. The answer your users see may still not be.' },
  9: { story: 'Popular questions are cached. The cache stores the generated answer, keyed by the question. The handbook is updated, the index picks it up correctly, and the cache keeps serving the old answer until it expires. The retrieval layer is right; the user still gets yesterday’s policy.',
       badTitle: 'Cache by question only', goodTitle: 'Store source versions with every cached answer',
       code: `cache.set(key, { answer, sources: { handbook: 13, 'kb-12-faq': 4 } })

const hit = cache.get(key)
const fresh = hit && Object.entries(hit.sources)
  .every(([doc, v]) => versions.active(doc) === v)
if (fresh) return hit.answer
// otherwise: invalidate, retrieve, regenerate`,
       next: '' },
}

// ── hero ────────────────────────────────────────────────────────────────────────
function hero() {
  const chunks = 60 + KB.length * 4
  $('heroStats').innerHTML = [
    ['Documents', num(KB.length + 1)], ['Chunks', num(chunks)], ['Failures replayed', '9'], ['Runtime', `${R.ms} ms`],
  ].map(([k, v]) => `<div class="stat"><div class="stat-v">${v}</div><div class="stat-k">${k}</div></div>`).join('')
}

// ── rail ────────────────────────────────────────────────────────────────────────
function rail() {
  $('rail').innerHTML = RUNGS.map(r => `<button class="rail-item ${r.n === current ? 'on' : ''}" data-n="${r.n}">
      <span class="rail-n">${r.n}</span>
      <span class="rail-txt"><span class="rail-p">${esc(r.problem)}</span><span class="rail-f">→ ${esc(r.fix)}</span></span>
    </button>`).join('')
  $('rail').querySelectorAll('.rail-item').forEach(b => b.onclick = () => go(+b.dataset.n))
}
function go(n) {
  current = n; rail(); renderRung()
  const u = new URL(location); u.searchParams.set('rung', n); u.searchParams.delete('tab'); history.replaceState(null, '', u)
}

// ── shared widgets ─────────────────────────────────────────────────────────────
const tile = (v, k, cls = '') => `<div class="tile ${cls}"><div class="tile-v">${v}</div><div class="tile-k">${k}</div></div>`
const verdict = (ok, text) => `<div class="verdict ${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✗'} ${text}</div>`
function hitsList(hits, { badge = h => h.tag ? `<span class="vb ${h.tag}">${h.tag}</span>` : '', mark } = {}) {
  return `<ol class="hits">${hits.map(h => {
    let t = esc(h.text.split('\n').slice(1).join(' ') || h.text)
    if (mark) t = t.replace(mark, m => `<mark>${m}</mark>`)
    const where = h.docId === 'handbook' ? `Handbook §${h.section}` : esc(h.docId)
    return `<li><div class="hit-meta">${badge(h)}<span class="hit-where">${where}</span><span class="hit-score">${h.score.toFixed(2)}</span></div><div class="hit-text">${t}</div></li>`
  }).join('')}</ol>`
}
function strip(chunks, { stable = false } = {}) {
  return `<div class="strip ${stable ? 'stable' : ''}">${chunks.map((c, i) => `<span class="cell ${c.reused ? 'reused' : 'changed'}" title="chunk ${i} · §${c.section} · ${c.reused ? 'hash unchanged: reused' : 'new hash: re-embedded'}"></span>`).join('')}</div>
    <div class="strip-legend"><span><i class="cell changed"></i>re-embedded</span><span><i class="cell reused"></i>hash unchanged, reused</span></div>`
}

// ── per-rung renderers: return [badHTML, goodHTML, wideHTML?] ───────────────────
const RENDER = {
  1() {
    const r = R[1], max = r.without.embeds
    const bar = (v, cls) => `<div class="hbar"><div class="hbar-fill ${cls}" style="width:${Math.max(1.5, v / max * 100)}%"></div></div>`
    return [
      `<div class="tiles">${tile(num(r.without.embeds), 'embeddings for one edit', 'bad')}${tile(num(r.docs), 'documents touched')}</div>${bar(r.without.embeds, 'bad')}
       <p class="note">Projected at 20M vectors and 2,000 embeddings/s: <b>${r.without.projected}</b> of embedding per edit.</p>
       ${verdict(false, 'One edit re-embeds the entire corpus')}`,
      `<div class="tiles">${tile(num(r.with.embeds), 'embeddings for one edit', 'good')}${tile('1', 'document touched')}</div>${bar(r.with.embeds, 'good')}
       <p class="note">Cost now scales with the size of <b>one document</b>, not the corpus: ${r.with.projected} even at 20M vectors.</p>
       ${verdict(true, `${pct(1 - r.with.embeds / r.without.embeds)} fewer embeddings`)}`,
    ]
  },
  2() {
    const r = R[2]
    return [
      `<div class="tiles">${tile(r.without.embeds, 'chunks re-embedded', 'bad')}${tile(r.chunks, 'chunks in handbook')}</div>${strip(r.without.strip)}
       ${verdict(false, 'Every chunk of the document is embedded again')}`,
      `<div class="tiles">${tile(r.with.embeds, 'chunks re-embedded', 'warn')}${tile(r.with.firstChange, 'chunks before the edit reused')}</div>${strip(r.with.strip)}
       <p class="note">The sentence was inserted in chunk ${r.with.firstChange}. Every window after it shifted, so <b>${r.with.cascade} of ${r.chunks}</b> hashes changed.</p>
       ${verdict(false, `Hashing saved only ${r.chunks - r.with.embeds} embeddings`)}`,
    ]
  },
  3() {
    const r = R[3]
    return [
      `<div class="tiles">${tile(r.without.embeds, 'chunks re-embedded', 'bad')}${tile(r.without.chunks, 'chunks')}</div>${strip(r.without.strip)}
       <p class="note">${r.without.label}: a positional cut means one insertion moves every later boundary.</p>${verdict(false, 'The edit cascades to the end of the document')}`,
      `<div class="tiles">${tile(r.with.embeds, 'chunk re-embedded', 'good')}${tile(r.with.chunks, 'chunks')}</div>${strip(r.with.strip, { stable: true })}
       <p class="note">${r.with.label}: more, smaller chunks, but a local edit stays local.</p>${verdict(true, `${r.with.chunks - r.with.embeds} of ${r.with.chunks} chunks untouched`)}`,
    ]
  },
  4() {
    const r = R[4], mark = /\b(15|20) days\b/g
    const facts = f => f.map(x => `<span class="fact">${x} days</span>`).join(' vs ')
    return [
      `<div class="tiles">${tile(`${r.without.applied}/${r.without.of}`, 'paragraph updates applied before crash', 'bad')}${tile(r.without.facts.length, 'answers to one question', 'bad')}</div>
       <div class="q">“${esc(r.query)}”</div>${hitsList(r.without.hits, { mark })}
       ${verdict(false, `Index contradicts itself: ${facts(r.without.facts)}`)}`,
      `<div class="tiles">${tile(`${r.with.written}/${r.with.of}`, 'v13 chunks written before crash', 'warn')}${tile('v12', 'active version (unchanged)', 'good')}</div>
       <div class="q">“${esc(r.query)}”</div>${hitsList(r.with.hits, { mark })}
       ${verdict(true, `One consistent answer: ${facts(r.with.facts)}. The v13 build resumes on retry.`)}`,
    ]
  },
  5() {
    const r = R[5]
    const grid = c => {
      const by = {}; c.coverage.forEach(x => (by[x.section] ||= []).push(x))
      return `<div class="cov">${Object.entries(by).map(([s, xs]) => `<div class="cov-row"><span class="cov-s">§${s}</span>${xs.map(x => `<span class="cov-c ${x.present ? 'on' : 'off'}"></span>`).join('')}</div>`).join('')}</div>`
    }
    const badge = h => `<span class="vb ${h.docId === 'handbook' ? 'v' + h.version : 'kb'}">${h.docId === 'handbook' ? 'v' + h.version : 'KB'}</span>`
    return [
      `<div class="tiles">${tile('v' + r.without.activeVersion, 'active version during deploy', 'bad')}${tile(`${r.without.landed}/${r.without.of}`, 'v13 writes visible')}${tile(pct(r.without.coveragePct), 'of handbook searchable', 'bad')}</div>
       ${grid(r.without)}<div class="q">“${esc(r.query)}”</div>${hitsList(r.without.hits, { badge })}
       ${verdict(false, 'The remote-work paragraphs have not landed; search answers with unrelated text')}`,
      `<div class="tiles">${tile('v' + r.with.activeVersion, 'active version during deploy', 'good')}${tile(`${r.with.landed}/${r.with.of}`, 'v13 writes visible')}${tile(pct(r.with.coveragePct), 'of handbook searchable', 'good')}</div>
       ${grid(r.with)}<div class="q">“${esc(r.query)}”</div>${hitsList(r.with.hits, { badge })}
       ${verdict(true, 'Readers stay on complete v12 until all 60 v13 writes are confirmed')}`,
    ]
  },
  6() {
    const r = R[6]
    const W = 1100, H = 240, pl = 60, pr = 20, pt = 14, pb = 34
    const maxY = Math.max(...r.without.series.map(p => p[1])), maxX = r.edits
    const X = x => pl + x / maxX * (W - pl - pr), Y = y => pt + (1 - y / maxY) * (H - pt - pb)
    let g = ''
    for (let i = 0; i <= 4; i++) { const v = maxY * i / 4; g += `<line class="grid" x1="${pl}" x2="${W - pr}" y1="${Y(v)}" y2="${Y(v)}"/><text class="axis" x="${pl - 8}" y="${Y(v) + 4}" text-anchor="end">${num(Math.round(v / 1000) * 1000)}</text>` }
    for (let e = 0; e <= maxX; e += 50) g += `<text class="axis" x="${X(e)}" y="${H - pb + 16}" text-anchor="middle">${e}</text>`
    g += `<text class="axis" x="${(pl + W - pr) / 2}" y="${H - 2}" text-anchor="middle">edits to the handbook</text>`
    const line = (s, c) => `<polyline fill="none" stroke="${c}" stroke-width="2.4" points="${s.map(p => `${X(p[0])},${Y(p[1])}`).join(' ')}"/>`
    g += line(r.without.series, '#f85149') + line(r.with.series, '#3fb950')
    g += `<text class="lbl" fill="#f85149" x="${X(maxX) - 10}" y="${Y(r.without.total) + 22}" text-anchor="end">no GC · ${num(r.without.total)} vectors</text>`
    g += `<text class="lbl" fill="#3fb950" x="${X(maxX) - 4}" y="${Y(r.with.total) - 8}" text-anchor="end">with GC · ${num(r.with.total)} vectors</text>`
    return [
      `<div class="tiles">${tile(num(r.without.total), 'vectors after 300 edits', 'bad')}${tile(pct(r.without.stalePct), 'of the index is stale', 'bad')}</div>
       ${verdict(false, `Every filtered search scans ${Math.round(r.without.total / r.with.total)}× more rows than it needs`)}`,
      `<div class="tiles">${tile(num(r.with.total), 'vectors after 300 edits', 'good')}${tile(pct(r.with.stalePct), 'stale (one rollback version kept)', 'good')}</div>
       ${verdict(true, 'Index size tracks the corpus, not its edit history')}`,
      `<div class="panel-sub">Total vectors in the index while the handbook is edited 300 times</div><svg class="chart" viewBox="0 0 ${W} ${H}">${g}</svg>`,
    ]
  },
  7() {
    const r = R[7]
    const bars = rows => `<div class="rbars">${rows.map(([k, v, cls]) => `<div class="rbar"><span class="rbar-k">${k}</span><div class="hbar"><div class="hbar-fill ${cls}" style="width:${v * 100}%"></div></div><span class="rbar-v">${pct(v)}</span></div>`).join('')}</div>`
    const mix = m => { const t = m.v1 + m.v2; return `<div class="mix"><div class="mix-a" style="width:${m.v1 / t * 100}%">${m.v1 ? 'embed-v1 · ' + num(m.v1) : ''}</div><div class="mix-b" style="width:${m.v2 / t * 100}%">${m.v2 ? (m.v2 / t > .3 ? 'embed-v2 · ' : 'v2 · ') + num(m.v2) : ''}</div></div>` }
    return [
      `${bars([['Before upgrade', r.before, 'good'], ['After upgrade, v2 queries', r.without.recall, 'bad'], ['After upgrade, v1 queries', r.without.recallOldQueries, 'warn']])}
       <div class="panel-sub">Vectors by model after ${r.editedDocs} documents were re-embedded</div>${mix(r.without.mix)}
       ${verdict(false, `Recall@5 drops from ${pct(r.before)} to ${pct(r.without.recall)}; no error is raised`)}`,
      `${bars([['Before upgrade', r.before, 'good'], ['During migration (pinned to v1)', r.with.recallDuring, 'good'], ['After cut-over to v2', r.with.recall, 'good']])}
       <div class="panel-sub">Vectors by model after the deliberate re-index and cut-over</div>${mix(r.with.mix)}
       ${verdict(true, `Recall@5 stays at ${pct(r.with.recall)}; ${num(r.with.reembedded)} vectors re-embedded once, on purpose`)}`,
    ]
  },
  8() {
    const r = R[8]
    const tl = (x, rep) => `<div class="events">${x.log.map(e => `<span class="ev ${e.dropped ? (rep.includes(e.docId) ? 'fixed' : 'lost') : ''}" title="event ${e.e} · ${e.docId}${e.dropped ? ' · dropped' : ''}"></span>`).join('')}</div>
      <div class="strip-legend"><span><i class="ev"></i>delivered</span><span><i class="ev lost"></i>lost</span><span><i class="ev fixed"></i>lost, then repaired by sweep</span></div>`
    const mark = /every \d years/g
    return [
      `<div class="tiles">${tile('2', 'events lost', 'bad')}${tile(r.without.stale.length, 'documents silently stale', 'bad')}</div>${tl(r.without, [])}
       <div class="q">“${esc(r.query)}”</div>${hitsList(r.without.hits.slice(0, 1), { mark })}
       ${verdict(false, `Answers “every ${r.without.answer} years”; the source says ${r.truth}`)}`,
      `<div class="tiles">${tile(r.with.drift.length, 'drifted documents found', 'warn')}${tile(r.with.stale.length, 'stale after sweep', 'good')}</div>${tl(r.with, r.with.repaired)}
       <div class="q">“${esc(r.query)}”</div>${hitsList(r.with.hits.slice(0, 1), { mark })}
       ${verdict(true, `Reconciliation compared ${num(KB.length + 1)} source hashes with the index and repaired ${r.with.repaired.length}`)}`,
    ]
  },
  9() {
    const r = R[9]
    const log = x => `<ol class="steps">${x.log.map(s => s.step === 'publish' ? `<li class="pub">📄 Handbook v${s.version} published: laptops every 4 years. Index updated.</li>`
      : s.step === 'invalidate' ? `<li class="inv">🧹 Cache entry invalidated: ${s.reason}</li>`
      : `<li class="${s.cached ? 'cached' : 'fresh'}">${s.cached ? '⚡ cache hit' : '🔎 retrieved + generated'} → “every ${s.answer} years”</li>`).join('')}</ol>`
    return [
      `<div class="q">“${esc(r.query)}” asked before and after an update</div>${log(r.without)}${verdict(false, `Serves “every ${r.without.answer} years” after the policy changed to ${r.truth}`)}`,
      `<div class="q">“${esc(r.query)}” asked before and after an update</div>${log(r.with)}${verdict(true, `Serves “every ${r.with.answer} years”: the cached sources no longer matched the active version`)}`,
    ]
  },
}

function renderRung() {
  const meta = RUNGS[current - 1], s = STORY[current]
  const [bad, good, wide] = RENDER[current]()
  $('rung').innerHTML = `
    <div class="rung-head">
      <div class="rung-kicker">Problem ${current} of 9</div>
      <h2>${esc(meta.problem)}</h2>
      <p class="story">${esc(s.story)}</p>
    </div>
    <div class="vs">
      <div class="side bad"><div class="side-h"><span class="pill bad">Without</span>${esc(s.badTitle)}</div><div class="side-b">${bad}</div></div>
      <div class="side good"><div class="side-h"><span class="pill good">With the fix</span>${esc(s.goodTitle)}</div><div class="side-b">${good}</div></div>
    </div>
    ${wide ? `<div class="wide">${wide}</div>` : ''}
    <div class="fix-row">
      <div class="fix-code"><div class="fix-h">The fix: ${esc(meta.fix)}</div><pre><code>${esc(s.code)}</code></pre></div>
      ${s.next ? `<button class="next" id="btnNext"><span class="next-k">But then…</span><span class="next-t">${esc(s.next)}</span><span class="next-a">Problem ${current + 1} →</span></button>`
               : `<div class="next done"><span class="next-k">That’s the ladder</span><span class="next-t">Nine problems, nine fixes. Try them together in Chaos week.</span><button class="btn btn-test" id="btnChaos">🔥 Run chaos week</button></div>`}
    </div>`
  $('btnNext')?.addEventListener('click', () => go(current + 1))
  $('btnChaos')?.addEventListener('click', () => tab('chaos'))
}

// ── chaos week ────────────────────────────────────────────────────────────────
const fixes = Object.fromEntries(RUNGS.map(r => [r.key, false]))
if (params.has('fixes')) for (const k of params.get('fixes').split(',')) if (k in fixes) fixes[k] = true
function setFix(k, v) {
  fixes[k] = v
  if (v) for (const d of REQUIRES[k] || []) setFix(d, true)
  if (!v) for (const [dep, reqs] of Object.entries(REQUIRES)) if (reqs.includes(k) && fixes[dep]) setFix(dep, false)
}
function renderChaos() {
  $('toggles').innerHTML = `<div class="presets"><button class="btn btn-ghost" data-p="none">None</button><button class="btn btn-ghost" data-p="all">All nine</button></div>` +
    RUNGS.map(r => `<label class="tog ${fixes[r.key] ? 'on' : ''}"><input type="checkbox" data-k="${r.key}" ${fixes[r.key] ? 'checked' : ''}/><span class="tog-n">${r.n}</span><span class="tog-t">${esc(r.fix)}${REQUIRES[r.key] ? `<span class="req">needs ${REQUIRES[r.key].map(k => RUNGS.find(x => x.key === k).n).join(', ')}</span>` : ''}</span></label>`).join('')
  $('toggles').querySelectorAll('input').forEach(i => i.onchange = () => { setFix(i.dataset.k, i.checked); renderChaos() })
  $('toggles').querySelectorAll('[data-p]').forEach(b => b.onclick = () => { for (const k in fixes) fixes[k] = b.dataset.p === 'all'; renderChaos() })
  const w = chaosWeek(fixes)
  $('report').innerHTML = `<div class="grade g-${w.grade}"><div class="grade-l">${w.grade}</div><div class="grade-t"><b>${w.passed} of 7</b> incidents handled<br><span class="muted">${w.passed === 7 ? 'Production-ready.' : w.passed >= 4 ? 'Works on a good day.' : 'Works in the demo.'}</span></div></div>
    <table class="inc"><tbody>${w.incidents.map(i => `<tr class="${i.ok ? 'ok' : 'bad'}"><td class="d">${i.day}</td><td class="s">${i.ok ? '✓' : '✗'}</td><td><div class="w">${esc(i.what)}</div><div class="i">${esc(i.impact)}</div></td><td class="rg">${i.rungs.map(n => `<a href="?rung=${n}" data-n="${n}">#${n}</a>`).join(' ')}</td></tr>`).join('')}</tbody></table>`
  $('report').querySelectorAll('a[data-n]').forEach(a => a.onclick = e => { e.preventDefault(); tab('ladder'); go(+a.dataset.n) })
  const u = new URL(location); u.searchParams.set('tab', 'chaos'); u.searchParams.set('fixes', Object.keys(fixes).filter(k => fixes[k]).join(',')); u.searchParams.delete('rung')
  if (!$('tabChaos').classList.contains('hidden')) history.replaceState(null, '', u)
}

function tab(name) {
  document.querySelectorAll('#tabbar button').forEach(b => b.classList.toggle('on', b.dataset.tab === name))
  $('tabLadder').classList.toggle('hidden', name !== 'ladder')
  $('tabChaos').classList.toggle('hidden', name !== 'chaos')
  if (name === 'chaos') renderChaos(); else go(current)
}
document.querySelectorAll('#tabbar button').forEach(b => b.onclick = () => tab(b.dataset.tab))

hero(); rail(); renderRung(); renderChaos()
if (params.get('tab') === 'chaos') tab('chaos')
