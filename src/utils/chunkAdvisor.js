// chunkAdvisor.js — Chunking Strategy Advisor (pure JS, no model downloads).
//
// Two halves:
//   • profileDocument(text): cheap structural signals (headings, numbered clauses,
//     cross-references, speaker turns, email headers, Q/A pairs, code, tables) and a
//     scored guess at the document category.
//   • recommend(category, constraints): a chunking recipe for that category, adjusted
//     for the context budget, the retriever and whether question–evidence pairs exist.
// The rules encode what the Chunking Lab measured plus the Sept–Oct 2026 papers it
// implements (StrucChunk 2610.26313, QESChunker 2610.36557, SpectRAG 2610.36991,
// Parser/Chunking interactions 2609.31660, adaptive top-k 2609.13489).

import { parseDocument, chunkFixed, chunkSentence, chunkStructure } from './chunkLab.js'

// ── Categories ────────────────────────────────────────────────────────────────

export const CATEGORIES = [
  {
    key: 'legal', icon: '⚖️', name: 'Legal, policy & regulations',
    examples: 'contracts, handbooks, compliance manuals, statutes, SOPs',
    strategy: 'struct', strategyName: 'Structure-aware (provision boundaries)',
    size: [250, 450], overlap: 0, headingPath: true, resolveRefs: true, retrieval: 'graph',
    why: [
      'Answers usually need a rule plus the definition or exception it points to. One chunk per clause, with "see Section x.y" resolved at index time, puts both in the same retrievable unit.',
      'In the lab this was the clear winner on a policy-shaped handbook: 62% complete-evidence recall vs 23% for fixed and sentence windows at a 120-word budget.',
      'Cross-reference edges are trustworthy, so a chunk graph helps here: definition clauses become hubs (62% → 69% in the lab).',
    ],
    pitfalls: [
      'Never cut between a clause heading and its first sentence; the heading is what keyword search matches.',
      'Long clauses still need splitting: split on sentences inside the clause and repeat the heading path on every piece.',
    ],
    papers: ['2610.26313', '2610.36991'],
  },
  {
    key: 'techdocs', icon: '🛠️', name: 'Technical docs & manuals',
    examples: 'API references, product manuals, runbooks, READMEs, wikis',
    strategy: 'struct', strategyName: 'Structure-aware (by heading)',
    size: [300, 600], overlap: 0, headingPath: true, resolveRefs: true, retrieval: 'flat',
    why: [
      'Docs are written as a heading tree and users ask about one section at a time, so heading boundaries match how questions are asked.',
      'Prefix every chunk with its heading path (e.g. "Auth › Tokens › Refresh"), because the same words ("timeout", "limit") appear in many sections.',
      'Resolve explicit links ("see Configuration") the same way as legal cross-references.',
    ],
    pitfalls: [
      'Keep fenced code blocks whole. A code sample split from its explanation is useless to both the retriever and the reader.',
      'Use lexical chunk-graph edges with care: docs reuse vocabulary heavily, and lexical edges spread relevance to unrelated sections.',
    ],
    papers: ['2610.26313', '2609.31660'],
  },
  {
    key: 'research', icon: '🔬', name: 'Research papers & reports',
    examples: 'papers, white papers, analyst reports, theses',
    strategy: 'struct', strategyName: 'Section-aware + sentence windows inside sections',
    size: [400, 700], overlap: 60, headingPath: true, resolveRefs: false, retrieval: 'flat',
    why: [
      'Sections (Abstract, Method, Results) are the natural units, but they are long, so split each section into sentence windows and keep the section name as a prefix.',
      'Questions span sections ("what dataset, and what score?"), so measure complete-evidence recall, not hit rate.',
      'The parser matters as much as the chunker for PDFs: column order, figure captions and tables change what is retrievable.',
    ],
    pitfalls: [
      'Keep figure and table captions with the paragraph that cites them ("as shown in Table 3").',
      'Strip reference lists and page headers before chunking; they pollute keyword retrieval.',
    ],
    papers: ['2609.31660', '2610.26951'],
  },
  {
    key: 'faq', icon: '❓', name: 'FAQ & knowledge base',
    examples: 'help-centre articles, FAQs, support macros, Q&A pages',
    strategy: 'record', strategyName: 'One Q&A pair per chunk',
    size: [150, 400], overlap: 0, headingPath: true, resolveRefs: false, retrieval: 'flat',
    why: [
      'Each entry already is the ideal chunk: a question and its full answer. Never split them.',
      'Small, self-contained chunks pack a small context budget well, which is where the lab saw the largest gaps between strategies.',
      'Keep the question text in the chunk: it matches how users phrase their query.',
    ],
    pitfalls: [
      'Answers that say "see the article on refunds" need that article linked or resolved, as with cross-references.',
      'Near-duplicate FAQs crowd the context; deduplicate at index time or filter low-novelty chunks at assembly time.',
    ],
    papers: ['2610.26313'],
  },
  {
    key: 'transcript', icon: '🎙️', name: 'Meeting & call transcripts',
    examples: 'meeting notes, call recordings, podcasts, interviews',
    strategy: 'qesov', strategyName: 'Speaker-turn windows with overlap (QES greedy if you have Q/E pairs)',
    size: [500, 900], overlap: 120, headingPath: false, resolveRefs: false, retrieval: 'flat',
    why: [
      'There are no headings to follow, so boundaries must come from turns and topic shifts. Pack whole speaker turns into windows with overlap.',
      'If you can generate question–evidence pairs with an LLM, optimise the boundaries against them. The overlapping QESChunker variant was the best learned method in the lab (46% held-out vs 23% for plain windows) without being told about structure.',
      'Facts are spread across turns ("as Priya said earlier…"), so overlap and a larger context budget pay off.',
    ],
    pitfalls: [
      'Prefix each chunk with the meeting title, date and speakers, or "what did we decide about X" has nothing to match.',
      'Evaluate the supervised chunker on different questions from the ones it trained on; the lab saw 83% train vs 46% held-out.',
    ],
    papers: ['2610.36557'],
  },
  {
    key: 'email', icon: '✉️', name: 'Email threads & tickets',
    examples: 'email, support tickets, chat threads, issue trackers',
    strategy: 'record', strategyName: 'One message per chunk, quoted text removed',
    size: [200, 600], overlap: 0, headingPath: true, resolveRefs: true, retrieval: 'graph',
    why: [
      'Each message is a natural unit. Strip quoted replies ("> …") first, or every chunk repeats the whole thread and retrieval returns duplicates.',
      'Prefix the subject, sender and date, so "what did the vendor confirm on Tuesday" can match.',
      'Reply-to links are explicit, trustworthy edges, so a chunk graph over the thread brings in the message being replied to.',
    ],
    pitfalls: [
      'Signatures and disclaimers repeat in every message and skew keyword scores; drop them.',
      'Long messages still need sentence-window splitting with the header repeated.',
    ],
    papers: ['2610.36991'],
  },
  {
    key: 'narrative', icon: '📖', name: 'Long-form prose',
    examples: 'books, articles, blog posts, essays, news',
    strategy: 'sentence', strategyName: 'Paragraph / sentence windows with overlap',
    size: [600, 1000], overlap: 100, headingPath: true, resolveRefs: false, retrieval: 'flat',
    why: [
      'Prose has weak structure and ideas run across paragraphs, so sentence or paragraph windows with overlap are the sensible default.',
      'With a generous budget, window chunkers catch up: in the lab they matched structure-aware chunking by 300 words of context and passed it by 400.',
      'Chapter and section titles are still worth prefixing when they exist.',
    ],
    pitfalls: [
      'Keep the chunk size well under the context budget. In the lab, 800-character windows scored 0% at a 120-word budget because no chunk fit.',
      'A chunk graph with lexical edges between big, mixed-topic windows tended to hurt (23% → 15% for sentence windows in the lab).',
    ],
    papers: ['2609.13489'],
  },
  {
    key: 'scanned', icon: '🖨️', name: 'Scanned PDFs & messy text',
    examples: 'OCR output, exported PDFs, slide decks, forms',
    strategy: 'qesov', strategyName: 'Sentence windows with overlap → QES greedy once you have Q/E pairs',
    size: [400, 700], overlap: 80, headingPath: false, resolveRefs: false, retrieval: 'flat',
    why: [
      'OCR breaks headings and line order, so structure-aware chunking has nothing reliable to follow. Start with sentence windows and overlap.',
      'Fix the parser first: a factorial study found parser and chunker choices interact and no one configuration wins everywhere.',
      'A supervised chunker learns good boundaries from question–evidence pairs without needing clean headings.',
    ],
    pitfalls: [
      'Hyphenated line breaks and page headers split words and sentences; clean them before chunking.',
      'If headings can be recovered (font size, numbering), switch to structure-aware chunking; it was the biggest win in the lab.',
    ],
    papers: ['2609.31660', '2610.36557'],
  },
  {
    key: 'tabular', icon: '📊', name: 'Tables & records',
    examples: 'CSV, spreadsheets, product catalogues, database exports',
    strategy: 'record', strategyName: 'Row groups with the header row repeated',
    size: [200, 500], overlap: 0, headingPath: true, resolveRefs: false, retrieval: 'flat',
    why: [
      'A row without its column names is meaningless to both the retriever and the model. Repeat the header row in every chunk, or serialise each row as "column: value" pairs.',
      'Group rows by a natural key (product, customer, month) so one chunk answers one entity question.',
      'Aggregation questions ("total sales in Q3") are a job for SQL, not retrieval; route them elsewhere.',
    ],
    pitfalls: [
      'Fixed character windows cut rows in half; always chunk on row boundaries.',
      'Wide tables can exceed the budget per row. Drop columns nobody asks about before indexing.',
    ],
    papers: ['2609.31660'],
  },
  {
    key: 'code', icon: '💻', name: 'Source code',
    examples: 'repositories, notebooks, config files, SQL',
    strategy: 'record', strategyName: 'One function / class per chunk (AST boundaries)',
    size: [400, 1200], overlap: 0, headingPath: true, resolveRefs: true, retrieval: 'graph',
    why: [
      'Functions and classes are the units people ask about. Split on syntax-tree boundaries, never in the middle of a function body.',
      'Prefix the file path, class and signature, so "where is the retry logic" matches on names.',
      'Imports and call sites are explicit edges, so a chunk graph can bring in the function being called.',
    ],
    pitfalls: [
      'Very long functions need splitting; repeat the signature on every piece.',
      'Keyword retrieval handles identifiers well but misses intent ("where do we validate emails"); pair it with dense retrieval.',
    ],
    papers: ['2610.36991'],
  },
]

export const STRATEGY_COLORS = { struct: '#d29922', sentence: '#58a6ff', qesov: '#bc8cff', record: '#3fb950', fixed: '#8b949e' }

// ── Profiling ─────────────────────────────────────────────────────────────────

const count = (text, re) => (text.match(re) || []).length

/** Measure structural signals and score every category. */
export function profileDocument(text) {
  const T = text.replace(/\r\n/g, '\n')
  const lines = T.split('\n')
  const nonEmpty = lines.filter(l => l.trim())
  const n = Math.max(1, nonEmpty.length)
  const sentences = T.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 2)
  const words = (T.match(/\S+/g) || []).length
  const s = {
    chars: T.length,
    words,
    lines: nonEmpty.length,
    mdHeadings: count(T, /^#{1,6}\s+\S/gm),
    numberedHeadings: count(T, /^\s*(?:#{1,6}\s+)?(?:§\s*)?\d+(?:\.\d+)+\.?\s+[A-Z]/gm),
    clauseWords: count(T, /\b(?:shall|must not|must|may not|hereinafter|pursuant|notwithstanding|provided that)\b/gi),
    crossRefs: count(T, /\b(?:see|under|per|in accordance with)\s+(?:Section|Clause|Article|§|Chapter|Appendix)\s*\d+/gi)
      + count(T, /\b(?:Section|Clause|Article)\s+\d+(?:\.\d+)+/g),
    speakerTurns: count(T, /^\s*(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*)?(?!(?:From|To|Cc|Bcc|Subject|Sent|Date|Re|Q|A|Question|Answer)\s*:)[A-Z][\w .'-]{1,30}:\s+\S/gm),
    timestamps: count(T, /\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?/g),
    emailHeaders: count(T, /^(?:From|To|Cc|Subject|Sent|Date):\s/gim),
    quoted: count(T, /^>/gm),
    qaPairs: count(T, /^\s*(?:Q[:.]|Question:|\*\*Q|#+\s.*\?\s*$)/gim) + count(T, /^[^\n]{8,140}\?\s*$/gm),
    codeFences: count(T, /^```/gm) / 2,
    codeLines: count(T, /^\s*(?:def |class |function |import |export |const |let |var |public |private |return\b|if \(|for \(|#include|SELECT |FROM )|[;{}]\s*$/gm),
    tableLines: count(T, /^\s*\|.*\|\s*$/gm) + count(T, /^[^,\n]+(?:,[^,\n]*){3,}$/gm) + count(T, /^[^\t\n]+(?:\t[^\t\n]*){2,}$/gm),
    ocrNoise: count(T, /\w-\n\w/g) + count(T, /^\s*(?:Page \d+|\d+\s*$)/gm) + count(T, /[^\x00-\x7F -ɏ‐-‧§›↳•]/g) / 5,
    avgSentenceWords: sentences.length ? words / sentences.length : 0,
    avgParagraphWords: words / Math.max(1, T.split(/\n\s*\n/).filter(p => p.trim()).length),
  }
  const r = (k) => s[k] / n // per non-empty line
  const headingRate = (s.mdHeadings + s.numberedHeadings) / n
  const score = {
    legal: 3 * Math.min(1, s.numberedHeadings / 4) + 2 * Math.min(1, s.crossRefs / 2) + 2 * Math.min(1, s.clauseWords / 6) + Math.min(1, headingRate * 4),
    techdocs: 2 * Math.min(1, s.mdHeadings / 4) + 2 * Math.min(1, s.codeFences / 1) + Math.min(1, r('codeLines') * 4) - Math.min(1, s.clauseWords / 8),
    research: 2 * (/\b(?:abstract|introduction|related work|method(?:s|ology)?|results|conclusion|references)\b/gi.test(T) ? Math.min(1, count(T, /^\W*(?:\d+\.?\s*)?(?:abstract|introduction|related work|method(?:s|ology)?|experiments?|results|discussion|conclusions?|references)\b/gim) / 3) : 0)
      + Math.min(1, count(T, /\[\d+(?:,\s*\d+)*\]|et al\./g) / 4) + Math.min(1, count(T, /\b(?:Table|Figure|Fig\.)\s*\d+/g) / 2),
    faq: 3 * Math.min(1, s.qaPairs / 4) + Math.min(1, s.qaPairs / n * 3) - 4 * Math.min(1, r('speakerTurns') * 2),
    transcript: 3 * Math.min(1, r('speakerTurns') * 2) + Math.min(1, s.timestamps / 6),
    email: 3 * Math.min(1, s.emailHeaders / 3) + Math.min(1, s.quoted / 4) + Math.min(1, count(T, /^(?:Hi|Hello|Dear|Thanks|Regards|Best|Cheers)\b/gim) / 2),
    narrative: Math.min(1, s.avgParagraphWords / 80) * 2 + Math.min(1, s.words / 600) - 2 * Math.min(1, headingRate * 6) - Math.min(1, r('speakerTurns') * 3),
    scanned: 2 * Math.min(1, s.ocrNoise / 6) + (headingRate === 0 && s.avgSentenceWords > 35 ? 1 : 0),
    tabular: 3 * Math.min(1, r('tableLines') * 1.5),
    code: 3 * Math.min(1, r('codeLines') * 2) - Math.min(1, s.clauseWords / 4),
  }
  const ranked = Object.entries(score).map(([key, v]) => ({ key, score: Math.max(0, v) })).sort((a, b) => b.score - a.score)
  const total = ranked.reduce((a, x) => a + x.score, 0) || 1
  ranked.forEach(x => { x.share = x.score / total })
  return { signals: s, headingRate, ranked, best: ranked[0].score > 0.4 ? ranked[0].key : 'narrative' }
}

// ── Recommendation ────────────────────────────────────────────────────────────

const BUDGET_WORDS = { small: 150, medium: 400, large: 1200 }

/**
 * Build the recipe for a category under the user's constraints.
 * constraints = { budget: 'small'|'medium'|'large', retriever: 'bm25'|'dense'|'hybrid', qe: boolean }
 */
export function recommend(categoryKey, { budget = 'medium', retriever = 'hybrid', qe = false } = {}, profile = null) {
  const cat = CATEGORIES.find(c => c.key === categoryKey) || CATEGORIES[0]
  let [lo, hi] = cat.size
  const notes = []
  let strategy = cat.strategy, strategyName = cat.strategyName

  // Budget: a chunk must fit several times into the context, or multi-fact answers cannot be assembled.
  const budgetWords = BUDGET_WORDS[budget]
  const maxChars = Math.round(budgetWords * 6 / 2.5) // ~6 chars/word, at least ~2.5 chunks per context
  if (hi > maxChars) {
    hi = Math.max(150, maxChars); lo = Math.min(lo, Math.round(hi * 0.6))
    notes.push(`Chunk size capped at ${hi} characters so about three chunks fit a ${budgetWords}-word context. In the lab, small chunks beat large ones by the widest margin when context was scarce, and chunks larger than the budget were never retrieved.`)
  }
  if (budget === 'large') notes.push('With a large budget, chunking choices matter less: in the lab every strategy reached 77–92% complete-evidence recall at 400 words. Spend effort on retrieval quality (reranking) instead.')

  // Supervised chunking needs Q/E pairs.
  if (strategy === 'qesov' && !qe) {
    strategy = 'sentence'
    strategyName = cat.key === 'transcript' ? 'Speaker-turn windows with overlap' : 'Sentence windows with overlap'
    notes.push('Without question–evidence pairs, fall back to overlapping windows. Generate 20–50 pairs with an LLM to unlock QESChunker-style boundary optimisation.')
  } else if (qe && (strategy === 'sentence')) {
    notes.push('You have question–evidence pairs: try the overlapping QESChunker variant against this baseline and keep whichever scores higher on held-out questions.')
  }

  // Retriever-specific advice.
  if (retriever === 'bm25') {
    notes.push('Keyword-only retrieval is sensitive to where headings land. Always keep a heading with the text under it, and prefix the heading path. In the lab, the exact QES DP cut right before each rule and stranded its heading, dropping to 15%.')
    if (qe) notes.push('If you use the non-overlapping QES DP, forbid cuts between a heading and its first sentence, or model BM25 in the objective.')
  } else if (retriever === 'dense') {
    notes.push('Dense-only retrieval misses exact identifiers (clause numbers, product codes, error codes). Add BM25 and fuse with Reciprocal Rank Fusion, as StrucChunk does.')
  } else {
    notes.push('Hybrid retrieval (BM25 + dense, fused with RRF) is the setting the StrucChunk and QESChunker papers evaluate. It also closes vocabulary gaps (e.g. "8-year-old" vs "under 10") that no chunker can fix.')
  }

  // Profile-driven tweaks.
  if (profile) {
    const s = profile.signals
    if (s.crossRefs >= 2 && !cat.resolveRefs) notes.push(`This document has ${s.crossRefs} explicit cross-references. Resolve them at index time by appending the referenced text to the chunk that cites it.`)
    if ((s.mdHeadings + s.numberedHeadings) >= 3 && strategy === 'sentence') notes.push(`This document has ${s.mdHeadings + s.numberedHeadings} headings. Consider structure-aware chunking even though the category default is windows; it was the strongest strategy in the lab whenever headings existed.`)
    if (s.quoted >= 3) notes.push(`${s.quoted} quoted lines found. Strip quoted replies before chunking to avoid duplicate chunks.`)
  }

  const overlap = Math.min(cat.overlap, Math.round(hi * 0.25))
  const graph = cat.retrieval === 'graph'
  return {
    cat, strategy, strategyName,
    params: [
      ['Chunk size', `${lo}–${hi} characters (≈${Math.round(lo / 4)}–${Math.round(hi / 4)} tokens)`],
      ['Overlap', overlap ? `${overlap} characters, whole sentences only` : 'none (boundaries follow structure)'],
      ['Heading path prefix', cat.headingPath ? 'yes' : 'speaker / title / date prefix instead'],
      ['Resolve cross-references', cat.resolveRefs ? 'yes, at index time' : 'only if the document has explicit ones'],
      ['Retrieval', graph ? 'chunk graph over explicit edges (references, replies, calls) + budgeted fill' : 'flat top-k + reranker, budgeted fill'],
      ['Context budget', `${budgetWords} words; tune per query (adaptive top-k)`],
      ['Evaluate with', 'complete-evidence recall on 20+ multi-fact held-out questions'],
    ],
    notes,
    labSize: Math.round((lo + hi) / 2),
    overlap,
    budgetWords,
  }
}

// ── Preview: run the recommended chunker on the user's text ───────────────────

/** Turn plain numbered headings ("2.3 Check-in") into markdown so the structure parser sees them. */
export function normaliseHeadings(text) {
  return text.replace(/\r\n/g, '\n').replace(/^(\s*)(\d+(?:\.\d+)+\.?\s+[A-Z][^\n]{0,80})$/gm, (m, sp, h) => '### ' + h.trim())
}

/** Split into records: Q&A pairs, emails, CSV rows or blank-line blocks. */
function chunkRecords(text, L) {
  const T = text.replace(/\r\n/g, '\n')
  let blocks
  if (/^(?:From|Subject):\s/im.test(T)) blocks = T.split(/\n(?=From:\s)/i).map(b => b.split('\n').filter(l => !/^>/.test(l)).join('\n'))
  else if (/^\s*\|.*\|\s*$/m.test(T) || /^[^,\n]+(?:,[^,\n]*){3,}$/m.test(T)) {
    const rows = T.split('\n').filter(l => l.trim())
    const header = rows[0]
    blocks = []
    let cur = []
    for (const row of rows.slice(1)) {
      if (cur.length && (header + '\n' + cur.concat(row).join('\n')).length > L) { blocks.push(header + '\n' + cur.join('\n')); cur = [] }
      cur.push(row)
    }
    if (cur.length) blocks.push(header + '\n' + cur.join('\n'))
  } else if (/^\s*(?:def |class |function |export |public |private )/m.test(T)) blocks = T.split(/\n(?=\s*(?:def |class |function |export |public |private |async function ))/)
  else blocks = T.split(/\n\s*\n(?=[^\n]*\?\s*$|Q[:.]|\*\*Q|#)/m)
  let offset = 0
  return blocks.filter(b => b.trim()).map(b => {
    const start = T.indexOf(b.slice(0, 40), offset); offset = Math.max(offset, start + 1)
    return { start, end: start + b.length, text: b.trim(), words: (b.match(/\S+/g) || []).length }
  })
}

export function previewChunks(text, strategy, L, overlap = Math.round(L * 0.15)) {
  const norm = normaliseHeadings(text)
  const doc = parseDocument(norm)
  let chunks, used = strategy
  if (strategy === 'struct' && doc.provisions.length >= 2) chunks = chunkStructure(doc, L)
  else if (strategy === 'record') chunks = chunkRecords(text, L)
  else if (strategy === 'fixed') chunks = chunkFixed(doc, L)
  else { chunks = chunkSentence(doc, L, overlap); used = strategy === 'struct' ? 'sentence' : strategy }
  const headingStarts = new Set(doc.units.filter(u => u.type === 'heading').map(u => u.start))
  const n = chunks.length
  return {
    used, chunks,
    count: n,
    avgChars: n ? Math.round(chunks.reduce((a, c) => a + c.text.length, 0) / n) : 0,
    maxWords: n ? Math.max(...chunks.map(c => c.words)) : 0,
    headingStartShare: n && used !== 'record' && headingStarts.size ? chunks.filter(c => headingStarts.has(c.start)).length / n : null,
    resolvedRefs: chunks.reduce((a, c) => a + (c.refs ? c.refs.length : 0), 0),
  }
}

// ── Built-in samples for the auto-detect demo ─────────────────────────────────

export const ADVISOR_SAMPLES = {
  transcript: `Weekly sync: Mobile release 4.2 (Oct 28)

[00:01] Priya: Okay, let's start with the release. Are we still on track for Thursday?
[00:09] Marcus: Mostly. The payment screen still crashes on Android 12 when the card scanner opens.
[00:18] Priya: Is that a blocker?
[00:21] Marcus: I'd say yes. It affects about four percent of sessions in the beta.
[00:30] Dana: We can ship with the scanner behind a flag and fall back to manual entry.
[00:38] Priya: Let's do that. Dana, can you own the flag?
[00:41] Dana: Yes, I'll have it in by Wednesday noon.
[00:47] Marcus: Then QA needs the build by Wednesday evening at the latest.
[01:02] Priya: Agreed. Second item: the onboarding experiment. Results?
[01:08] Dana: The short flow lifted day-one retention by two points, but only on iOS.
[01:15] Priya: Keep it running another week on Android before we decide.`,
  email: `From: Lena Ortiz
Subject: Re: Delivery window for order 88213
Date: Tue, Oct 21

Hi Sam,

Confirmed: the replacement pallets ship Thursday and arrive Monday before 10:00. The driver will call 30 minutes ahead.

Thanks,
Lena

> Can you confirm the new delivery window? We need it before Monday.

From: Sam Patel
Subject: Re: Delivery window for order 88213
Date: Wed, Oct 22

Thanks Lena. Our dock closes at 15:00 on Mondays, so morning works. Please send the bill of lading once it ships.

Best,
Sam`,
  faq: `How do I reset my password?
Open Settings, choose Security, then Reset password. A link is sent to your verified email and expires after 30 minutes.

Can I change my username?
Usernames can be changed once every 90 days from Settings › Profile. The old name is reserved for 14 days.

What happens if I cancel my plan?
Your workspace switches to read-only at the end of the billing period. Data is kept for 60 days, then deleted.

Do you offer refunds?
Annual plans are refundable within 14 days of purchase. Monthly plans are not refunded, but you can cancel at any time.`,
  contract: `1. Definitions

1.1 Confidential Information
Confidential Information means any non-public information disclosed by either party, in any form, that is marked confidential or would reasonably be understood to be confidential.

1.2 Term
The Term begins on the Effective Date and continues for twenty-four months unless terminated under Section 6.2.

2. Obligations

2.1 Non-disclosure
The Receiving Party shall not disclose Confidential Information to any third party except as permitted under Section 2.2.

2.2 Permitted disclosure
The Receiving Party may disclose Confidential Information to employees and advisers who need to know it, provided that they are bound by obligations no less protective than this Agreement.

6. Termination

6.2 Termination for breach
Either party may terminate this Agreement on thirty days' written notice if the other party materially breaches Section 2.1 and fails to cure the breach within that period.`,
  code: `export async function fetchWithRetry(url, { retries = 3, backoffMs = 250 } = {}) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const res = await fetch(url)
    if (res.ok) return res
    if (res.status < 500 || attempt === retries) throw new Error('Request failed: ' + res.status)
    await new Promise(r => setTimeout(r, backoffMs * 2 ** attempt))
  }
}

export function validateEmail(value) {
  const at = value.indexOf('@')
  return at > 0 && value.indexOf('.', at) > at + 1 && !/\\s/.test(value)
}

export class RateLimiter {
  constructor(perSecond) { this.perSecond = perSecond; this.tokens = perSecond; this.last = Date.now() }
  take() {
    const now = Date.now()
    this.tokens = Math.min(this.perSecond, this.tokens + (now - this.last) / 1000 * this.perSecond)
    this.last = now
    if (this.tokens < 1) return false
    this.tokens -= 1
    return true
  }
}`,
  table: `sku,product,category,price,stock,warehouse
A-1001,Trail running shoe,Footwear,129.00,42,Leeds
A-1002,Waterproof jacket,Outerwear,189.00,17,Leeds
A-1003,Merino base layer,Apparel,69.00,88,Bristol
A-1004,Trekking poles,Equipment,79.00,23,Bristol
A-1005,Headlamp 400lm,Equipment,45.00,61,Leeds
A-1006,Insulated bottle,Equipment,32.00,120,Bristol`,
}
