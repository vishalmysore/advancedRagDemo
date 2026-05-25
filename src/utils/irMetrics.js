// irMetrics.js — Information Retrieval metrics for RAG evaluation
// Computes Precision@K, Recall@K, MRR, NDCG@K, AP, and P-R curve
// All functions take a binary relevance array sorted by rank position:
//   relevance[0] = is rank-1 chunk relevant? (true/false)
//   relevance[1] = is rank-2 chunk relevant? ...

/**
 * Precision@K — of the top K chunks, what fraction are relevant?
 */
export function precisionAtK(relevance, k) {
  if (k === 0) return 0
  const top = relevance.slice(0, k)
  return top.filter(Boolean).length / k
}

/**
 * Recall@K — of all relevant chunks in the retrieved set, what
 * fraction appear in the top K?
 * totalRelevant = relevance.filter(Boolean).length when all chunks are labeled.
 */
export function recallAtK(relevance, k) {
  const totalRelevant = relevance.filter(Boolean).length
  if (totalRelevant === 0) return 0
  const hitsInTopK = relevance.slice(0, k).filter(Boolean).length
  return hitsInTopK / totalRelevant
}

/**
 * Mean Reciprocal Rank — reciprocal of the rank of the first relevant chunk.
 * Perfect = 1.0 (first chunk is relevant). 0 = no relevant chunks found.
 */
export function mrr(relevance) {
  for (let i = 0; i < relevance.length; i++) {
    if (relevance[i]) return 1 / (i + 1)
  }
  return 0
}

/**
 * NDCG@K — Normalized Discounted Cumulative Gain at cutoff K.
 * Rewards relevant chunks appearing earlier in the ranking.
 * 1.0 = perfect ranking. 0.0 = no relevant chunks in top K.
 */
export function ndcgAtK(relevance, k) {
  const top = relevance.slice(0, k)

  // DCG: sum of rel_i / log2(i+2)
  let dcg = 0
  for (let i = 0; i < top.length; i++) {
    if (top[i]) dcg += 1 / Math.log2(i + 2)
  }

  // Ideal DCG: assume all relevant chunks ranked first
  const totalRelevant = relevance.filter(Boolean).length
  const idealK = Math.min(totalRelevant, k)
  let idcg = 0
  for (let i = 0; i < idealK; i++) {
    idcg += 1 / Math.log2(i + 2)
  }

  return idcg === 0 ? 0 : dcg / idcg
}

/**
 * Average Precision — area under the P-R curve (approximated as
 * precision at each rank where a relevant chunk is found, averaged).
 */
export function averagePrecision(relevance) {
  const totalRelevant = relevance.filter(Boolean).length
  if (totalRelevant === 0) return 0
  let ap = 0
  let hits = 0
  for (let i = 0; i < relevance.length; i++) {
    if (relevance[i]) {
      hits++
      ap += hits / (i + 1)
    }
  }
  return ap / totalRelevant
}

/**
 * Generate P-R curve points — one point per rank position
 * where a relevant chunk occurs.
 * Returns array of { precision, recall, rank }.
 */
export function precisionRecallCurve(relevance) {
  const totalRelevant = relevance.filter(Boolean).length
  if (totalRelevant === 0) return []
  const points = [{ precision: 1, recall: 0, rank: 0 }] // origin
  let hits = 0
  for (let i = 0; i < relevance.length; i++) {
    if (relevance[i]) {
      hits++
      points.push({
        precision: hits / (i + 1),
        recall: hits / totalRelevant,
        rank: i + 1
      })
    }
  }
  // Extend to recall=1 if all relevant found
  if (points[points.length - 1].recall < 1 && totalRelevant > 0) {
    points.push({ precision: 0, recall: 1, rank: relevance.length + 1 })
  }
  return points
}

/**
 * Render a Precision-Recall SVG curve into a container element.
 * @param {HTMLElement} container
 * @param {Array<{precision, recall, rank}>} points
 * @param {number} apScore — AUC-PR to annotate
 */
export function renderPRCurve(container, points, apScore) {
  const W = container.clientWidth || 280
  const H = 200
  const PAD = { top: 16, right: 16, bottom: 36, left: 40 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const toX = r => PAD.left + r * innerW
  const toY = p => PAD.top + (1 - p) * innerH

  // Build polyline path
  const pathD = points.map((pt, i) =>
    `${i === 0 ? 'M' : 'L'}${toX(pt.recall).toFixed(1)},${toY(pt.precision).toFixed(1)}`
  ).join(' ')

  // Fill under curve
  const fillD = pathD +
    ` L${toX(points[points.length - 1]?.recall || 0).toFixed(1)},${toY(0).toFixed(1)}` +
    ` L${toX(0).toFixed(1)},${toY(0).toFixed(1)} Z`

  // Grid lines
  const gridLines = [0, 0.25, 0.5, 0.75, 1].map(v => `
    <line x1="${toX(v)}" y1="${PAD.top}" x2="${toX(v)}" y2="${PAD.top + innerH}"
          stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
    <line x1="${PAD.left}" y1="${toY(v)}" x2="${PAD.left + innerW}" y2="${toY(v)}"
          stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
    <text x="${toX(v)}" y="${PAD.top + innerH + 14}" text-anchor="middle"
          font-size="9" fill="#6b7892">${v.toFixed(2)}</text>
    <text x="${PAD.left - 6}" y="${toY(v) + 3}" text-anchor="end"
          font-size="9" fill="#6b7892">${v.toFixed(2)}</text>
  `).join('')

  // Data point dots
  const dots = points.filter(pt => pt.rank > 0).map(pt => `
    <circle cx="${toX(pt.recall)}" cy="${toY(pt.precision)}" r="3.5"
            fill="#7c3aed" stroke="#a78bfa" stroke-width="1.2">
      <title>Rank ${pt.rank} — P: ${(pt.precision * 100).toFixed(1)}%  R: ${(pt.recall * 100).toFixed(1)}%</title>
    </circle>
  `).join('')

  // Random baseline (y = 1 - x diagonal in P-R space is not meaningful;
  // baseline for P-R is precision = totalRelevant/N, a horizontal line)
  const baselineY = toY(points.length > 1 ? (points.filter(p => p.rank > 0).length / (points[points.length-1]?.rank || 1)) : 0.5)

  container.innerHTML = `
    <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
      <!-- Grid -->
      ${gridLines}
      <!-- Axes -->
      <line x1="${PAD.left}" y1="${PAD.top}" x2="${PAD.left}" y2="${PAD.top + innerH}"
            stroke="#3a4558" stroke-width="1.5"/>
      <line x1="${PAD.left}" y1="${PAD.top + innerH}" x2="${PAD.left + innerW}" y2="${PAD.top + innerH}"
            stroke="#3a4558" stroke-width="1.5"/>
      <!-- Axis labels -->
      <text x="${PAD.left + innerW / 2}" y="${H - 2}" text-anchor="middle"
            font-size="10" fill="#8896b3">Recall</text>
      <text x="10" y="${PAD.top + innerH / 2}" text-anchor="middle"
            font-size="10" fill="#8896b3"
            transform="rotate(-90, 10, ${PAD.top + innerH / 2})">Precision</text>
      <!-- Fill under curve -->
      ${points.length > 1 ? `<path d="${fillD}" fill="rgba(124,58,237,0.12)"/>` : ''}
      <!-- Curve line -->
      ${points.length > 1 ? `<path d="${pathD}" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linejoin="round"/>` : ''}
      <!-- Data points -->
      ${dots}
      <!-- AUC-PR annotation -->
      <text x="${PAD.left + innerW - 4}" y="${PAD.top + 12}" text-anchor="end"
            font-size="10" fill="#a78bfa" font-weight="600">
        AP = ${(apScore * 100).toFixed(1)}%
      </text>
      <!-- No data message -->
      ${points.length <= 1 ? `
        <text x="${W / 2}" y="${H / 2}" text-anchor="middle"
              font-size="11" fill="#4a5568">Label chunks as relevant to plot curve</text>
      ` : ''}
    </svg>
  `
}
