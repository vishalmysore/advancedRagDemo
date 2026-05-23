// pdfParser.js — Client-side PDF and text file parser
// Utilises window.pdfjsLib loaded via CDN in index.html

/**
 * Initialise PDF.js worker if it hasn't been set up yet.
 */
function initPdfWorker() {
  if (typeof window !== 'undefined' && window.pdfjsLib && !window.pdfjsLib.GlobalWorkerOptions.workerSrc) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js'
  }
}

/**
 * Extracts text page-by-page from a PDF file.
 * @param {File} file - HTML5 File object
 * @param {Function} onProgress - Optional callback for page-by-page loading (pageIndex, totalPages)
 * @returns {Promise<Array<{ pageNumber: number, text: string }>>}
 */
export async function parsePdf(file, onProgress) {
  initPdfWorker()
  const pdfjsLib = window.pdfjsLib
  if (!pdfjsLib) {
    throw new Error('PDF.js library is not loaded. Please check your internet connection or script references.')
  }

  // Load file into ArrayBuffer
  const arrayBuffer = await file.arrayBuffer()
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer })
  
  const pdf = await loadingTask.promise
  const numPages = pdf.numPages
  const pages = []

  for (let i = 1; i <= numPages; i++) {
    const page = await pdf.getPage(i)
    const textContent = await page.getTextContent()
    
    // Group text items by their positional flow
    const text = textContent.items
      .map(item => item.str)
      .join(' ')
      .replace(/\s+/g, ' ') // normalize whitespace
      .trim()

    pages.push({
      pageNumber: i,
      text: text || `[Empty page ${i}]`
    })

    if (onProgress) {
      onProgress(i, numPages)
    }
  }

  return pages
}

/**
 * Reads text content from a plain text file.
 * @param {File} file - HTML5 File object
 * @returns {Promise<string>}
 */
export function parseTxt(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(reader.error)
    reader.readAsText(file)
  })
}
