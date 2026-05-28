import { defineConfig } from 'vite'

const securityHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

export default defineConfig(({ command }) => ({
  root: 'src',
  base: command === 'build' ? '/advancedRagDemo/' : '/',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'esnext',
  },
  worker: {
    format: 'es',
  },
  plugins: [
    {
      name: 'wasm-content-type',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.endsWith('.wasm')) res.setHeader('Content-Type', 'application/wasm')
          if (req.url?.endsWith('.mjs'))  res.setHeader('Content-Type', 'application/javascript')
          for (const [k, v] of Object.entries(securityHeaders)) res.setHeader(k, v)
          next()
        })
      },
    },
  ],
  server:  { headers: securityHeaders },
  preview: { headers: securityHeaders },
}))
