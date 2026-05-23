import { defineConfig } from 'vite'

export default defineConfig(({ command }) => ({
  root: 'src',
  base: command === 'build' ? '/advancedRagDemo/' : '/',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'esnext'
  }
}))
