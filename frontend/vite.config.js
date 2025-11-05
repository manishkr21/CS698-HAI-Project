import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH || '/CS698-HAI-Project/',
  server: {
    proxy: {
      '/student_app': {
        target: 'https://cdis.iitk.ac.in',
        changeOrigin: true,
        secure: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})