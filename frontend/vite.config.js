// import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'

// export default defineConfig({
//   plugins: [react()],
//   base: process.env.VITE_BASE_PATH || '/CS698-HAI-Project',
//   server: {
//     proxy: {
//       '/api': 'http://localhost:8001'
//     },
//     fs: {
//       strict: true
//     },
//     // headers: {
//     //   'Content-Type': 'application/javascript',
//     //   '.jsx': 'application/javascript',
//     //   '.js': 'application/javascript',
//     //   '.ts': 'application/javascript',
//     //   '.tsx': 'application/javascript'
//     // }
//   },
//   build: {
//     outDir: 'dist'
//   }
// })


import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/CS698-HAI-Project/',
  server: {
    proxy: {
      '/api': 'http://localhost:8001'
    },
  },
  build: {
    outDir: 'dist'
  }
})
