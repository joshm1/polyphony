import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite builds the React app into `../src/polyphony/static/`, which is
// committed to the repo. At runtime Python's FastAPI server:
//   - mounts /assets/* → static/assets/
//   - serves /              → static/index.html (with __POLYPHONY_DATA__ substituted)
//   - handles /api/*        → audio streaming, JSON payload, etc.
// Keep a single-entry bundle with predictable filenames so the HTML can
// reference `/assets/main.js` + `/assets/main.css` without manifest lookups.
export default defineConfig({
  plugins: [react()],
  root: ".",
  base: "/",
  build: {
    outDir: "../src/polyphony/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/main.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
    // Source maps land next to the built files — handy when debugging a
    // deployed playground without re-installing Node.
    sourcemap: true,
    minify: true,
    target: "es2020",
  },
  server: {
    // During `vite dev`, proxy API + static-file requests to the Python
    // backend so the React app can reuse the same contract as production.
    proxy: {
      "/api": "http://localhost:8787",
    },
  },
});
