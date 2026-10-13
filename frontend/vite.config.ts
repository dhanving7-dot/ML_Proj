import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const target = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";
const api = Object.fromEntries(["/api", "/docs", "/redoc", "/openapi.json"].map((p) => [p, { target, changeOrigin: true }]));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true, proxy: api },
  preview: { port: 4173, strictPort: true, proxy: api },
  build: { chunkSizeWarningLimit: 900 },
});
