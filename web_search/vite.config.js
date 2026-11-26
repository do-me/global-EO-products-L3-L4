import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const docsDir = path.resolve(repoRoot, 'docs');
const distDir = path.resolve(repoRoot, 'dist');

const DOCS_COPY_ITEMS = ['index.html', 'app.js', 'sources.html', 'SOURCES.md', 'data'];

function copyStaticDocsPlugin() {
  return {
    name: 'copy-static-catalogue-docs',
    closeBundle() {
      fs.mkdirSync(distDir, { recursive: true });
      for (const item of DOCS_COPY_ITEMS) {
        const src = path.join(docsDir, item);
        const dst = path.join(distDir, item);
        if (!fs.existsSync(src)) continue;
        fs.cpSync(src, dst, { recursive: true, force: true });
      }
      // Prevent GitHub Pages Jekyll processing when serving static assets
      fs.writeFileSync(path.join(distDir, '.nojekyll'), '');
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const cleanUrl = (req.url || '/').split('?')[0];
        const candidates = [
          path.join(distDir, cleanUrl.replace(/^\/+/, '')),
          path.join(docsDir, cleanUrl.replace(/^\/+/, '')),
        ];
        for (const fp of candidates) {
          if (fs.existsSync(fp) && fs.statSync(fp).isFile()) {
            if (fp.endsWith('.json')) res.setHeader('Content-Type', 'application/json; charset=utf-8');
            else if (fp.endsWith('.gz')) res.setHeader('Content-Type', 'application/octet-stream');
            else if (fp.endsWith('.js')) res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
            else if (fp.endsWith('.html')) res.setHeader('Content-Type', 'text/html; charset=utf-8');
            else if (fp.endsWith('.md')) res.setHeader('Content-Type', 'text/markdown; charset=utf-8');
            fs.createReadStream(fp).pipe(res);
            return;
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  root: __dirname,
  base: './',
  plugins: [tailwindcss(), copyStaticDocsPlugin()],
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  worker: {
    format: 'es',
  },
  build: {
    outDir: distDir,
    emptyOutDir: false,
    target: 'es2022',
    rollupOptions: {
      input: {
        search: path.resolve(__dirname, 'search.html'),
      },
    },
  },
});
