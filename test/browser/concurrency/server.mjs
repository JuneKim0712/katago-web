import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '../../../');

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.txt': 'text/plain',
};

export function startServer(port = 8888) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      // Set COOP/COEP headers for SharedArrayBuffer / pthreads
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Access-Control-Allow-Origin', '*');

      const urlPath = req.url.split('?')[0];
      let filePath = path.join(ROOT_DIR, urlPath.replace(/^\//, ''));

      if (urlPath === '/onnxruntime-web') {
        filePath = path.join(ROOT_DIR, 'node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs');
      } else if (!fs.existsSync(filePath) && fs.existsSync(filePath + '.js')) {
        filePath = filePath + '.js';
      }

      if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        console.error(`[SERVER 404] ${req.url} -> ${filePath}`);
        res.statusCode = 404;
        res.end(`Not found: ${req.url}`);
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = MIME_TYPES[ext] || 'application/octet-stream';
      res.setHeader('Content-Type', contentType);

      if (ext === '.js' || ext === '.mjs') {
        let content = fs.readFileSync(filePath, 'utf-8');
        // Rewrite bare onnxruntime-web import to direct bundle path
        content = content.replace(/from\s+['"]onnxruntime-web['"]/g, `from '/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs'`);
        // Rewrite relative extensionless imports
        content = content.replace(/from\s+['"](\.[^'"]*?)(?<!\.js)(?<!\.mjs)['"]/g, `from '$1.js'`);
        res.end(content);
        return;
      }

      const stream = fs.createReadStream(filePath);
      stream.pipe(res);
    });

    server.listen(port, () => {
      resolve(server);
    });
  });
}

if (process.argv[1] && process.argv[1].endsWith('server.mjs')) {
  const port = parseInt(process.env.PORT || '8888', 10);
  startServer(port).then(() => {
    console.log(`Server running at http://localhost:${port}/`);
  });
}
