import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';
import {
  parseKataGoModelV8,
  KataGoModelV8Tf,
  MctsSearch,
  moveToGtp,
  type BoardState,
} from '../src/index.js';

async function main() {
  console.log('--- KataGo Web Standalone Analysis Demo ---');

  // 1. Initialize TF.js WASM
  const wasmDir = resolve(process.cwd(), 'node_modules/@tensorflow/tfjs-backend-wasm/dist');
  setWasmPaths(`${wasmDir}/`);
  await tf.setBackend('wasm');
  await tf.ready();
  console.log(`Backend initialized: ${tf.getBackend()}`);

  // 2. Load Model
  const modelPath = process.env.MODEL_PATH ?? resolve('../baduk/public/models/katago-small.bin.gz');
  if (!existsSync(modelPath)) {
    console.error(`Model file not found at ${modelPath}. Set MODEL_PATH env variable.`);
    process.exit(1);
  }
  console.log(`Loading model from: ${modelPath}`);
  const raw = readFileSync(modelPath);
  const parsed = parseKataGoModelV8(gunzipSync(raw));
  const model = new KataGoModelV8Tf(parsed, { fuseRotary: true, fuseRmsNorm: true });
  console.log(`Model loaded successfully: ${parsed.modelName}`);

  // 3. Setup Board (19x19 Empty Board)
  const emptyBoard: BoardState = Array.from({ length: 19 }, () => Array(19).fill(null));

  // 4. Initialize MCTS Engine
  const search = await MctsSearch.create({
    model,
    board: emptyBoard,
    currentPlayer: 'black',
    moveHistory: [],
    rules: 'japanese',
    komi: 6.5,
    nnRandomize: false,
    conservativePass: false,
    maxChildren: 64,
    ownershipMode: 'root',
    wideRootNoise: 0,
    rootSymmetrySamples: 1,
  });

  // 5. Run Search for 80 visits
  console.log('Running MCTS search for 80 visits...');
  const t0 = performance.now();
  await search.run({ visits: 80, maxTimeMs: 10000, batchSize: 4 });
  const elapsed = performance.now() - t0;

  // 6. Inspect Results
  const analysis = search.getAnalysis({ topK: 5, analysisPvLen: 4 });
  console.log(`Search completed in ${elapsed.toFixed(1)}ms (${((80 / elapsed) * 1000).toFixed(1)} visits/sec)`);
  console.log(`Root Win Rate: ${(analysis.rootWinRate * 100).toFixed(2)}%`);
  console.log(`Root Score Lead: ${analysis.rootScoreLead.toFixed(2)} points`);
  console.log('\nTop 5 Moves:');
  analysis.moves.slice(0, 5).forEach((m, i) => {
    const moveGtp = m.x < 0 ? 'pass' : `${'ABCDEFGHJKLMNOPQRST'[m.x]}${19 - m.y}`;
    console.log(` #${i + 1}: ${moveGtp.padEnd(5)} | visits: ${String(m.visits).padStart(3)} | winRate: ${(m.winRate * 100).toFixed(1)}% | scoreLead: ${m.scoreLead.toFixed(1)} | PV: ${m.pv.join(' ')}`);
  });

  model.dispose();
}

main().catch(console.error);
