import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import {
  parseKataGoModelV8,
  KataGoModelV8Tf,
  MctsSearch,
  fillInputsV7Fast,
} from '../dist/index.js';

const modelPath = [
  process.env.KATAGO_MODEL,
  'C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz',
  resolve('models/kata1-tf2-b10c384-s2941M-d5872M.bin.gz'),
].find(p => p && existsSync(p));

if (!modelPath) {
  console.error('Model not found!');
  process.exit(1);
}

const wasmDir = resolve(process.cwd(), 'node_modules/@tensorflow/tfjs-backend-wasm/dist');
setWasmPaths(`${wasmDir}/`);
await tf.setBackend('wasm');
await tf.ready();

const raw = readFileSync(modelPath);
const parsed = parseKataGoModelV8(gunzipSync(raw));
const model = new KataGoModelV8Tf(parsed, {
  fuseRotary: true,
  fuseRmsNorm: true,
  fuseTransformerProjections: true,
});

// 1. Forward Pass Latency Benchmark (Batch = 1)
const spatialData = new Float32Array(19 * 19 * 22);
const globalData = new Float32Array(19);
fillInputsV7Fast({
  stones: new Uint8Array(361),
  koPoint: -1,
  currentPlayer: 'black',
  recentMoves: [],
  komi: 7.5,
  outSpatial: spatialData,
  outGlobal: globalData,
});

const spatial = tf.tensor4d(spatialData, [1, 19, 19, 22]);
const global = tf.tensor2d(globalData, [1, 19]);

// Warmup
const w = model.forwardPolicyValue(spatial, global);
await Promise.all([w.policy.data(), w.value.data()]);
tf.dispose(w);

// 3 iterations
const forwardTimes = [];
for (let i = 0; i < 3; i++) {
  const t0 = performance.now();
  const out = model.forwardPolicyValue(spatial, global);
  await Promise.all([out.policy.data(), out.value.data()]);
  forwardTimes.push(performance.now() - t0);
  tf.dispose(out);
}
spatial.dispose();
global.dispose();

forwardTimes.sort((a, b) => a - b);
const medianForwardMs = forwardTimes[1];

// 2. MCTS Search Throughput Benchmark (10 visits, batch=2)
const emptyBoard = Array.from({ length: 19 }, () => Array(19).fill(null));
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
  ownershipMode: 'none',
  wideRootNoise: 0,
  rootSymmetrySamples: 1,
});

const tSearch0 = performance.now();
await search.run({ visits: 10, maxTimeMs: 60000, batchSize: 2 });
const searchElapsed = performance.now() - tSearch0;
const analysis = search.getAnalysisLight({ topK: 1, analysisPvLen: 1 });
const visitsPerSec = analysis.rootVisits / (searchElapsed / 1000);

model.dispose();

console.log(JSON.stringify({
  forwardMs: Number(medianForwardMs.toFixed(2)),
  evalsPerSec: Number((1000 / medianForwardMs).toFixed(2)),
  mctsVisits: analysis.rootVisits,
  mctsSearchMs: Number(searchElapsed.toFixed(2)),
  mctsVisitsPerSec: Number(visitsPerSec.toFixed(2)),
  topMovePrior: Number((analysis.moves[0]?.prior ?? 0).toFixed(4)),
}));
