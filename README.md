# katago-web

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-orange.svg)](https://www.tensorflow.org/js)

High-performance, pure TypeScript port of the [KataGo](https://github.com/lightvector/KataGo) Go/Baduk AI engine for the Web and Node.js. Runs official KataGo neural network models directly in the browser via **WebGPU** and **WASM** with zero native C++ binaries or external backend servers.

---

## Features

- **Direct Binary Model Parser:** Parses official KataGo `.bin` and `.bin.gz` weight files directly in JavaScript. Supports network architectures from V8 up to modern V17 (Transformer Attention, Nested Bottlenecks, GPool, SwiGLU, and RoPE).
- **Pipelined Asynchronous MCTS:** Full Monte Carlo Tree Search implementation with double-buffered ping-pong GPU/CPU pipelining, zero-copy typed arrays, and in-place feature extraction (~57+ visits/second on CPU WASM and ultra-fast throughput on WebGPU).
- **100% Bitwise Mathematical Parity:** Verified with strict cryptographic SHA-256 and floating-point tests against native KataGo v1.18.1 across Japanese and Chinese rule sets.
- **Off-Main-Thread Worker Architecture:** Non-blocking Web Worker client with automatic hardware capability probing, watchdog auto-recovery, cancellation, and real-time streaming progress callbacks.
- **Fast Rule Engine & Bitboard:** Complete 19x19 Go logic, liberty tracking, ladder calculation, and area/territory scoring.

---

## Installation

```bash
npm install katago-web @tensorflow/tfjs @tensorflow/tfjs-backend-wasm @tensorflow/tfjs-backend-webgpu
```

---

## Quick Start

### 1. Browser (Web Worker)

```typescript
import { KataGoEngineClient } from 'katago-web';

// Initialize the background worker
const client = new KataGoEngineClient();

// Listen to model download & loading progress
client.onLoadProgress((progress) => {
  console.log(`[${progress.stage}] ${(progress.progress * 100).toFixed(0)}%`);
});

// Load official KataGo weight file
await client.init({
  modelUrl: '/models/katago-small.bin.gz',
  backend: 'webgpu', // Automatically falls back to WASM if WebGPU is unavailable
});

// Run analysis on a position
const emptyBoard = Array.from({ length: 19 }, () => Array(19).fill(null));

const result = await client.analyze({
  board: emptyBoard,
  currentPlayer: 'black',
  rules: 'japanese',
  komi: 6.5,
  visits: 100,
  maxTimeMs: 5000,
  onProgress: (progress) => {
    console.log(`Visits: ${progress.rootVisits}, WinRate: ${(progress.rootWinRate * 100).toFixed(1)}%`);
  },
});

console.log('Top candidate move:', result.moves[0]);
```

### 2. Node.js (Direct MCTS)

```typescript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import {
  parseKataGoModelV8,
  KataGoModelV8Tf,
  MctsSearch,
} from 'katago-web';

// Initialize WASM
await tf.setBackend('wasm');
await tf.ready();

// Load & parse binary model
const raw = readFileSync('./models/katago-small.bin.gz');
const parsed = parseKataGoModelV8(gunzipSync(raw));
const model = new KataGoModelV8Tf(parsed, { fuseRotary: true, fuseRmsNorm: true });

// Run 80-visit MCTS
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
  ownershipMode: 'root',
});

await search.run({ visits: 80, maxTimeMs: 10000, batchSize: 4 });
const analysis = search.getAnalysis({ topK: 5, analysisPvLen: 4 });

console.log(`Root Win Rate: ${(analysis.rootWinRate * 100).toFixed(1)}%`);
console.log('Top Moves:', analysis.moves.map(m => `(${m.x},${m.y}): ${m.visits} visits`));
```

---

## Architecture Overview

```
katago-web
├── engine/         # Pipelined MCTS, fast board representation, and rule engines
│   ├── analyzeMcts.ts     # Pipelined MCTS search with ping-pong buffering
│   ├── fastBoard.ts       # 19x19 board bitboard, liberties, ladders, scoring
│   ├── featuresV7Fast.ts  # KataGo V7 22-plane spatial & global feature tensor extractor
│   └── evalV8.ts          # Post-processing: value logits to win rate & score lead
├── model/          # Neural network inference in TensorFlow.js
│   ├── modelV8.ts         # KataGo trunk & policy/value head architecture (v8-v17)
│   ├── binModelParser.ts  # Binary weight format decoder (.bin / .bin.gz)
│   └── loadModelV8.ts     # Gzip weight decompressor
└── worker/         # Off-main-thread Web Worker orchestration
    ├── client.ts          # Main-thread client with request coalescing & watchdog
    ├── worker.ts          # Background worker execution loop
    └── resourceProfile.ts # Hardware memory & backend preference detection
```

---

## Verification & Testing

### 1. Rigorous Native KataGo v1.18.1 Parity Suite (`kata1-b10c384`)
The package includes an extensive test suite verifying mathematical and bitwise parity against official native KataGo v1.18.1 with the high-capacity `kata1-tf2-b10c384-s2941M-d5872M.bin.gz` network (10 blocks, 384 channels, Transformer attention, rotary embeddings):
- **Win Rate & Lead**: Evaluates black win probability ($\Delta < 0.0002$), score lead ($\Delta < 0.01$ pts), and selfplay score variance.
- **Full Policy Distribution**: Compares all 361 board coordinates + pass move ($\Delta < 0.0003$).
- **Top Move Selection & Ordering**: Verifies that the top-5 policy move choices and ranking orders match native KataGo 100%.
- **Complex Edge Cases**: Tests Japanese vs Chinese rules, pass moves, multi-stone captures, active ko fights (`koPoint` legality masking), running ladders, and 9-stone handicap games.
- **Multi-Batch Equivalence**: Proves single-sample vs batched inference ($B=4$) are bitwise equivalent.

```bash
npm run parity
```

### 2. Performance Benchmarking
Measure forward inference latency and pipelined MCTS visit speed:
```bash
npm run bench
```

### 3. Generate Parity Fixtures
Regenerate ground-truth fixtures directly from an official native KataGo executable:
```bash
npm run generate:parity [path/to/katago.exe] [path/to/model.bin.gz]
```

### 4. Build Distribution
Build ESM and CommonJS bundles with TypeScript `.d.ts` declaration maps:
```bash
npm run build
```

---

## License

[MIT](LICENSE)
