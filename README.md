# katago-web

[![npm version](https://img.shields.io/npm/v/katago-web.svg)](https://www.npmjs.com/package/katago-web)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-Hardware_Accelerated-green.svg)](https://www.w3.org/TR/webgpu/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.30-blue.svg)](https://onnxruntime.ai/)
[![WebAssembly](https://img.shields.io/badge/WASM-Pthreads_%2B_JSPI-purple.svg)](https://webassembly.org/)

High-performance, browser-native [KataGo](https://github.com/lightvector/KataGo) Go/Baduk AI analysis engine. Executes official KataGo neural network models directly in the client browser and Node.js via **WebGPU**, **Multi-Threaded WebAssembly (JSPI + POSIX Pthreads)**, and pure TypeScript with zero external server dependencies.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Browser Requirements & Headers (COOP / COEP)](#browser-requirements--headers-coop--coep)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Modern Contract v1 Engine (Recommended)](#1-modern-contract-v1-engine-recommended)
  - [2. MCTS Search Tree Reuse (Visit Carryover)](#2-mcts-search-tree-reuse-visit-carryover)
  - [3. Backend-Neutral Standalone MCTS Search](#3-backend-neutral-standalone-mcts-search)
  - [4. Legacy TensorFlow.js Engine](#4-legacy-tensorflowjs-engine)
- [API Reference](#api-reference)
- [Supported Rulesets & Komi](#supported-rulesets--komi)
- [Development, Testing & Build](#development-testing--build)
- [Publishing to npm and GitHub](#publishing-to-npm-and-github)
- [License](#license)

---

## Key Features

- **Upstream C++ Native Precision (KataGo v1.18.1)**: Compiled directly from Lightvector KataGo source (`92ee95c0`) via Emscripten. Search tree traversals, dynamic First Play Urgency (FPU), parent weighting, superko hash tables, and territory scoring match upstream KataGo bit-for-bit with zero drift.
- **Hardware-Accelerated WebGPU**: Leverages ONNX Runtime WebGPU to evaluate 10-block, 18-block, and larger KataGo neural networks directly on user GPUs with zero server GPU costs.
- **JSPI + Pthread Suspension**: Background evaluator pthreads suspend cooperatively via JavaScript Promise Integration (JSPI) during asynchronous WebGPU dispatches, preventing CPU thread starvation and enabling high-throughput pipelining.
- **High-Efficiency Subtree Reuse**: Automatically preserves explored MCTS branches across consecutive player moves. Inherited visits carry over directly, providing instantaneous (<1ms) initial analysis and slashing compute requirements by 2× to 5×.
- **Full Ruleset Support**: Supports Positional Superko (Chinese, Tromp-Taylor), Situational Superko (AGA, New Zealand), simple ko, multi-stone suicide, Button Go, and custom komi/handicap configurations.
- **Contract v1 Conformance**: Strictly conforms to the standard `@baduk/engine-client` Contract v1 specification with lifecycle management (`uninitialized` $\to$ `ready` $\to$ `analyzing` $\to$ `terminated`), operation IDs, cancellation tokens, and progress streaming.

---

## Architecture Overview

```text
katago-web/
├── native/                  # Upstream C++ Wasm implementation
│   ├── upstream/            # Pinned Lightvector KataGo v1.18.1 source
│   ├── backend/             # Custom nninterface bridging Wasm to JSPI WebGPU
│   ├── bridge/              # bridge_abi.h (168B packed struct, shared memory slots)
│   └── host/                # wasm_analysis_host.cpp (MCTS analysis lifecycle & tree reuse)
├── src/
│   ├── api/                 # Contract v1 normative interfaces, types, & validation
│   ├── client/              # Contract v1 client wrapper & typed event emitter
│   ├── worker/              # Coordinator worker hosting Wasm + ONNX Runtime WebGPU
│   ├── inference/           # ONNX Runtime WebGPU bridge with direct buffer mapping
│   ├── runtime/             # Wasm loader, JSPI bridge binding, & capabilities
│   ├── models/              # Model manifest schemas, checksum validation, & storage
│   ├── core-contract/       # Pure TS rules, canonical position, & EvaluationBackend
│   ├── core-runtime/        # Pure TS BoardHistory, superko Zobrist, features, & MCTS search
│   └── engine/              # Pure TypeScript MCTS engine (TFJS, fastBoard, ping-pong pipeline)
```

| Engine Architecture | Runtime | Inference Engine | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Pattern C Hybrid (Recommended)** | WebAssembly (Pthreads + JSPI) | ONNX Runtime WebGPU | Production web apps, Baduk platforms, desktop Electron apps |
| **Core Runtime (Pure TS)** | Pure TypeScript / JavaScript | Backend-Neutral (`EvaluationBackend`) | Headless Node.js, unit testing, custom neural backends |
| **Legacy Engine Client** | Pure TypeScript Worker | TensorFlow.js (WebGPU / WASM) | Backward compatibility with `.bin` / `.bin.gz` models |

---

## Browser Requirements & Headers (COOP / COEP)

Because KataGo Web runs multi-threaded WebAssembly with `SharedArrayBuffer` and accesses the GPU via WebGPU, browsers require **Cross-Origin Isolation**. 

### 1. Browser Support
- Google Chrome / Chromium $\ge$ 113 (WebGPU + JSPI + SharedArrayBuffer)
- Microsoft Edge $\ge$ 113
- Safari $\ge$ 18 (with WebGPU enabled)
- Firefox Nightly (with WebGPU & SharedArrayBuffer enabled)

### 2. Required Security Headers
Your web server **must** deliver these two HTTP response headers:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

#### Vite (`vite.config.ts`):
```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
```

#### Next.js (`next.config.js`):
```javascript
module.exports = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        ],
      },
    ];
  },
};
```

#### Express:
```javascript
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});
```

---

## Installation

```bash
npm install katago-web onnxruntime-web
```

*Optional peer dependencies for legacy TensorFlow.js backends:*
```bash
npm install @tensorflow/tfjs @tensorflow/tfjs-backend-wasm @tensorflow/tfjs-backend-webgpu
```

---

## Quick Start

### 1. Modern Contract v1 Engine (Recommended)

```typescript
import { createEngine, BASELINE_B10C384_MANIFEST } from 'katago-web';

// 1. Instantiate engine pointing to the bundled coordinator worker
const engine = createEngine({
  workerFactory: () => new Worker(
    new URL('katago-web/coordinator', import.meta.url),
    { type: 'module' }
  ),
});

// 2. Subscribe to progress events and search updates
engine.subscribe((event) => {
  if (event.type === 'load-progress') {
    console.log(`[Loading] ${event.stage}: ${(event.fraction * 100).toFixed(0)}%`);
  } else if (event.type === 'ready') {
    console.log('[Ready] Engine initialized with model:', event.info.modelId);
  }
});

// 3. Initialize engine with official KataGo model manifest
await engine.initialize({
  operationId: 'init-1',
  model: { kind: 'manifest', manifest: BASELINE_B10C384_MANIFEST },
  runtime: 'webgpu',
});

// 4. Request position analysis
const result = await engine.analyze({
  requestId: 'req-1',
  position: {
    kind: 'game',
    dimensions: { width: 19, height: 19 },
    initialPlayer: 'black',
    initialStones: [],
    rules: { kind: 'preset', name: 'japanese' },
    komi: 6.5,
    moves: [
      { player: 'black', move: { kind: 'play', point: { x: 3, y: 3 } } }, // D4
    ],
  },
  search: {
    limits: { kind: 'visits', maxVisits: 200 },
    searchThreads: 2,
  },
  output: {
    pvLength: 15,
    ownership: 'tree',
    includePolicy: true,
  },
});

console.log('Root WinRate (Black):', (result.snapshot.evaluation.blackWinRate * 100).toFixed(1) + '%');
console.log('Score Lead:', result.snapshot.evaluation.blackScoreLead.toFixed(1));
console.log('Top Moves:', result.snapshot.candidates.slice(0, 3));

// 5. Clean teardown when done
await engine.dispose();
```

---

### 2. MCTS Search Tree Reuse (Visit Carryover)

When consecutive analysis calls represent sequential moves in a game, `katago-web` automatically detects the move prefix and promotes the matching child subtree to the root:

```typescript
// Move 1: Search root position for 500 visits
const res1 = await engine.analyze({
  requestId: 'req-move-1',
  position: { /* empty board */ },
  search: { limits: { kind: 'visits', maxVisits: 500 } },
});
// Candidate move Q16 accumulated e.g. 320 visits during search.

// Move 2: Black plays Q16. Request 500 visits.
const res2 = await engine.analyze({
  requestId: 'req-move-2',
  position: {
    /* board with Q16 played */
    moves: [{ player: 'black', move: { kind: 'play', point: { x: 15, y: 3 } } }],
  },
  search: { limits: { kind: 'visits', maxVisits: 500 } },
});

// Accounting accurately reflects reused vs new visits:
console.log(res2.snapshot.accounting.reusedVisits);       // 320 visits inherited!
console.log(res2.snapshot.accounting.completedNewVisits); // Only 180 new visits executed!
console.log(res2.snapshot.rootVisits);                    // 500 total visits
```

---

### 3. Backend-Neutral Standalone MCTS Search

For custom evaluation pipelines, Node.js scripts, or offline simulation, use the pure TypeScript core:

```typescript
import {
  BoardHistory,
  RULES_PRESETS,
  SearchSession,
  type EvaluationBackend,
} from 'katago-web';

// Initialize board history with standard Chinese rules
const history = new BoardHistory(RULES_PRESETS.chinese, 'black', 19, 19);

// Create session with any Contract v1 EvaluationBackend implementation
const session = new SearchSession(history, myEvaluationBackend, 7.5);

// Run 50 MCTS visits
const report = await session.runVisits(50);
console.log(`Visits: ${report.rootVisits}, WinRate: ${(report.rootWinrate * 100).toFixed(1)}%`);

// Play a move and reuse tree
const topLoc = report.topMoves[0].loc;
const reusedVisits = session.makeMove(topLoc);
console.log(`Preserved ${reusedVisits} visits for next turn!`);

// Run until total visits reaches 100 (executes only 100 - reusedVisits)
const nextReport = await session.runUntilVisits(100);
```

---

### 4. Legacy TypeScript Engine Client

```typescript
import { KataGoEngineClient } from 'katago-web';

const client = new KataGoEngineClient();
await client.init({
  modelUrl: '/models/kata1-b10c384.bin.gz',
  backend: 'webgpu',
});

const result = await client.analyze({
  board: Array.from({ length: 19 }, () => Array(19).fill(null)),
  currentPlayer: 'black',
  rules: 'japanese',
  komi: 6.5,
  visits: 100,
});
console.log('Top Move:', result.moves[0]);
```

---

## API Reference

### `createEngine(options?: EngineOptions): KataGoEngine`

Creates a new engine instance.

```typescript
interface EngineOptions {
  workerFactory?: () => Worker;
}
```

### Engine Methods

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `initialize()` | `request: InitRequest, options?: { signal? }` | `Promise<EngineInfo>` | Loads neural model and initializes WebAssembly runtime |
| `analyze()` | `request: AnalyzeRequest, options?: { signal? }` | `Promise<AnalysisResult>` | Dispatches MCTS analysis on specified position |
| `subscribe()` | `listener: (event: EngineEvent) => void` | `() => void` | Subscribes to engine events (`load-progress`, `ready`, etc.) |
| `cancel()` | `requestId: string` | `Promise<void>` | Preemptively aborts an in-flight analysis request |
| `reset()` | none | `Promise<void>` | Resets search tree and terminates pending jobs |
| `dispose()` | none | `Promise<void>` | Terminates worker and releases GPU/WASM memory |

### Output Data Model (`AnalysisSnapshot`)

```typescript
interface AnalysisSnapshot {
  requestId: string;
  rootVisits: number;
  evaluation: {
    blackWinRate: number;         // [0.0, 1.0]
    blackWinProbability: number;  // [0.0, 1.0]
    whiteWinProbability: number;  // [0.0, 1.0]
    blackScoreLead: number;       // Predicted score difference
    blackScoreMean: number;
    scoreStdev: number;
  };
  candidates: Array<{
    move: Move;                   // { kind: 'play', point: { x, y } } or { kind: 'pass' }
    visits: number;
    winrate: number;
    scoreLead: number;
    prior: number;
    pv: Move[];                   // Principal variation
  }>;
  policy: Float32Array | null;     // Prior policy distribution (362 floats)
  ownership: {
    ownerMap: Float32Array;       // Territory ownership [-1.0 (White) to +1.0 (Black)]
  } | null;
  accounting: {
    elapsedMs: number;
    completedNewVisits: number;
    reusedVisits: number;         // Visits inherited from previous turn's tree
  };
}
```

---

## Supported Rulesets & Komi

| Ruleset Preset | Scoring | Ko Rule | Suicide | Button Go | Handicap Compensation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `japanese` | Territory | Simple | Disallowed | No | 0 |
| `korean` | Territory | Simple | Disallowed | No | 0 |
| `chinese` | Area | Positional Superko | Disallowed | No | $N$ |
| `chinese-rulesist` | Area | Positional Superko | Disallowed | Yes (+0.5) | $N$ |
| `aga` | Area | Situational Superko | Disallowed | No | $N-1$ |
| `tromp-taylor` | Area | Positional Superko | Allowed | No | 0 |
| `new-zealand` | Area | Situational Superko | Allowed | No | 0 |

---

## Development, Testing & Build

### Prerequisites
- Node.js $\ge$ 20
- npm $\ge$ 10

### 1. Build and Typecheck
```bash
npm run typecheck    # Validate TypeScript types (tsc --noEmit)
npm run build        # Compile dist/ via tsconfig.build.json
```

### 2. Run Test Suites
```bash
# Core rules, features, ABI layout, and contract validation:
npm test

# Search tree reuse and visit carryover tests:
npx vitest run test/searchReuse.test.ts

# Browser-native WebGPU integration tests (Playwright):
npx playwright test test/browser/engine_e2e.spec.ts
```

---

## Publishing to npm and GitHub

For detailed release procedures, security requirements, and CI/CD setup, refer to [PUBLISHING.md](PUBLISHING.md).

### Quick Release Checklist

1. **Verify quality gates**:
   ```bash
   npm run typecheck && npm run build && npm test
   ```
2. **Commit and push to GitHub**:
   ```bash
   git add .
   git commit -m "chore: release v1.0.1"
   git push origin master
   ```
3. **Publish to npm (with 2FA)**:
   ```bash
   npm publish --access public --otp=YOUR_6_DIGIT_2FA_CODE
   ```

---

## Package Exports

The package provides subpath exports configured in `package.json`:
- `katago-web`: Main entry point (Public Contract v1 client, pure TS core runtime).
- `katago-web/coordinator`: Dedicated Web Worker script for Pattern C WASM + WebGPU execution.
- `katago-web/worker`: Legacy pure TypeScript worker.
- `katago-web/contracts`: Type declarations conforming to `@baduk/engine-client` Contract v1.

---

## License

[MIT](LICENSE) — Includes code from [KataGo](https://github.com/lightvector/KataGo) (David J. Wu / Lightvector, BSD-3 / MIT).
