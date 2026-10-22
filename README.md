# katago-web

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-Ready-green.svg)](https://www.w3.org/TR/webgpu/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.30-blue.svg)](https://onnxruntime.ai/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-orange.svg)](https://www.tensorflow.org/js)

High-performance, web-native [KataGo](https://github.com/lightvector/KataGo) Go/Baduk AI analysis engine. Executes official KataGo neural network models directly in the browser and Node.js via **WebGPU**, **WASM (JSPI + Pthreads)**, and pure TypeScript with zero external server dependencies.

---

## Key Highlights & Architectures

`katago-web` provides two complementary engine implementations:

### 1. Pattern C Hybrid Architecture (Upstream C++ Wasm + ONNX Runtime WebGPU)
- **Upstream C++ Core (v1.18.1)**: Pinned Lightvector KataGo v1.18.1 (`92ee95c0`) compiled to WebAssembly via Emscripten using **JSPI (JavaScript Promise Integration)** and **POSIX Pthreads** (`SharedArrayBuffer`). Search tree, rule enforcement, liberty tracking, and feature preparation run at native C++ speeds with zero drift.
- **ONNX Runtime WebGPU Bridge**: Evaluates official KataGo models on the GPU with DirectML/WebGPU kernels. Evaluator pthreads suspend cooperatively via JSPI during async GPU inference while background worker threads progress uninterrupted.
- **Fixed-Width ABI Bridge**: Double-buffered shared memory descriptor (`BridgeSlotDescriptor`) communicating directly between Wasm and the TypeScript coordinator with `Atomics` signaling.
- **Contract v1 Conformance**: Strictly satisfies the `@baduk/engine-client` Contract v1 specification with `createEngine()`, typed state machine (`uninitialized` $\to$ `ready` $\to$ `analyzing` $\to$ `terminated`), operation IDs, cancellation, and progress streaming.

### 2. Upstream-Faithful Pure TypeScript Core & Contract v1
- **Canonical Game State & Untruncated History**: Retains full move histories and initial stone setups from move 0 to endgame without 5-move truncation.
- **Superko & Complete Rulesets**: 64-bit Zobrist hashing for Positional Superko (Chinese, Tromp-Taylor) and Situational Superko (AGA, New Zealand), simple ko, multi-stone suicide legality, and Button Go (+0.5 pt). Presets for Japanese, Chinese, Chinese-Rulesist, Korean, AGA, Tromp-Taylor, New Zealand, and Stone Counting.
- **Backend-Neutral Evaluation Boundary (`EvaluationBackend`)**: Decouples MCTS search from the underlying inference runtime (ONNX Runtime, TensorFlow.js, or mock evaluators). Supports both **NHWC** and **NCHW** tensor layouts.
- **Full Upstream PUCT & Dynamic FPU**: MCTS selection matching `cpp/search/searchpuct.cpp` with parent visit weighting and dynamic First Play Urgency (FPU) reduction.

### 3. Legacy Pure TypeScript TensorFlow.js Engine
- Maintained for backward compatibility: direct `.bin` and `.bin.gz` weight parser, pipelined double-buffered ping-pong evaluation, and `KataGoEngineClient` API.

---

## Installation

```bash
npm install katago-web onnxruntime-web
```

Optional peer dependencies for legacy TensorFlow.js backends:
```bash
npm install @tensorflow/tfjs @tensorflow/tfjs-backend-wasm @tensorflow/tfjs-backend-webgpu
```

---

## Quick Start

### 1. Modern Contract v1 Engine (Recommended for Web Applications & Baduk)

```typescript
import { createEngine, BASELINE_B10C384_MANIFEST } from 'katago-web';

// 1. Create engine instance with background coordinator worker
const engine = createEngine({
  workerFactory: () => new Worker(
    new URL('katago-web/coordinator', import.meta.url),
    { type: 'module' }
  ),
});

// 2. Subscribe to progress & telemetry events
engine.subscribe((event) => {
  if (event.type === 'load-progress') {
    console.log(`[Loading] ${event.stage}: ${(event.fraction * 100).toFixed(0)}%`);
  } else if (event.type === 'analysis-update') {
    console.log(`[Visits] ${event.update.visits}, WinRate: ${(event.update.rootWinRate * 100).toFixed(1)}%`);
  }
});

// 3. Initialize with official KataGo model manifest
await engine.initialize({
  operationId: 'init-1',
  model: { kind: 'manifest', manifest: BASELINE_B10C384_MANIFEST },
  backendPreference: 'webgpu',
});

// 4. Run position analysis
const emptyBoard = Array.from({ length: 19 }, () => Array(19).fill(null));

const result = await engine.analyze({
  operationId: 'analyze-1',
  position: {
    board: emptyBoard,
    currentPlayer: 'black',
    rules: 'chinese',
    komi: 7.5,
  },
  limits: {
    maxVisits: 100,
    maxTimeMs: 15000,
  },
});

console.log('Analysis Complete! Root WinRate:', result.rootInfo.rootWinRate);
console.log('Top move:', result.moves[0]);

// 5. Clean teardown
await engine.dispose();
```

### 2. Backend-Neutral Core Runtime (Independent MCTS Search)

```typescript
import {
  BoardHistory,
  RULES_PRESETS,
  SearchSession,
  type EvaluationBackend,
} from 'katago-web';

// Initialize upstream BoardHistory with full ruleset
const history = new BoardHistory(RULES_PRESETS.chinese, 'black', 19, 19);

// Connect any Contract v1 EvaluationBackend (ORT, TFJS, or custom)
const session = new SearchSession(history, myEvaluationBackend, 7.5);

// Run 50 MCTS visits
const report = await session.runVisits(50);
console.log(`Root Visits: ${report.rootVisits}, WinRate: ${(report.rootWinrate * 100).toFixed(1)}%`);
console.log('Top Moves:', report.topMoves);
```

### 3. Legacy TypeScript Engine Client

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

## Architecture Overview

```text
katago-web/
├── native/                  # Upstream C++ Wasm implementation
│   ├── upstream/            # Pinned Lightvector KataGo v1.18.1 source (git submodule)
│   ├── backend/             # Custom nninterface bridging Wasm to JSPI WebGPU
│   ├── bridge/              # bridge_abi.h (168B packed struct, memory slots)
│   └── host/                # wasm_analysis_host.cpp (MCTS analysis lifecycle)
├── src/
│   ├── api/                 # Contract v1 normative interfaces, types, & validation
│   ├── client/              # Contract v1 client wrapper & typed event emitter
│   ├── worker/              # Coordinator worker hosting Wasm + ONNX Runtime WebGPU
│   ├── inference/           # ONNX Runtime WebGPU bridge with direct buffer mapping
│   ├── runtime/             # Wasm loader, JSPI bridge binding, & capabilities
│   ├── models/              # Model manifest schemas, checksum validation, & storage
│   ├── core-contract/       # Pure TS rules, canonical position, & EvaluationBackend (Contract v1)
│   ├── core-runtime/        # Pure TS BoardHistory, superko Zobrist, features, & MCTS search
│   └── engine/              # Legacy TypeScript engine (TFJS, fastBoard, ping-pong pipeline)
├── test/
│   ├── abi.test.ts          # C++ vs TypeScript ABI memory layout static validation
│   ├── contract_validation.test.ts  # Contract v1 schema & input validation suite
│   ├── core-upstream/       # Upstream rules, 22-plane features, superko, & search tests
│   ├── native-parity/       # Native C++ reference fixture parity tests
│   └── browser/             # Playwright Chrome tests (JSPI concurrency & E2E WebGPU analysis)
└── docs/
    ├── pattern-c/           # Pattern C architecture, progress, & migration records
    └── core-upstream/       # Core upstream handoff & progress specifications
```

---

## Verification & Testing

### 1. Contract & Unit Test Suite
Run the 50+ unit and differential tests across ABI layout, Contract v1 validation, upstream rules, feature planes, and search integration:
```bash
npx vitest run test/abi.test.ts test/contract_validation.test.ts test/nativeRequest.test.ts test/core-upstream/
```

### 2. Browser Qualification Tests (Playwright Chrome + WebGPU)
Run the headless Chrome qualification verifying JSPI pthread suspension and end-to-end neural network inference on WebGPU:
```bash
npx playwright test test/browser/concurrency_gate.spec.ts test/browser/engine_e2e.spec.ts
```

### 3. Build & Typecheck
Compile TypeScript sources and build declaration maps:
```bash
npm run typecheck
npm run build
```

---

## Package Exports

The package exports distinct entry points for tree-shaking and isolation:
- `katago-web`: Main entry point (Public Contract v1 client, pure TS core, and legacy client).
- `katago-web/coordinator`: Dedicated Web Worker entry point for Pattern C Wasm + WebGPU execution.
- `katago-web/contracts`: Type declarations conforming to `@baduk/engine-client` Contract v1.

---

## License

[MIT](LICENSE) — Includes code from [KataGo](https://github.com/lightvector/KataGo) (David J. Wu / Lightvector, BSD-3 / MIT).
