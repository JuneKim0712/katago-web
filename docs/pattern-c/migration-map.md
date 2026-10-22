# Pattern C Subsystem Migration Map

## Overview
This document specifies the migration mapping from the legacy TypeScript engine to the Pattern C Hybrid Emscripten C++ Wasm + WebGPU ONNX Bridge.
The legacy TypeScript engine remains available during transition under an explicit legacy namespace without leaking internal types.

---

## 1. Subsystems Replaced by Upstream C++ Wasm

The following subsystems are replaced by the canonical upstream Lightvector C++ implementation (`native/upstream/cpp/`) compiled via Emscripten:

| Subsystem | Legacy TypeScript Path | Canonical Upstream C++ Implementation |
|---|---|---|
| **Board, Liberties, Ko, Groups** | `src/engine/fastBoard.ts`, `src/engine/board/` | `native/upstream/cpp/game/board.h`, `board.cpp` |
| **Move History, Zobrist, Superko** | `src/engine/board/context.ts`, `src/utils/ko.ts` | `native/upstream/cpp/game/boardhistory.h`, `boardhistory.cpp` |
| **Rulesets (Presets & Custom Clauses)** | `src/engine/types.ts` (partial) | `native/upstream/cpp/game/rules.h`, `rules.cpp` |
| **Ladders, Area & Territory Scoring** | `src/engine/board/ladders.ts`, `area.ts`, `territoryKomi.ts` | `native/upstream/cpp/game/board.cpp`, `boardhistory.cpp` |
| **Neural Feature Extraction (Spatial 22, Global 19)** | `src/engine/featuresV7Fast.ts`, `src/engine/features/` | `native/upstream/cpp/neuralnet/nninputs.h`, `nninputs.cpp` |
| **Symmetries & Masking** | `src/engine/features/encoder.ts` | `native/upstream/cpp/neuralnet/symmetry.h`, `symmetry.cpp`, `nneval.cpp` |
| **Neural Postprocessing (Softmax, Tanh, Score Moments)** | `src/engine/evalV8.ts`, `src/engine/scoreValue.ts` | `native/upstream/cpp/neuralnet/nneval.h`, `nneval.cpp`, `nninterface.h` |
| **MCTS Tree & Node Structures** | `src/engine/search/tree.ts` | `native/upstream/cpp/search/searchnode.h`, `searchnode.cpp` |
| **MCTS Selection, PUCT & Dynamic FPU** | `src/engine/search/selection.ts` | `native/upstream/cpp/search/searchpuct.cpp` |
| **MCTS Expansion & Backpropagation** | `src/engine/search/expansion.ts`, `tree.ts` | `native/upstream/cpp/search/search.cpp`, `searchnode.cpp` |
| **Search Utility & Contempt** | `src/engine/search/utility.ts`, `searchParams.ts` | `native/upstream/cpp/search/searchparams.h`, `searchparams.cpp` |
| **Candidate Ordering & PV Construction** | `src/engine/search/report.ts` | `native/upstream/cpp/search/search.cpp`, `searchprint.cpp` |
| **Analysis Protocol Reporting** | `src/engine/analyzeMcts.ts` | `native/upstream/cpp/command/command.h`, `analyze.cpp` |

---

## 2. Subsystems Retained and Adapted

| Subsystem | Existing Path | Pattern C Adaptation / Target Path | Role in Architecture |
|---|---|---|---|
| **Public Engine Client API** | `src/worker/client.ts` | `src/client/engineClient.ts`, `src/api/` | Implements Contract v1 `KataGoEngine`, exposes `createEngine`, handles subscriptions and request IDs. |
| **Coordinator Worker** | `src/worker/worker.ts` | `src/worker/coordinator.ts` | Dedicated worker hosting Wasm module, JSPI bridge, and ONNX Runtime WebGPU session. |
| **Model Fetching & Integrity** | `src/model/loadModelV8.ts`, `modelCache.ts` | `src/models/modelLoader.ts`, `src/storage/` | SHA256 validation, Cache API / IndexedDB storage, manifest parsing. |
| **Resource Profiles & Caps** | `src/worker/resourceProfile.ts` | `src/runtime/resourceProfiles.ts` | Hardware concurrency, memory budgets, batch/thread tuning. |
| **Capability Detection** | `src/worker/backendFallback.ts` | `src/runtime/capabilities.ts` | WebGPU, JSPI, SharedArrayBuffer, OPFS, thread capabilities inspection. |
| **Error Hierarchy** | `src/worker/gpuError.ts` | `src/api/errors.ts` | Normalized `EngineFailure` with normative `ErrorCode` enum. |
| **Legacy Entry Points** | `src/index.ts`, `src/engine/` | `src/legacy/` (internal fallback) | Maintained for backward compatibility; isolated from new v1 contract. |

---

## 3. New Architecture Modules (Pattern C Target Layout)

```text
native/
  upstream/         # Pinned Lightvector KataGo v1.18.1 source (untouched)
  patches/          # Minimal CMake and platform integration patches
  backend/          # Custom nninterface.h implementation bridging to JSPI
  bridge/           # ABI struct layouts, static assertions, memory slots
  host/             # Native analysis host, command queue, cancellation
  tests/            # Native unit tests and C++ reference fixtures
src/
  api/              # Normative Public Contract v1 types and validation
  client/           # Client wrapper, typed event emitter, request lifecycle
  worker/           # Coordinator worker, state machine, JSPI RPC dispatcher
  inference/ort/    # ONNX Runtime WebGPU session, memory descriptors, placement verification
  runtime/          # Wasm module loader, JSPI bridge binding, capabilities
  models/           # Manifest schema, download integrity, variant selection
  storage/          # Cache API / IndexedDB model storage and OPFS mount helper
test/
  fixtures/         # Reproducible board, feature, and neural output fixtures
  native-parity/    # C++ vs Wasm parity verification suite
  browser/          # Chrome browser integration, JSPI, pthread, and lifecycle tests
benchmarks/
  pattern-c/        # Repeatable throughput, latency, and memory benchmark harness
docs/
  pattern-c/        # Progress logs, decisions, ABI spec, migration map
```
