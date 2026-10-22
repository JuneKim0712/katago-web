# Core Upstream Integration Handoff (Contract v1)

## 1. Executive Summary & Objective

This document provides complete instructions for integrating the upstream-faithful KataGo core (`src/core-contract/` and `src/core-runtime/`) into the main engine pipeline once the benchmark between TensorFlow.js WebGPU and ONNX Runtime WebGPU is concluded.

The upstream core is pinned to **Lightvector KataGo v1.18.1** (`4f8849b...`), delivering:
- Full untruncated move history and initial stone setup tracking.
- Full ruleset support: Japanese, Chinese, Korean, AGA, Tromp-Taylor, New Zealand, Stone Counting.
- 64-bit Zobrist hashing for positional and situational superko detection.
- Multi-stone suicide legality matching Tromp-Taylor and New Zealand rules.
- Upstream neural net feature generation (22 spatial channels, 19 global channels) in both **NHWC** and **NCHW** layouts.
- Backend-neutral **Contract v1** evaluation boundary (`EvaluationBackend`).
- Upstream MCTS search with dynamic First Play Urgency (FPU) and PUCT selection formula matching `cpp/search/searchpuct.cpp`.

---

## 2. Contract v1 Evaluation Boundary

The evaluation boundary cleanly decouples MCTS search mechanics from any specific tensor framework.

### Interface Definition (`src/core-contract/evaluation.ts`)

```typescript
export interface EvaluationBackend {
  readonly name: string;
  readonly capabilities: EvaluationCapabilities;

  /**
   * Evaluates a batch of positions. Throws EvaluationCancelledError if the batch generation is obsolete.
   */
  evaluate(batch: EvaluationBatch): Promise<EvaluationResult>;

  /**
   * Optional warmup method to allocate GPU buffers, compile shaders/kernels, and verify pipeline readiness.
   */
  warmup?(dimensions: { width: number; height: number; batchSize: number }): Promise<void>;

  /**
   * Releases GPU/CPU memory and resources.
   */
  dispose(): Promise<void> | void;
}
```

### Implementing `EvaluationBackend` for the Winning Runtime

Whether TFJS WebGPU or ONNX Runtime WebGPU is chosen, the implementation must adhere to:

1. **Capabilities Registration**:
   ```typescript
   class RuntimeEvaluationBackend implements EvaluationBackend {
     readonly name = 'TFJS-WebGPU' /* or 'ONNXRuntime-WebGPU' */;
     readonly capabilities: EvaluationCapabilities = {
       supportedLayouts: ['NHWC'], // or ['NCHW', 'NHWC']
       maxBatchSize: 16,
       supportsFp16: true,
       supportsOwnership: true,
       supportsIntermediateActivations: false,
     };
   ...
   ```

2. **Batch Layout Handling**:
   - `batch.layout`: `'NHWC'` (default) or `'NCHW'`.
   - `batch.spatial`: `Float32Array` of size `batchSize * 19 * 19 * 22`.
   - `batch.global`: `Float32Array` of size `batchSize * 19`.
   - `batch.generation`: Search generation index. If `batch.generation < this.activeGeneration`, the backend should reject or discard the evaluation promptly by throwing `EvaluationCancelledError`.

3. **Output Format**:
   - `policy`: Flattened `Float32Array` of shape `[batchSize, 362]` (moves 0..360 on board + 361 for pass).
   - `values`: Array of `SingleEvaluationValue` per batch item:
     - `winrate`: Number in $[0, 1]$ (current player perspective).
     - `scoreLead`: Number (expected points ahead/behind).
     - `scoreMean`: Number.
     - `scoreStdev`: Number (typical range 10.0–20.0).
     - `utility`: Number in $[-1, 1]$.
   - `ownership`: Optional `Float32Array` of shape `[batchSize, 361]`.

---

## 3. Intentional Behavioral Corrections vs Legacy TypeScript Engine

The legacy engine in `src/engine/` had several historical limitations that caused divergence from upstream KataGo. The new core intentionally rectifies them:

| Area | Legacy TS Engine | Upstream Core v1.18.1 (`core-runtime`) | Upstream Justification |
|---|---|---|---|
| **Move History** | Truncated to the last 5 moves (`recentMoves.slice(-5)`). | Full untruncated move history and explicit initial stones tracked from move 0. | Planes 9–13 history and superko rules require full history. |
| **Ko Rule** | Simple single-stone ko only (`koPoint`). | Simple Ko, Positional Superko (Chinese, Tromp-Taylor), Situational Superko (AGA, New Zealand). | Upstream KataGo `BoardHistory` enforces superko via 64-bit Zobrist hashes. |
| **Suicide Legality** | Hardcoded unconditional error on any 0-liberty move (`playMove` threw `Illegal suicide move`). | Configurable via `rules.multiStoneSuicideLegal` (legal under Tromp-Taylor & New Zealand). | Upstream `cpp/game/board.cpp` allows multi-stone suicide when permitted by rules. |
| **Rules Presets** | Only `'japanese'`, `'chinese'`, `'korean'`. | 8 presets (`japanese`, `chinese`, `chinese-rulesist`, `korean`, `aga`, `tromp-taylor`, `new-zealand`, `stone-counting`) + custom clause parser. | Upstream `cpp/game/rules.h` supports full suite of rule variations. |
| **Scoring** | Basic territory heuristic. | Full Area & Territory scoring, Button Go (+0.5 pt), White Handicap Bonus rules (`WHB_ZERO`, `WHB_N`, `WHB_N_MINUS_ONE`). | Upstream scoring logic in `cpp/game/boardhistory.cpp`. |
| **MCTS Selection** | Basic UCB selection. | Upstream KataGo PUCT formula with dynamic First Play Urgency (FPU) reduction. | Upstream `cpp/search/searchpuct.cpp`. |

---

## 4. Migration Path for `src/engine/`

When integrating the upstream core into production:

1. **Adopt `CanonicalPosition` in Public API**:
   - In `src/engine/types.ts`, update `KataGoAnalyzeRequest` to use `CanonicalPosition` (or normalize incoming requests through `validateAndNormalizePosition()`).
2. **Switch Search Orchestration**:
   - Point `analyzeMcts.ts` or modern replacement to `SearchSession` from `src/core-runtime/search.ts`.
   - Wire the winning runtime adapter (`EvaluationBackend`) into `SearchSession`.
3. **Double-Buffering & Evaluation Pipeline**:
   - The dual-slot ping-pong evaluation lease mechanism (`EvaluationLease` in `src/engine/evaluation/batch.ts`) can be preserved: wrap each slot call into `EvaluationBackend.evaluate(batch)`.
4. **Export Public Contract**:
   - Re-export `src/core-contract/index.ts` from root `src/index.ts`.

---

## 5. Verification & Benchmark Instructions

To verify the integrated engine:

1. **Unit & Differential Tests**:
   ```bash
   npx vitest run test/core-upstream/
   ```
2. **Parity Tests**:
   ```bash
   npx vitest run test/parity/
   ```
3. **Visits/Second Measurement**:
   - Initialize `SearchSession` with the chosen GPU backend.
   - Run fixed-visit benchmark (e.g. 500 visits on 19x19 standard board).
   - Verify visits/sec, memory stability, and output sanity.
