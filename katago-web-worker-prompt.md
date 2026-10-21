# Worker AI execution prompt: optimize and organize KataGo Web

You are the implementation engineer for this task. You have no prior conversation context. Implement the specification included below; do not merely return another plan. Work incrementally and preserve exact output parity. Read this entire prompt before editing code.

## Assignment

Repository: `C:\JuneKim0712\katago-web`.

Improve measured KataGo visits/sec in Chrome/WebGPU and reorganize the implementation into clear, maintainable modules. Preserve existing public APIs and exact fixed-visit outputs under the same execution environment and frozen configuration. Keep TypeScript orchestration, TensorFlow.js inference, and the current WGSL kernels. A language/runtime rewrite is outside this assignment.

KataGo performs Monte Carlo tree search (MCTS). A visit selects a path, prepares Go-board features, evaluates a neural network, and backpropagates results. CPU board/feature work, GPU inference/readback, and reporting can each limit throughput. This implementation overlaps two evaluation slots. Changing batch membership, random draws, numerical operation order, or backpropagation order can change later search decisions even when individual differences appear tiny.

The full numbered specification below is part of this prompt. You do not need another chat, an external plan file, or the reviewing agent's memory. Its source observations are historical evidence, not proof that the current checkout still matches. Validate each affected function before implementing its proposed change.

## Operating constraints

- Work in the target repository, not the directory containing this prompt. Read applicable `AGENTS.md`, `.cursorignore`, package scripts, and repository status first. Respect ignored files; do not recursively read dependencies, generated output, logs, binaries, or model weights as source text. Read model bytes programmatically only when required for hashing/inference.
- Preserve existing uncommitted user work. Do not reset, clean, overwrite, or revert unrelated changes. If Git is unavailable, record file hashes and explain the limitation; do not pretend the working tree is clean.
- Do not spawn agents, create demos, make screenshots/recordings, or run computer-use/browser testing unless the user explicitly authorizes that activity. You may implement the optional browser harness; this prompt does not waive the browser-execution restriction. If Chrome qualification is blocked by that restriction, complete independent work and report the exact remaining command and required authorization. Never claim Chrome speedup without that measurement.
- Use the existing package manager and toolchain. Add only the development dependencies justified by the specification, when their implementation stage needs them. Do not upgrade numerical dependencies or install ONNX Runtime in the production package. Honor environment access/approval requirements; do not bypass restrictions.
- Do not publish, deploy, push, or modify external services. The specification's “commits” are small implementation stages. Create actual Git commits only if repository instructions or the user authorize them; otherwise keep clearly separable patches and a stage log.
- Use targeted tests while iterating. Run broader applicable checks once at final integration. Do not repeatedly run large suites for documentation-only changes.
- Do not ask for routine implementation choices. Continue authorized independent work when a particular stage is blocked. Stop dependent work when its required baseline or correctness gate is unavailable.

## Non-negotiable interpretation of parity

“Exact parity” means identical feature bytes, raw neural output bits, numerical results, selected paths, batch membership/order, random draws, and final result fields at the same completed-visit count on the same recorded environment/profile. It does not mean approximate agreement, the same top move, equal playing strength, or equality across different GPU drivers.

Faster execution under a fixed wall-clock deadline can complete more visits and produce a different result. Preserve stop/drain behavior and test deadlines with a scripted clock. Measure performance at fixed visits. Do not reduce output detail, search effort, model size, ladder limits, or ownership computation to obtain a speedup.

Existing correctness defects are separate from optimization. Do not silently fix output-changing defects, regenerate reference outputs, widen tolerances, or claim a corrected output is equal to the old one. If a correctness defect blocks qualification, describe the minimal repair and affected output contract; continue other independent stages. Acceptance of a new correctness baseline must be explicit.

## Execution workflow

1. **Establish the actual starting point.** Confirm the repository exists, inspect instructions/status, read the maintained implementation and relevant tests, compare the historical hashes below, and identify source drift. If the repository is missing, report the missing path instead of editing another project.
2. **Create a compact durable progress record.** Use `docs/optimization-progress.md`. Record initial source identity, completed/current/blocked stages, exact commands and outcomes, qualification profiles, and next actions. Keep it concise; never store model weights or secrets in it.
3. **Build the reference before optimizing.** Preserve an immutable copy/build of the actual starting implementation, including relevant uncommitted source changes. Record the copied files and hashes. Introduce random/clock seams as specified and prove the instrumented reference still matches the original behavior. The candidate must never generate its own expected outputs. A historical default-profile toggle is not a baseline.
4. **Execute section 22 in order**, using section 26's extraction-before-optimization discipline and section 25's tools only when needed. Each numbered stage can contain several small patches. Do not replace large files wholesale or implement all stages before testing.
5. **For each patch:** state the invariant, inspect the exact affected code, make the smallest coherent edit, run targeted checks, compare exact outputs where affected, and update the progress record. Keep cleanup separate from changed algorithms and resource lifetimes. If a candidate fails parity, diagnose it; remove only your failing candidate edits if it cannot be repaired.
6. **Qualify throughput separately.** Follow the fixed-profile paired benchmark procedure. Mark a candidate as unqualified if real-model/backend tests cannot run. Do not enable a gated optimization by default without its required proof. Purely mechanical cleanup still needs its relevant behavior checks.
7. **Handle unavailable prerequisites honestly.** Missing model files, unavailable WebGPU, permission failures, nondeterministic reference behavior, or absent Chrome authorization are blockers for the affected gate. Model-free tests may still run, but cannot substitute for GPU parity. Report exact missing inputs and commands; do not invent results or silently skip required checks.
8. **Finish with integration and a concise handoff.** Inspect the final diff, run applicable package checks, and report implemented stages, measured results with environment, exact-parity status, tests passed/failed/not run, remaining blockers, and links to changed documentation. A stage is complete only when its acceptance gates pass. Distinguish implemented-but-unqualified work from completed work.

## Context-window and decision rules

If context becomes short, update the progress record with the next concrete action and unresolved invariants before continuing. On resumption, read that record and the relevant specification sections. Do not restart completed stages or rely on recollection of numerical details.

When the specification proposes an optimization, first verify that the current code still performs the redundant work. If it no longer does, mark the proposal obsolete with evidence and move on. Do not introduce complexity to implement an already-solved idea. Prefer neutral, simpler cleanup over an unmeasured abstraction. Do not claim global optimality or promise a speedup factor.

The numbered specification contains review-time statements such as “no tests passed” and “no tools installed.” These describe the review, not your future results. Keep historical review evidence intact and record your new results separately. Relative paths in the specification are relative to `C:\JuneKim0712\katago-web` unless expressly stated otherwise.

## Full implementation specification

Everything below is included for execution. Start with sections 1–6, use section 22 as the stage checklist, apply section 26 throughout, and refer to sections 7–21 for each stage's exact logic and acceptance gates. Section 24 is the coverage checklist; section 25 supplies optional tooling. Do not skip the detailed invariants merely because the stage checklist is shorter.

---

# KataGo Web: exact-parity visit-throughput implementation plan

Reviewed 2026-10-20. Target: `C:\JuneKim0712\katago-web`.

This is an implementation specification, not an implemented optimization or a benchmark report. Optimize Chrome/WebGPU execution while retaining the current TypeScript API and numerical execution profile. Do not begin with a C++/Rust rewrite, FP16 conversion, or wholesale replacement of TensorFlow.js.

Code organization is an equal deliverable: separate board algorithms, feature preparation, search scheduling, inference resources, and worker transport so each can be understood and changed independently. Section 26 specifies the dependency rules, extraction procedure, cleanup sequence, and completion criteria. Maintainability refactors must preserve outputs and avoid measurable throughput regressions; they need not claim a speedup.

## 1. Scope and evidence

I read all 28 maintained files under `src`, all four test files, the benchmark scripts/configuration, fixture generator, example, README, package/build configuration, CI workflow, ignore file, and license. I parsed the complete native fixture JSON and inspected its metadata, cases, fields, and array lengths rather than reproducing thousands of numeric entries. It contains nine 19x19 positions with 361 policy entries and 361 ownership entries each; illegal policy entries can be null.

No target `.cursorignore` was present. Dependencies, build output, logs, lockfile, Git internals, and model binaries were excluded from source review. Those are not maintained implementation code. Dependency implementation details must be checked narrowly when implementing the backend adapter.

The code changed since the earlier review: `dispatchEvalBatchAsync` now calls `createInputTensors`; do not implement the earlier recommendation on the assumption that it bypasses this helper. The present helper is asynchronous and its reusable tensors are model-global.

Important source snapshot hashes, SHA-256:

```text
src/engine/analyzeMcts.ts  97E23C090B530C73FF47F114C228E85C142F10B5DAB933CAC408226AD716F517
src/engine/fastBoard.ts    99B98AA5A568250DADDF9B7179101A6CB73B471B3168F1C55DABFF8A54134155
src/model/modelV8.ts      2FAF22AD27AA0C1E62FA706498CB22655DA2DB14F9B2CCFEB62737BFA17B2A2E
src/worker/worker.ts      DBE3CB76EF8E2820E907A73783BC934486110817BAF9D6784EDA9A8785CFAD0A
src/worker/client.ts      E53582310DE95EDA2CFC0D2633642BDB58F4571CEA048A16723F70893F7EB220
```

Line references below describe this snapshot; function names are the durable anchors. Git working-tree status was unavailable in this execution environment, so these hashes, not an assumed clean commit, identify the reviewed implementation.

Attempted verification: `npm exec -- vitest run test/pipeline.test.ts`. Vitest failed before collecting tests because a sandbox temporary-file rename returned EPERM. No test passed in this review. No browser testing or model benchmark was run. Every performance opportunity below is a code-derived hypothesis until measured.

## 2. Define the non-negotiable parity contract

### 2.1 The contract that is achievable

For the same model bytes, board/history, resolved rules/options, random-number stream, backend configuration, actual batch shapes, search scheduling policy, and runtime/GPU environment:

1. Board state, ko, liberty/area/ladder maps, and input-feature bytes must be identical.
2. Raw neural-output Float32 bytes must be identical to the selected baseline backend.
3. JavaScript numerical results must match exactly, including signed zero where exposed.
4. Selected leaves, reservation counts, evaluation batch membership/order, and backpropagation order must match.
5. At the same completed-visit count, all final fields must match: visits, policy, ownership, ownership deviation, root statistics, candidate statistics/order, and PVs.
6. Existing output modes remain intact. Do not speed up a tree-ownership request by returning root-only ownership, or remove PVs to make a benchmark look faster.

Exact equality across arbitrary GPU vendors, drivers, browser/compiler versions, or different arithmetic graphs is not a promise this implementation can make. Establish per-environment baseline equivalence. WGSL permits floating-point evaluation behavior that makes algebraic equivalence insufficient for bitwise equivalence. See the [WGSL floating-point specification](https://www.w3.org/TR/WGSL/#floating-point-evaluation).

### 2.2 Fixed time and fixed output are different requirements

A faster engine running for the same time generally completes more visits and returns a different search result. Preserve exact results at a fixed completed-visit budget; measure how quickly it reaches that budget. At a fixed deadline, preserve the algorithm and stop/drain semantics, not an impossible promise of the same visit count and result.

For deterministic deadline/cancel regression tests, replay a scripted clock/cancel schedule at the same logical check points. For real-time performance tests, report changed visit counts honestly. Progress-message wall-clock timing also cannot remain identical after speeding up execution; compare message payloads at matched logical checkpoints.

If literal identical deadline-limited output is required, the engine must stop at the baseline visit count, sacrificing the extra work it could otherwise perform. Do not silently adopt that behavior.

### 2.3 Freeze these choices throughout the strict optimization lane

- Model content and parser math; no smaller-model substitution.
- Float32 neural storage and JavaScript Number/Float64 search statistics.
- Convolution/matmul/reduction algorithms and shapes.
- Existing rotary/RMS/projection flags, individually recorded. Production currently enables rotary and RMS paths and leaves projection fusion off.
- Batch size, partial-batch shapes, two-slot scheduling, collection-attempt multiplier, selection order, and reservation semantics.
- Number/order of root symmetries, leaf symmetries, policy optimism, root noise, and every random draw.
- `maxChildren`, ROI semantics, score utility constants, ladder limits, rules, and komi behavior.
- Float32 rounding points in scratch arrays, ownership accumulation, and policy mixing.
- API-specific defaults and result sorting, including ties.

Do not enable FP16, quantization, approximate activations, fused QKV, changed reduction trees, FlashAttention-style replacements, changed root sampling, wider batching, tree promotion, or asynchronous out-of-order backpropagation as part of this lane.

### 2.4 Existing defects are not permission to change output

Maintain two separately reviewed streams:

- **Strict refactor/performance stream:** exact comparison against current successful behavior, on a fixed profile and workload.
- **Correctness stream:** explicitly documented changes to currently incorrect behavior, each with an independent oracle and changed-output fixture. Never call these changes parity-preserving optimizations.

If a defective path cannot provide a reliable baseline, mark that path unsupported for strict qualification until its correctness change has been accepted and a new baseline recorded. Do not silently regenerate goldens or widen tolerances. The plan identifies these blockers; it does not authorize changing their outputs during optimization.

## 3. Findings that determine the implementation order

### F1. The parity suite is too weak for this request

`test/nativeParity.test.ts` creates a root and calls `getAnalysis`; its nine native comparisons do not run a multi-visit search. Native comparison uses tolerances, and the batch test also permits `1e-5` differences. The SHA-256 assertion checks model identity, not equality of results. Model-dependent suites skip if weights are absent. CI does not provision the model.

Keep those tests as native-compatibility checks. Add a separate exact baseline suite; do not relabel tolerant tests as bitwise proof.

### F2. Reusable input ownership and dispatch timing need attention

`analyzeMcts.ts:createInputTensors` (around 1425) awaits a dynamically detected `Tensor.write`. `modelV8.ts:ModelInputBuffers` owns one spatial/global pair and recreates it when batch size changes. Both pipelined slots can call it. If writes are supported, overlapping uploads can overwrite the other slot or dispose its tensors. Existing pipeline fakes force `supportsWrites: false`, so they do not cover this path.

The public TF.js Tensor API does not advertise this `write` method; it must not be assumed to work merely because an optimization flag is on. Verify the pinned runtime and report the path actually used. [TF.js Tensor API](https://js.tensorflow.org/api/latest/#class:Tensor)

Even the allocation fallback returns through an async helper. Consequently, `forward` is reached only after an `await`, while the caller synchronously collects/prepares the next batch. The comment saying GPU dispatch happens immediately is not guaranteed by this implementation. Fix host scheduling without changing which leaves are collected before backpropagation.

### F3. Several arrays are not actually retaining pooled objects

`collectBatch` resets `pathMoves.length = 0`, then attempts to reuse `pathMoves[pathIdx]`; truncation removed those entries. Jobs allocate an object and `path.slice()` per leaf. `playMove` returns an undo object per simulated move. `takeRecentMoves` truncates its destination, losing spare entries when a later job needs a longer history.

Use logical counts over persistent capacity, with separate storage for each in-flight job. Do not replace Float64 values with Float32 when moving to arrays.

### F4. Repeated ladder and traversal work is a direct CPU target

`prepareBatchFeatures` computes current, previous, and previous-previous ladders per sample, including repeated histories across siblings and root symmetry samples. Ladder code allocates `Int16Array(16)` in `getNumLibertiesAfterPlay`, and `Int16Array(2)` in `searchIsLadderCapturedAttackerFirst2Libs`. `findLibertyGainingCaptures` traverses a group with `findLibertiesIntoBuf`, then traverses it again to recover its length.

The ladder search has a 25,000-node budget and ordering-dependent early exits. Do not alter its search order or budget while eliminating these costs.

### F5. Model execution already contains intentional numerical boundaries

`rotaryWebGpu.ts` uses two passes to store rounded products before addition/subtraction. Preserve that boundary. `rmsNormWebGpu.ts` computes `(x * r) * weight`, whereas the generic path in `modelV8.ts` computes `x * (r * weight)`. These are not generally bitwise equivalent. Freeze the currently selected path; do not treat toggling the flag as a harmless performance switch.

`forward`, `forwardPolicyValue`, and `forwardValueOnly` duplicate head code. Per-block `tf.tidy` is useful, but each returned residual activation survives into its enclosing scope. Earlier block outputs can remain live until that scope exits. Audit ownership and release only activations proven dead.

### F6. Worker/client correctness gaps affect parity

- `client.ts:379` sends only the last five moves. `territoryKomi.ts:getNNKomi` explicitly needs full history for territory-rule move balance. Direct search and worker search can therefore disagree.
- Light progress uses `cloneBuffers: false`. When move ownership is requested, child-owned arrays can reach the transfer list and become detached inside the search tree.
- `worker.ts:521` captures interactive preemption as a boolean once. An interactive request arriving later does not change that boolean.
- Init messages lack a correlation ID, while the client can supersede one pending init with another. A late earlier response can resolve the wrong init.
- Same-URL init coalescing ignores backend preference; old promise handlers can clear newer initialization state.
- Mutable board globals can change after a transformer model's RoPE tables were constructed. The warmup probe is fixed at 19x19.

Handle these as correctness changes, separately from throughput claims.

### F7. Completed-result caching is not exact-result caching

`positionCache.ts` stores four scalar fields, hashes only a subset of position state into 32 bits, rounds komi to half points, and partitions by model name. `worker.ts` looks it up after creating/evaluating the root, and returns placeholder arrays and an empty candidate list. Writes mark requested visits rather than actual visits. A result with more visits is not the exact result of a lower-visit search either.

Do not use this cache to qualify strict throughput. Fix or version its contract independently. Feature caching is a safer first optimization.

### F8. Current performance numbers do not isolate the target

Performance tests use WASM/CPU and enable projection fusion, unlike worker production. They count root visits in a timer that starts after root evaluation. `run()` clamps visits to at least 16, but tests request 6 or 8. Feature microbenchmarks omit the expensive ladder/area preparation. Baseline JSON has no hardware/runtime/model hash. The standalone benchmark depends on `dist`, which can be stale.

The batch probe measures zero-input value-only inference, not policy/value/ownership search. Its localStorage cache is called inside a Worker, where this storage API is unavailable; errors are swallowed. Worker-compatible storage is required. [HTML worker interfaces](https://html.spec.whatwg.org/multipage/workers.html#the-workers-interfaces)

## 4. Target module boundaries

Keep these public entry points: `MctsSearch`, `analyzeMcts`, `KataGoModelV8Tf`, parser exports, board helpers, and `KataGoEngineClient`. Preserve their signatures initially. New internal files should have one concrete owner:

```text
src/engine/board/context.ts       dimensions, neighbor tables, traversal scratch
src/engine/board/moves.ts         play/undo core, compatibility wrappers
src/engine/board/liberties.ts     full/incremental liberty traversal
src/engine/board/area.ts          existing Benson/territory feature algorithm
src/engine/board/ladders.ts       existing bounded ladder algorithm
src/engine/features/encoder.ts   V7 encoding, symmetry placement
src/engine/features/cache.ts     bounded exact board-derived feature memoization
src/engine/evaluation/batch.ts    feature prep, submit/read/decode lifetime
src/engine/evaluation/types.ts    backend-independent arrays and leases
src/engine/search/tree.ts        Node/Edge types and optional arenas
src/engine/search/selection.ts   current PUCT/FPU/noise math
src/engine/search/expansion.ts   legality, policy softmax, top-child selection
src/engine/search/session.ts     MctsSearch facade implementation
src/engine/search/scheduler.ts   explicit serial vs two-slot scheduling policies
src/engine/search/report.ts      current output construction/PV/ownership
src/engine/search/utility.ts     unchanged score-utility/root-weighting math
src/model/parsedTypes.ts         parsed descriptors, no TF.js runtime dependency
src/model/tfjs/backend.ts        inference adapter and configuration fingerprint
src/model/tfjs/heads.ts          shared head functions, identical op order
src/model/tfjs/inputSlots.ts     slot-owned input resources
src/model/tfjs/outputReadback.ts exact copy/packing experiment
src/worker/runtime.ts           model/backend generation and lifetime
src/worker/protocol.ts          canonical request/result types
src/worker/serialization.ts     snapshot cloning and transfer ownership
```

Do not create empty abstractions in advance. Introduce each file when extracting real code in the steps below. Keep old modules as forwarding facades. Do not reorganize numerical code and change its algorithm in the same commit.

Apply the ownership and dependency rules in section 26 to these modules. The ordered commits in section 22 are delivery stages, not a requirement to combine extraction and optimization: split each affected stage into a behavior-preserving extraction commit followed by its independently measured optimization commit.

## 5. Step 01 — Freeze the reference and build exact comparisons

**Files:** tests, benchmark tooling, `visitSpeedConfig.ts`, the random/clock call sites in `analyzeMcts.ts`.

1. Capture a source-content manifest and exact dependency versions in the implementation branch. Keep an immutable baseline checkout/build outside the candidate package. Do not use the mutable `LEGACY_VISIT_SPEED` profile as the reference: it changes behavior and two flags are unused.
2. Record a resolved execution profile: model SHA-256, input/output channels, model version, board size, backend, all three fusion flags, batch size, two-slot policy, attempt limit, root symmetries, ownership mode, noise, rules, komi, ROI, maxChildren, reporting options, browser/runtime/GPU identifier, and backend flags.
3. Add `RandomSource.next(): number` internally. Production delegates to `Math.random` exactly as before. Both `Rand` and leaf-symmetry selection must use it. Preserve Gaussian spare handling, rejection-loop draws, and call order.
4. Test mode records/replays a finite random tape; fail if exhausted or if the candidate consumes a different number of values. A PRNG alone is insufficient unless draws/order are also compared.
5. Add an internal clock dependency defaulting to `getAnimationNow`. Fake clocks let deadline tests stop at the same logical checks. Do not add extra calls to the search clock from instrumentation.
6. Implement `assertFloat32BitsEqual` on Uint32 views respecting byteOffset/length. Compare shape, length, dtype, and every bit; log first mismatch with index and values. For JS numbers use Object.is and, for serialized baselines, DataView Float64 encoding. Reject unexpected non-finite values separately.
7. Store exact baselines in binary or lossless bit encodings. JSON decimal values alone lose signed-zero distinctions. Use hashes for quick mismatch detection, then compare contents for diagnostics.
8. Add optional test tracing: selected edge paths, pending collisions, accepted jobs, slot/batch membership, symmetry draws, effective komi, and backprop events. Trace only in parity runs; it is disabled during timing runs.
9. Compare final snapshots with all arrays and all candidate fields. Compare reported snapshots at matched committed batches as well.

**Tests to add:** `test/parity/floatBits.test.ts`, `test/parity/featuresExact.test.ts`, `test/parity/searchTrace.test.ts`, `test/parity/inferenceExact.test.ts`.

**Acceptance:** baseline vs itself is repeatable on the qualification environment; a deliberate one-bit mutation and a changed random draw both fail. If baseline is nondeterministic, stop exact qualification and identify why rather than adding tolerance.

## 6. Step 02 — Establish a representative, honest WebGPU benchmark

**Files:** `benchmarks/run-bench.mjs`, `performance-baseline.json`, all performance tests; add shared fixtures/config.

1. Make backend explicit. For a WebGPU run, failure to initialize WebGPU is a failure, not silent WASM fallback. Keep WASM tests as separate compatibility tests.
2. Reuse production model options: projection fusion remains false unless that is the explicitly selected baseline profile. Log the options.
3. Benchmark three distinct intervals:
   - Cold load: fetch/decompress/parse/model creation/warmup, separately reported.
   - Root creation: feature preparation, root inference, utility-table setup.
   - Search: timer around `run`; throughput is `(visitsAfter - visitsBefore) * 1000 / elapsedMs`.
4. Add a fourth user-facing latency metric from request start to final snapshot. Never count a completed-result cache hit as newly executed MCTS visits.
5. Use fixed targets at least 16, matching the existing clamp. Start with 128 visits for the small model and 32 for b10c384; increase targets once until each trial runs at least one second on that machine, then freeze the same target for baseline/candidate. If a trial hits the time ceiling before its target, exclude it from fixed-visit comparison and report it separately.
6. Use legal fixtures for opening, middle-game, capture/ko, long ladder, Chinese area computation, pass/conservative-pass, and handicap/history. Run ownership none/root/tree separately. Use exact same input history on both sides.
7. Warm two throwaway searches for each fixed configuration. Run at least seven paired baseline/candidate trials, alternating order to reduce thermal/order bias. Keep trial searches fresh; do not include useful tree state from warmup.
8. Report each pair, median speedup, p95 search/request latency, completed visits, allocated/reused slots, CPU feature time, and memory high-water marks. Require zero parity mismatches first. As an initial promotion rule, require median end-to-end improvement of at least 3%, positive improvement in at least six of seven pairs, and no repeatable >5% regression in another required workload. If noise exceeds those margins, gather more pairs rather than declaring victory.
9. Separate instrumentation from timing. Coarse optional counters: collect/selection, move simulation, liberty updates, current/history ladders, area, encoding/symmetry, host forward dispatch, output-await span, decode, expansion/backprop, reporting. CPU/GPU overlap means their durations cannot simply be summed.
10. Initially use wall-clock phase timers. GPU timestamp queries are useful only when inserted around actual submitted workload on the same device/encoder; an empty encoder measures nothing useful. Do not block search on `onSubmittedWorkDone` every batch.

**Qualification:** Node WebGPU/Dawn is useful for automation, but is not proof of Chrome performance. A final Chrome worker benchmark with the same profile is required before calling the change a Chrome improvement. This review did not run a browser; the harness and later qualification are implementation work, not a demo or screenshot exercise.

**Acceptance:** saved results include enough environment and model identity to reproduce them; root visits are not overcounted; missing models fail the dedicated qualification job. No speedup target is claimed before this baseline exists.

## 7. Step 03 — Isolate evaluation state and consolidate decoding

**Files:** `analyzeMcts.ts` functions `getEvalScratch`, `prepareBatchFeatures`, `evaluateBatch`, `dispatchEvalBatchAsync`, `processNeuralOutputs`.

1. Extract the existing `processNeuralOutputs` into `evaluation/batch.ts` unchanged. Replace the duplicate decoder in `dispatchEvalBatchAsync` with this function. Preserve `base + (opt - base) * mix`, Float32 stores, score stride 4, and sample order.
2. Give each search an `EvaluationContext` with two `BatchSlot` objects. Root evaluation uses an exclusively leased slot before the run begins. Do not keep the buffers in module-global pairs.
3. Each slot owns spatial/global arrays, liberty/area maps, symmetries, policy/pass mixing scratch, and any input tensors. Shared temporary ladder arrays are allowed only within synchronous feature encoding; retained batch views must be per slot.
4. Define the lifecycle: `free -> preparing -> submitted -> outputs-ready -> consumed -> free`. Output-ready is not free: mixed policy and maps may still alias the slot until backpropagation consumes them.
5. Use an exact shape key `(boardSize, actualBatchSize, inputChannels, globalChannels, modelGeneration, backendGeneration)`. Capacity can grow on the CPU; the tensor's logical batch shape must remain the actual batch length. No padding to a larger batch.
6. Reject two simultaneous `run()` calls on one SearchSession. Different sessions must have independent scratch. Serialize access to global TF.js backend changes via the worker runtime.
7. Make read cleanup symmetric in root and pipelined evaluation. On output-read failure, await settlement of all started reads before releasing their resources; preserve the first error. Catch synchronous forward/feature errors as well.
8. Backprop/discard reservations exactly once. Track the uncommitted job index if an exception occurs halfway through a batch; never decrement already-committed paths twice. Mark a partially failed search unusable if its tree cannot be restored, rather than resuming corrupt state.

Proposed internal shape:

```ts
interface EvaluationLease {
  readonly slot: 0 | 1;
  readonly sequence: number;
  readonly result: Promise<readonly NeuralEval[]>;
  releaseAfterConsumption(): void;
}
// submit synchronously starts the existing forward/read path when inputs are ready.
// result completion does not release scratch ownership.
```

**Tests:** delayed reads with distinct A/B inputs; batch sizes 4 then 1; one rejected read while others remain pending; synchronous forward exception; cancellation with both slots submitted; resumed search after a clean cancel; two independent sessions. Check no pending reads, no negative reservations, no retained transient tensors.

**Acceptance:** exact trace and result equivalence on successful baseline paths; defects on failure paths are explicitly classified as correctness changes.

## 8. Step 04 — Restore real CPU/GPU overlap without altering search order

**Files:** `createInputTensors`, `dispatchEvalBatchAsync`, `ModelInputBuffers`.

1. Make the ordinary tensor-creation path synchronous. Create the two tensors, invoke the same model forward method, and start the same output reads before returning the evaluation promise.
2. Do not insert an unconditional `await` before `forward` when all input setup is synchronous. For any genuinely asynchronous upload adapter, expose upload/submit distinctly so the scheduler knows whether submission has started.
3. Preserve this logical schedule exactly: collect A, prepare A including its random symmetry draws, submit A, collect B, prepare/submit B, await/commit A, then continue. Do not commit A early merely because it finished while collecting B.
4. Keep FIFO backpropagation even if B's promise resolves first. Keep the same attempt budget, virtual reservations, pending-leaf handling, retry rule, and partial-batch sizes.
5. Initially use the known allocation path. Treat `Tensor.write` reuse as a separate capability, disabled in strict qualification unless verified. If enabled, allocate one independently owned pair per slot, not one pair per model.
6. Before adding persistent GPU input buffers, inspect the exact installed TF.js backend API in isolation. Put any supported integration behind `tfjs/inputSlots.ts`, pinned to the qualified version. Never mutate TF.js private buffers without also honoring its reference counts and cached CPU values.
7. Test disposal/model switching only after both leases settle. Backend generation changes invalidate all slot resources.

**Acceptance:** event instrumentation proves forward A starts before synchronous preparation of B, while the accepted-leaf trace is unchanged. Benchmark with tracing off. A new submission timing may alter cancellation timing in real time; validate cancellation at logical checkpoints, not identical wall-clock delivery.

## 9. Step 05 — Replace ineffective object reuse with actual retained storage

**Files:** `MctsSearch.run`, `takeRecentMoves`, `fastBoard.ts:playMove/undoMove`.

1. Replace the working `pathMoves` array with `pathMoveStorage` plus `pathMoveCount`. Never shrink storage while traversing. Set the existing object fields at index `pathMoveCount`, then increment the count.
2. Update internal helpers to accept `(storage, count)`. Only inspect `[0, count)`; old entries beyond count are not live history. Keep a public array wrapper for existing callers.
3. Add `JobSlot[2][batchCapacity]`. Each owns its persistent path array, recent-move records, typed-array views, and a logical count. Grow capacity outside the hot collection loop where possible. Clear Node references when the search is disposed.
4. Replace `path.slice()` with a copy into the job's persistent path storage. Do not point every job at the single traversal scratch array. Backprop/discard loops use the saved path count.
5. Cache subarray views per slot/job after growing a backing buffer; recreate all dependent views on growth. Do not build new views on every leaf if capacity and board size are unchanged.
6. Add an internal `playMoveInto` variant that writes `koPointBefore` and `captureStart` into caller-owned depth arrays. Preserve the exact capture traversal and legality checks. Existing public `playMove` wraps it and returns a fresh snapshot as before.
7. Use Float64/Number for arithmetic state, Int32 for capture offsets/counters, and integer move storage with a validated board-size range. Do not use Uint16 visit counters; the ceiling is 2,000,000.
8. Retain a `recentMoveStorage` capacity of five for each job; set its logical length separately instead of deleting records when short histories occur.
9. Hoist immutable history plane/global index arrays; derive alternating expected player by index rather than allocating `[opp, pla, opp, pla, opp]` every sample. Preserve the break on the first player mismatch.

**Tests:** depth grows beyond 64; long-short-long paths; pass moves; repeated captures; variable batch size on successive runs; distinct A/B slot history; exact undo and input bytes.

**Acceptance:** after warmup at fixed capacity, job/path/recent-move/undo record creation counters stop growing. Search traces and all exposed results remain identical.

## 10. Step 06 — Remove redundant simulation cleanup and ladder traversals

### 10.1 Avoid undo work that is immediately overwritten

Currently each new selection begins with `sim.stones.set(rootStones)` and resets ko. Therefore full restoration of the scratch simulation after harvesting a job is unnecessary if nothing reads it before that next reset.

1. Continue undoing the last one/two moves needed to materialize previous and previous-previous boards. Preserve their ko snapshots and depth-1/depth-2 special cases.
2. After copying those boards, do not undo the remaining ancestors merely to reset a scratch board that will be overwritten.
3. Explicitly reset `captureStack.length = 0`, traversal counts, and ko before the next traversal. Apply the same rule to a pending-leaf collision.
4. Audit every exit and exception path; the scratch simulation must not escape to jobs. Jobs own copies or immutable root references.
5. Compare generated histories, capture stack balance, liberty maps, and selected paths against the baseline. Keep the old cleanup in the reference oracle only.

### 10.2 Eliminate tiny allocations and duplicate group walks

1. Put the 16-entry temporary liberty list and two-entry attacker liberty list into `LadderScratch`. Maintain separate fields because their lifetimes can overlap. Read `move0`/`move1` into scalars before invoking deeper search.
2. Change internal `findLibertiesIntoBuf` to also expose the group length from the traversal it already performs. Return it through a dedicated scalar result record or output parameter, without another traversal.
3. In `findLibertyGainingCaptures`, mark `LADDER_OPP_GROUP_SEEN` directly from the just-collected group. Remove the second `collectGroupAndLiberties` call. Preserve the order of the liberty list and captured-group marking.
4. Keep the 25,000-node budget, 8192 move buffer, neighbor order, defender/attacker order, and all early exits unchanged.
5. Do not generalize group collectors with one globally shared mutable result object unless every caller copies scalar fields before nested group searches overwrite it. Several callers retain `g.groupLen` across nested work; a naive shared result object breaks them.

**Acceptance:** all ladder masks/working moves and feature bytes match, including budget-exhaustion cases; allocation/group-traversal counters fall. Retain only measured end-to-end wins.

## 11. Step 07 — Add exact, bounded feature memoization

**Files:** `prepareBatchFeatures`, `fastBoard.ts` ladder/area functions; new `features/cache.ts`.

Start with a per-search cache, not IndexedDB and not a neural-result cache.

1. Cache only deterministic derived data: laddered-stone masks, current-player working-move masks, and Chinese area maps. Retain the existing incremental liberty path.
2. Keys include board size and exact stones; ladder entries also include ko and working-move entries include current player. Include algorithm-version identity and suicide legality for area. Never key only on a mutable typed-array object identity.
3. Use a fast hash for bucket selection, followed by byte-by-byte comparison of stored immutable stone bytes and all scalar key fields. Hash collision must yield a miss, never a wrong hit.
4. Preserve current fast-path eligibility: consult the cache only where the baseline would compute the corresponding map, or prove the baseline's skipped output is identical. Incremental liberty maps can contain stale values at empty points; do not accidentally make that separate defect part of this optimization.
5. A full current-position ladder evaluation can populate the laddered-only entry for the same stones/ko. The current implementation's laddered mask is independent of currentPlayer; currentPlayer only filters working moves. Assert that property in tests before sharing the result.
6. On a hit, copy cached data into caller-owned scratch. Never hand out writable views into cache storage, and never store views into a leased batch slot.
7. Resolve repeated current/previous/pre-previous boards within a sample first. Then share across samples and across root symmetries. Keep feature generation and random symmetry draws in the original sample order.
8. Start with an explicit 8 MiB per-search byte budget, accounting for copied boards and map buffers, plus a bounded entry count. Evict least-recently-used entries. Record hit/miss/eviction and saved-computation counts. Tune the budget only after memory and throughput measurement.
9. Empty-board/no-ladder fast paths should avoid hash/copy overhead. Clear caches when context/rules-algorithm version changes or the session is disposed.

**Tests:** forced hash collisions; same stones/different ko; player-swapped working moves; history aliases; captures and passes; cached zero masks; two root symmetries; mutate source arrays after insertion; eviction; exact cached/uncached search tapes.

**Acceptance:** equality is byte-for-byte, cache residency stays bounded, and hash/copy overhead does not regress ladder-free workloads beyond the promotion limits.

## 12. Step 08 — Write symmetry-transformed features directly

**Files:** `featuresV7Fast.ts`, `prepareBatchFeatures`, symmetry map helper.

Currently a nonzero symmetry fills `spatialScratch`, then copies all 22 planes through a second board loop.

1. Add an internal encoder taking an optional precomputed position map. Public `fillInputsV7Fast` delegates with identity mapping.
2. Clear the destination slice exactly as before. For original point `p`, write features at `(symmetryMap[p] * 22) + channel`.
3. Map the ko point and each history move through the same map. Keep pass/global features untransformed.
4. Read stones/liberties/ladders/area in original board coordinates. Only destination coordinates change.
5. Keep `global[5]` komi arithmetic and Chinese parity-wave arithmetic unchanged, with the same Float32 assignment points.
6. Use the existing symmetry numbering, including mirror-before-rotation convention. Do not canonicalize positions or change which symmetry is selected.
7. Preserve output zeroing, including planes 7/8/20/21 that are unused by this encoder.

**Tests:** all eight symmetries, both players, all supported rules, ko, pass histories, and conservative-pass suppression. Compare every Float32 bit against old encode-then-permute output. Verify encode and inverse policy/ownership mapping compose correctly.

**Acceptance:** remove the extra full feature copy and `spatialScratch` only after exact equivalence passes; benchmark symmetry 0 separately so the common identity case is not slowed down.

## 13. Step 09 — Scope board state without changing rule algorithms

**Files:** `fastBoard.ts`, `scoreValue.ts`, `featuresV7Fast.ts`, `utils/ko.ts`, model RoPE setup.

1. Introduce immutable `BoardGeometry {size, area, passMove, neighborStarts, neighborCounts, neighbors, symmetryMaps}`. Preserve neighbor ordering: left, right, up, down.
2. Introduce mutable `BoardScratch` containing every visited/stamp/group/region/ladder buffer currently global. Initially migrate whole functions unchanged, one subsystem at a time.
3. Keep existing exported board helpers forwarding to a default context for backward compatibility. New SearchSessions retain their own context and never consult mutable global dimensions after construction.
4. Keep the byte-board representation. It is not a bitboard; converting to bitboards or union-find group tracking is a separate algorithm rewrite with a much larger proof burden.
5. Capture model geometry when building RoPE; either qualify per-size model resources or reject a mismatch explicitly. Do not use 19x19 RoPE tables for a differently shaped tensor.
6. Characterize current behavior for 9/13/19 and unsupported sizes before changing validation. Int16 move storage and unbounded `setBoardSize` cannot be assumed compatible for arbitrary input sizes.
7. Keep score-value tables per geometry, lazily created. Do not reset/rebuild them every time another context is used.
8. Add a deliberate stamp rollover rule with tests near Int32 overflow. This repairs a long-run defect and must be recorded as such, although normal-range arithmetic/output is unchanged.

**Tests:** alternating contexts; simultaneous independent root evaluations; capture/undo properties; full vs incremental liberties on occupied points; area maps including seki/pass-alive cases; model shape mismatch.

**Acceptance:** no numerical or rule-policy changes for qualified sizes. Mixed-board-size correctness is a distinct added guarantee; do not claim it existed previously.

## 14. Step 10 — Consolidate search while preserving both exported behaviors

**Files:** `analyzeMcts.ts:MctsSearch`, separate `analyzeMcts`, selection/expansion/report functions.

Do not simply replace `analyzeMcts` with `MctsSearch.create/run/getAnalysis`. They currently differ in ways that change results:

- Serial batches versus two-slot prefetched batches.
- Root-symmetry defaults, collection-attempt behavior, ownership handling, and conservative-pass options.
- Legacy report includes only visited children; class report can add unvisited policy candidates.
- Tie sorting and top-K selection/report construction differ.

Implementation sequence:

1. Extract pure shared math and types without changes: utility, selection, expansion, symmetry transforms, decode, and root weighting.
2. Add explicit internal `SchedulingPolicy = serial | prefetchedTwoSlot`, plus a report compatibility mode. Each public entry point retains its current policy/defaults.
3. Route both through shared feature/evaluation lifetime machinery, but preserve serial dispatch/commit barriers for the legacy API.
4. Extract root initialization with parameters for every existing behavioral difference; do not rely on newly inferred defaults.
5. Share the traversal and backprop implementation only after exact traces match separately for each public API. Keep behavior-specific wrappers rather than forced equivalence between APIs.
6. Keep selection's per-edge summation order. Do not replace `policyProbMassVisited` with an incrementally accumulated floating sum: its summation order changes as children are first visited.
7. If caching selection-derived scalars, key them by a revision updated on every child visit, reservation, discard, and expansion. Recompute using the original edge order. Leave caching out if invalidation costs exceed saved work.
8. Preserve `expandNode` min-slot replacement semantics under tied priors. Do not substitute a generic heap/top-K implementation unless it reproduces both selected children and ordering on ties.
9. Rename/remove inert flags `unifiedBatchPrep` and `interactiveCache` after confirming no external deep-import consumers rely on them; otherwise deprecate them. Snapshot active options per session instead of reading mutable process-global flags mid-search. Preserve existing defaults in the compatibility facade.

**Acceptance:** each public API matches its own immutable baseline. Fewer duplicated code paths is the goal; changing the legacy algorithm is not required for this task.

## 15. Step 11 — Reduce report allocation without changing report contents

**Files:** `getAnalysis`, `averageTreeOwnership`, `buildPv/getPvForEdge`, worker serialization.

1. Keep the existing top-K limit and PV cache. They are already useful optimizations.
2. Add reusable internal child-stat records and top-move scratch capacity. Reset `weightAdjusted` on every call because noise pruning mutates/sorts records. Return detached snapshots, not reusable mutable report arrays.
3. Preserve each caller's comparator and tie behavior. Root weighted statistics must consume the same child sequence, including the same stable policy sort in noise pruning.
4. In `averageTreeOwnership`, replace per-recursion child arrays with two edge scans: first compute the two weight sums in existing order; second recurse over the same children in that order. Preserve when selfProp is incremented and when the parent map is accumulated.
5. If recursion depth is a risk, use an explicit DFS frame stack, including an enter/children/exit phase. Keep the same child-before-parent accumulation order and Float32 stores in `out`/`outSq`. Do not accumulate into Float64 then round once.
6. Keep existing refresh-interval semantics for compatibility. A stale final ownership snapshot when throttling is requested is a separate correctness decision; do not silently change it in a reporting speed patch.
7. For worker snapshots, copy every nonempty array that aliases search-owned state before transferring. In particular, light mode's per-move ownership must be copied even though root policy/ownership arrays are empty. This fixes a transfer-lifetime defect, not an arithmetic optimization.
8. Test repeated `getAnalysis` calls and consumer mutation of returned arrays/PVs. Establish which aliasing is public behavior; use explicit internal immutable snapshots rather than accidentally expanding external mutability.

**Acceptance:** at the same tree revision, full/light payload contents remain compatible; reporting allocates less internal scratch; transferred arrays never detach live tree buffers.

## 16. Step 12 — Consolidate model heads and tighten activation lifetimes

**Files:** `modelV8.ts`, new `parsedTypes.ts`, `tfjs/heads.ts`.

1. Move parsed descriptor types out of `modelV8.ts` into `parsedTypes.ts`; re-export the existing type names. This removes the parser's type coupling to TF execution without changing runtime model parsing.
2. Extract `forwardPolicyHead(trunk)` and `forwardValueHead(trunk, includeOwnership)`. Preserve the exact tensor operation sequence and shapes, including score channels being sliced after the existing matmul/bias, not by changing the weight matrix dimensions.
3. Keep public `forward`, `forwardPolicyValue`, and `forwardValueOnly` signatures as wrappers. Avoid computing unused heads, exactly as the current wrappers do.
4. Keep outer synchronous `tf.tidy` scopes and existing per-block scopes. Never make the tidy callback asynchronous.
5. Audit old trunk activation liveness. Refactor the block runner to own its incoming trunk activation explicitly; after the next activation has been produced, dispose the previous activation only if it is no longer used by a residual path or outer nested block.
6. For nested bottlenecks, retain the outer residual input until the final addition. Do not dispose caller-owned spatial/global tensors, weights, or returned outputs. Write an ownership diagram in code comments for the handful of nested cases.
7. Verify aliasing when an operation returns its input (identity activation). Dispose by unique tensor identity and documented ownership, not by blindly disposing every local variable.
8. Make constructor failure cleanup explicit: if partial weight construction throws, dispose all tensors already created. Preserve weight bytes and BN merge arithmetic; this is lifetime robustness.
9. Keep projection fusion disabled in the production-equivalent profile. Existing benchmark enablement must be removed from that profile, not copied into production.

**Tests:** ordinary/gpool/nested/attention/FFN blocks; all activation kinds; all three head modes; failures halfway through construction; repeated forwards and disposal; exact per-layer and output bytes.

**Acceptance:** same GPU arithmetic graph and results, lower retained-tensor high-water mark where measurable, no premature disposal, no memory growth after repeated warmed calls.

## 17. Step 13 — Add copy-only GPU output packing as a gated experiment

**Files:** new `tfjs/outputReadback.ts`, inference adapter.

This is a concrete WebGPU optimization candidate that avoids changing arithmetic. It still requires measurement because extra copies may cost more than they save.

1. Preserve policy, pass, value, score, and ownership computations exactly.
2. Define a flat batch layout in elements: policy `B*A*P`, pass `B*P`, value `B*3`, score `B*4`, optional ownership `B*A`. Compute offsets from actual B/A/P, not hardcoded 361/2.
3. Preferred implementation, if supported safely by the pinned adapter: copy existing output GPUBuffer ranges into one packed GPU buffer, then one staging buffer/readback. Use byte copies, not float arithmetic or CPU postprocessing on the GPU.
4. If buffer access requires unsupported TF internals, first test a TF.js reshape/concat adapter behind the same interface. Count its extra kernels/copies; do not assume concat is free.
5. Read one packed Float32Array; slice views using offsets and pass them into the unchanged CPU decoder. The lease owns the backing readback until consumption.
6. Preserve all policy channels in the initial implementation. Dropping unused channels can be a later exact-copy specialization after proving indexing for 1/2/4-channel models.
7. Use distinct staging resources for overlapping slots. Do not reuse a mapped buffer, and do not destroy source tensors before queued copies have valid ownership.
8. Preserve NaN/signed-zero bytes for copy tests, while keeping production model-output finiteness checks separate.
9. Keep the existing multi-read path as fallback. Enable packing only on qualified runtime/backend configurations when exact bytes match and end-to-end performance improves.

**Acceptance:** all output bytes identical on every required batch/ownership/model shape, correct partial-batch offsets, safe cancellation, and a measured visit/sec improvement. If packing only reduces API call count without improving runtime, do not ship it.

## 18. Step 14 — Optimize score-table startup without approximating values

**File:** `scoreValue.ts:initScoreValueTables`.

For 19x19, the table has 842*421 cells, each with 101 integration iterations: about 35.8 million iterations. It is initialized during the first utility computation, before normal search timing. This is chiefly startup/root latency, not sustained visits/sec.

1. First expose and measure its initialization time separately; do not misattribute it to model inference.
2. Keep mean/stdev indexing, interpolation, normal-PDF samples, arctangent precomputation, and Float64 arithmetic unchanged.
3. Candidate A: compute invariant `wSum` once using the same 101 additions in the same order; reuse that exact Number for every cell. The per-cell numerator loop remains unchanged.
4. Candidate B: allocate the same Float64 table plus a Uint8 initialized bitmap; compute only the four cells demanded by interpolation. Each missing cell executes exactly the former cell's numerator loop in the same order, storing the identical quotient. Share PDF/arctangent precomputation per geometry.
5. Test lazy and eager values at every cell for 19x19 in a dedicated test, and boundary/interpolation inputs for each supported geometry. NaN must not be used as the only initialized sentinel.
6. Measure cold root and long searches. Lazy lookup branches may regress warmed throughput; retain eager computation if the tradeoff fails the goals. Prewarming merely moves cost and must be described that way.

**Acceptance:** bit-exact utility results and honest separate cold/warm measurements. Do not ship a precomputed table generated under a different math runtime as universally exact without verification.

## 19. Step 15 — Worker/runtime refactor and separately tracked correctness repairs

**Files:** `worker.ts`, `client.ts`, `engine/types.ts`, `resourceProfile.ts`, caches.

First extract a runtime object owning model/backend generation, search, queue, and slot resources. Preserve serial model execution, latest-request semantics, and existing fallback support in compatibility mode. A second worker with a second model copy is not part of this plan.

Implement the following as explicit correctness commits with their own baseline decisions:

1. **History:** send full history initially. Later send a validated territory move-balance summary plus last five moves only if setup-stone semantics are represented explicitly. Never infer total move balance from the last five moves. Test >5 moves, captures, passes, handicap placement, both territory rules, and worker/direct equality.
2. **Transfer ownership:** copy live move-ownership arrays before transferring light updates; test subsequent updates and final tree ownership after transfer.
3. **Dynamic preemption:** compute `analysisGroup !== 'interactive' && interactiveToken !== tokenAtEnqueue` inside `shouldAbort`, not once before the search. Test background cancellation when interactive input arrives during a readback.
4. **Init correlation:** add monotonically increasing init request IDs and worker generation IDs internally; only the matching pending entry may resolve/reject. Promise completion handlers clear state only when their generation is still current. Coalesce by model identity plus resolved backend, not URL alone.
5. **Model geometry:** invalidate/rebuild shape-dependent RoPE resources on a supported board-size change. Keep model/board-size mismatches visible; do not silently continue with incompatible tables.
6. **Probe storage:** move persistence to IndexedDB or an explicit main-thread storage service. Key by model digest, backend/runtime fingerprint, geometry, head mode, and probe version. Store validated finite integer sizes. A cached new batch selection can change search output; strict qualification always uses an explicit frozen batch profile.
7. **Device loss:** a lost device is not repaired by simply retrying a smaller batch against the same model/device. Stop that probe sequence, dispose/recreate the backend/model generation or report failure. In a WebGPU-only qualification run, no CPU fallback is permitted. Existing application fallback remains a separately declared behavior.
8. **Head-complete warmup:** warm the exact policy/value/ownership profile and actual shapes outside timed search. Keep batch-selection decisions fixed in the strict lane; redesigning the auto-tuner is a separate policy change.
9. **Resource accounting:** compute actual stored weight bytes from descriptor arrays, avoiding double-counting aliases; separately estimate activations and two in-flight batches. Do not change model tiers/default ownership based on the new estimate in the parity-preserving commit.
10. **Final snapshot freshness:** characterize ownership-refresh throttling. If final output should always be fresh, implement that as a documented behavior correction rather than changing it inside report pooling.

**Cache correction specification:** the current scalar cache must not satisfy a full analysis request. For an exact full-result cache, require the exact resolved input/profile and exact completed target visits, including RNG seed/tape identity when applicable. Store the full immutable payload. Do not reuse a greater-visit result for a smaller target. Do not serve nondeterministic/time-limited searches as equivalent fresh runs. Compare exact serialized keys after hashing. Store actual visits, version records, and avoid late LRU touches overwriting newer results. Check before root evaluation only after these semantics are established. This reduces repeated-request latency; it is not increased engine visit/sec.

## 20. Step 16 — Parser, package, and documentation cleanup

These are necessary maintenance tasks, but not primary steady-state throughput work.

1. `binModelParser.ts`: bound reads against `data.byteLength`, not the larger backing ArrayBuffer. Validate finite nonnegative integer counts and multiplication overflow. Preserve valid-file Float32 values and BN merge arithmetic. Tests use synthetic tiny byte buffers, including nonzero byteOffset and truncated views.
2. Keep current defensive copying of binary floats initially. A zero-copy view can be unaligned and can retain the entire decompressed model; do not call it a guaranteed memory improvement. Any later aligned-view path must track backing-buffer lifetime and exact bytes.
3. `loadModelV8.ts`: preserve V8–V17 parsing and unsupported metadata rejection. Add shape validation against the actual 22 spatial/19 global implementation, supported trunk norm modes, grouped norm semantics, attention dimensions, and head channels. Unsupported-model rejection is a compatibility change to document, not a silent approximation.
4. `modelCache.ts`: reset a rejected database-open promise so later calls can retry, handle version changes, and retain explicit blob URL revocation. Do not wire this cache into search results; its job is model assets.
5. `package.json`/build: emit the advertised ESM/CJS files through a deliberate build pipeline, or remove unsupported CJS exports as an explicit packaging change. `tsc` ESNext alone does not emit `index.cjs`.
6. Emit a real worker JS artifact and reference its emitted URL; do not leave a published `client.js` looking for a nonexistent `worker.ts`. Ensure worker dependencies are bundled/resolvable. Keep browser worker loading separate from Node import behavior.
7. Normalize internal emitted import extensions (`gpuError`, `territoryKomi` feature import, parsed-type import) and verify package imports from a clean consumer fixture. Use a package-content smoke check, not a UI demo.
8. Unify duplicate public/internal result type declarations through aliases while preserving exported fields. Do not drop fields merely because the simplified `types/index.ts` interface lacks them.
9. Correct README examples to actual signatures: `init(modelUrl, backend, modelByteSize?)`; `analyze` needs `modelUrl` and `moveHistory`; progress exposes `percent`, not `progress`; `MctsSearch.create` currently requires `wideRootNoise`.
10. Replace the bitwise-native-parity claim with the exact guarantee actually tested. Keep native numerical tolerances documented independently. Correct the bitboard description: the current board is a byte array with traversal scratch.
11. Fix benchmark/example visit/sec calculations to actual visit deltas; remove unused imports/variables only after typechecking. Keep model default choices and license unchanged.

## 21. Verification matrix and promotion procedure

### Fast tests, required for every relevant change

- Board play/undo for legal captures, ko/pass, merge/split situations; no stale capture-stack entries.
- Feature byte equivalence for all eight symmetries, rules, players, histories, and conservative-pass settings.
- Fake evaluator with varied outputs, not only uniform zero logits; deterministic selected paths at fixed batch size.
- Two-slot lifetime, partial batches, write-capability branch, failures, cancellation, and exactly-once reservations.
- Float32/Float64 exact comparator self-tests and random-tape/clock tests.
- Serializer transfer tests using owned versus tree-owned arrays.
- Cache collisions, key completeness, memory bounds, and mutation isolation.

### Model-backed exact tests, required before a performance candidate is promoted

- Every existing native fixture plus targeted long history, pass, ladder budget, same-board/different-ko/history cases.
- Root symmetries 1/2/4/8 as separately frozen profiles; no comparison that assumes changing their count preserves outputs.
- Actual batch sizes 1/2/4/8 and partial tails, subject to device limits. Compare old vs new at the same shape. A separate batch-invariance diagnostic does not replace this.
- Ownership none/root/tree; candidate ownership true/false; full/light reporting and PV lengths 0/1/15.
- Multiple supported architecture families: ordinary/gpool, nested bottleneck, and transformer. Use small synthetic descriptors for unit branches and real known models for end-to-end qualification. Missing architecture coverage means the optimization remains gated off for it.
- Independent exact comparisons for class API, legacy API, and worker protocol; no assumption they already behave identically.

### Independent native compatibility

Preserve current nine native fixtures and their tolerances; never regenerate them to make a candidate pass. Add fixtures through the native generator only with recorded engine version, model digest, rules, symmetry, and configuration. Their printed decimal values are not a lossless bitwise reference for WebGPU arithmetic. Native root agreement does not prove native MCTS equivalence.

### Per-change promotion

1. Record the change's intended saved work and whether it changes any output contract.
2. Run only the relevant fast tests while iterating.
3. Run same-environment exact baseline tests for affected operations and fixed-visit searches.
4. Run native compatibility when features, parsing, inference, or postprocessing changed.
5. Run paired performance trials on the frozen profiles, without detailed trace instrumentation.
6. Require exact parity, bounded memory, no tensor/lease leak, and the performance criteria in Step 02. Keep maintainability-only refactors if neutral and simpler, but do not label them throughput improvements.
7. Keep each optimization independently reversible. Do not bundle several arithmetic/lifetime/scheduling changes into one benchmark result.
8. Only after focused checks pass, run package typecheck/build and the complete applicable suite once before release. Do not repeatedly run the entire model suite for trivial documentation edits.

Dedicated qualification jobs must fail when required model files or WebGPU support are missing. Ordinary lightweight CI can remain model-free, but must clearly report that it is not a WebGPU/parity certification job. Record skipped test counts in artifacts.

## 22. Ordered implementation commits

Execute in this order, with the acceptance gates above:

1. Source/profile manifest, random/clock seams, exact comparators and reference harness.
2. Correct WebGPU benchmark accounting and representative workloads; add the optional Chrome qualification harness and diagnostic hooks from section 25.
3. Shared decoder extraction; no formula changes.
4. Per-session/per-slot evaluation ownership and exceptional cleanup.
5. Synchronous input allocation/forward start; preserve selection/commit schedule.
6. Persistent job/path/recent-move/undo storage.
7. Remove redundant full undo after harvesting history; separate commit from pooling.
8. Ladder scratch reuse and removal of duplicate group traversal; add seeded fast-check equivalence cases from section 25 before changing these helpers.
9. Exact bounded ladder/area feature cache.
10. Direct symmetry-aware encoding.
11. Board-context extraction with compatibility wrappers.
12. Search/math/report extraction preserving serial and prefetched modes.
13. Report scratch reuse and allocation-free ownership traversal.
14. Shared model heads and explicit dead-activation disposal.
15. Copy-only packed output readback experiment; retain only if faster and exact.
16. Score-table cold-start optimization, reported separately.
17. Worker runtime extraction and separately approved/versioned correctness repairs.
18. Parser/package/docs cleanup and final qualification.

Correctness blockers that prevent steps 1–4 from establishing a trustworthy baseline must be surfaced and resolved before that affected path is qualified. They must not be buried in later commits merely to follow this numbering.

## 23. What to keep, defer, and stop doing

**Keep:** TypeScript API, one worker/model owner, typed arrays, existing direct binary-model loading, current rule/feature algorithms, two-slot overlap policy, existing model arithmetic, root symmetry semantics, native fixture generator, and accurate transfer ownership. Keep current custom WGSL kernels with their current selection flags until independently qualified.

**Defer:** full raw-WebGPU inference engine, FP16, quantization, alternative attention reductions, QKV fusion, C++/WASM core migration, adaptive batch-size changes, tree promotion, transposition merging, neural-cache compaction that changes batches, and multi-worker search. These can be useful future projects but cannot be presumed exact to the current baseline. Do not spend this implementation budget building them first.

**Stop treating as optimizations:** lower visits/maxChildren, fewer root symmetries, omitted ownership/PVs, shortened ladder search, looser native tolerances, cached scalar placeholders, or claiming cache hits as performed visits. Do not advertise a universal speedup factor without measurements.

**Stopping criterion:** deliver the fastest measured candidate among the exact-parity variants tested on the declared target profiles, with all accepted changes individually justified. “Optimized as much as possible” is not a provable global maximum; it means no known measured bottleneck is left with an untested, feasible exact-preserving remedy within this scope. If GPU arithmetic dominates after host improvements and exact-preserving copy/lifetime changes do not help, report that limit instead of weakening parity.

## 24. File-by-file disposition checklist

- `src/index.ts`: preserve export compatibility; re-export moved types/facades.
- `src/types/index.ts`, `src/engine/types.ts`: unify through canonical aliases; protocol changes versioned/correlated.
- `src/engine/analyzeMcts.ts`: extract evaluation/search/report modules; preserve both API behaviors; pool scratch and eliminate redundant work.
- `src/engine/fastBoard.ts`: context-owned scratch and internal nonallocating move helpers; preserve rule/ladder order.
- `src/engine/featuresV7Fast.ts`: exact direct symmetry writes and immutable constants; preserve plane/global values.
- `src/engine/evalV8.ts`: keep formulas and rounding behavior; extract offset-based decoding only after exact tests.
- `src/engine/scoreValue.ts`: exact cold-start improvements; preserve interpolation.
- `src/engine/searchParams.ts`: keep constants unchanged.
- `src/engine/territoryKomi.ts`: preserve arithmetic; repair transport history separately.
- `src/engine/limits.ts`: keep ceilings and clamps for compatibility; make tests honor them.
- `src/engine/positionCache.ts`: exclude from strict throughput; redesign exact-result contract separately.
- `src/engine/visitSpeedConfig.ts`: remove misleading inert flags via compatibility policy; snapshot options per session.
- `src/model/modelV8.ts`: shared heads, owned lifetimes, isolated backend adapter; fixed arithmetic profile.
- `src/model/rotaryWebGpu.ts`: keep two-pass rounding boundary; exact tests for dimensions/GQA.
- `src/model/rmsNormWebGpu.ts`: freeze path; test reduction and multiplication order rather than assuming generic equivalence.
- `src/model/binModelParser.ts`: valid-byte semantics unchanged; bound/count validation and explicit ownership.
- `src/model/loadModelV8.ts`: parsed types separated; validate supported dimensions without approximating unsupported models.
- `src/model/modelCache.ts`: asset-cache retry/lifetime cleanup; no search-result responsibilities.
- `src/model/modelDefaults.ts`: keep model choices unchanged.
- `src/utils/ko.ts`: context-aware geometry and characterization of explicit/inferred ko; preserve behavior until correctness changes are explicit.
- `src/utils/helpers.ts`: clock seam, unchanged production clock/opponent behavior.
- `src/utils/browserWorker.ts`: emitted worker/base-path integration; no new UI dependency.
- `src/worker/worker.ts`: runtime extraction, slot lifetime, request identity, exact serializer, separately versioned corrections.
- `src/worker/client.ts`: history/correlation/transfer tests; preserve API facade.
- `src/worker/resourceProfile.ts`: benchmark profile frozen; worker-compatible cache and actual accounting separate from changed tuning policy.
- `src/worker/backendFallback.ts`, `src/worker/gpuError.ts`: preserve application fallback; strict WebGPU qualification fails instead of falling back.
- `test/nativeParity.test.ts`: retain independent tolerant native checks; expand coverage, enforce model availability in qualification.
- `test/pipeline.test.ts`: add reusable-write/error/partial-batch/serialization coverage with nonuniform outputs.
- `test/performance.test.ts`, `test/visitSpeed.test.ts`: share harness; actual visit deltas, explicit backend/options, model-free feature tests outside skipped model suite.
- `test/fixtures/katago-parity-positions.json`, `test/fixtures/katago-model17-native.json`: preserve current references; add independently generated cases, never overwrite to mask regressions.
- `scripts/generate-katago-parity.mjs`: preserve independent executable reference; strengthen required-field validation and metadata; extend only for new cases.
- `benchmarks/run-bench.mjs`, `benchmarks/performance-baseline.json`: environment-keyed paired trials; fresh build provenance and no silent fallback.
- `examples/node-analysis.ts`: correct reported actual visits; keep example behavior, no new demo artifacts.
- `package.json`, `tsconfig.json`, `tsconfig.build.json`: real exports/build artifacts; no numerical dependency upgrades in optimization commits.
- `.github/workflows/ci.yml`: separate lightweight tests from required model/WebGPU qualification.
- `README.md`: accurate API, parity and performance statements.
- `.gitignore`, `LICENSE`: retain; add generated benchmark artifacts to ignore policy only when the implementation creates them.

## 25. Supporting packages and tools (added 2026-10-21)

These tools help identify, implement, and qualify improvements; installing them does not itself increase visits/sec. Keep TensorFlow.js as the production inference backend. Add only development dependencies for the first implementation stages. Pin the selected tool versions and record them in the benchmark manifest; do not upgrade numerical dependencies in the same change. No tools were installed and no browser tests were run while preparing this plan.

### 25.1 Chrome DevTools: identify host allocations and stalls

Use the built-in Performance and Memory panels; no application dependency is needed. This is an optional, explicitly requested profiling activity, not a prerequisite for every patch. Respect the project's restriction on browser testing and recordings unless the user authorizes that activity.

1. Load the same model, fixed board/history, options, random tape, and fixed-visit budget used by the reference harness. Warm up separately.
2. Select the search worker's execution context. Use allocation sampling to locate repeated allocations in selection, `prepareBatchFeatures`, ladder helpers, job/path construction, and reporting. Inspect retained memory separately from allocation rate.
3. Add optional phase markers around batch collection, feature preparation, tensor upload/forward dispatch, output-read completion, backpropagation, and reporting. Include session/slot/batch IDs so overlap can be reconstructed. Keep the instrumentation behind a disabled-by-default diagnostic option.
4. Distinguish CPU execution from time awaiting GPU readback. A long asynchronous interval is not proof of expensive CPU computation, and DevTools timing alone does not establish an individual GPU kernel's duration.
5. Use the findings to select one optimization from this plan. Repeat the normal paired benchmark with profiling and markers disabled; never publish instrumented throughput as the production result.

Reference: [Chrome DevTools memory profiling](https://developer.chrome.com/docs/devtools/memory).

### 25.2 Existing TensorFlow.js diagnostics: measure tensor lifetime

Use the existing `tf.profile` and `tf.memory` APIs before adding another profiler. Verify their exact signatures and backend support against the installed TFJS version during implementation.

1. Add `benchmarks/tensor-diagnostics.ts` as a development-only diagnostic module. It must not be imported by production entry points.
2. Record `tf.memory()` after model initialization and warm-up, after completed batches, and after search/session disposal. Record tensor, data-buffer, and byte counts together with the API's reliability flags/reasons. These counters are not a complete measurement of physical GPU memory or allocator caches.
3. Profile a representative forward evaluation independently from the throughput run. Record available kernel timing and allocation information, tensor shapes, batch size, and selected custom-kernel flags. Treat missing backend timing as unavailable rather than zero.
4. For lifetime changes in section 22, compare repeated same-shape batches after all reads finish. Expected persistent model and slot buffers may remain; unexplained growth across completed searches must be investigated. Do not dispose an input, residual, or output merely to lower a counter while another operation owns it.
5. Keep these diagnostics out of the hot path. Do not add extra `.data()` calls or GPU synchronization to normal search execution.

Reference: [TensorFlow.js API](https://js.tensorflow.org/api/latest/).

### 25.3 Playwright: optional real-Chrome qualification harness

Add `@playwright/test` as a pinned development dependency when implementing the browser qualification stage. This provides repeatable execution and metadata collection; it is not a runtime performance dependency. Browser execution requires explicit authorization under the project instructions. Do not enable screenshots, video, or walkthrough artifacts.

Proposed files: `playwright.config.ts`, `test/browser/webgpu-parity.spec.ts`, `benchmarks/browser-harness.ts`, and a minimal test-only HTML entry. The HTML is a harness, not a product demo. Add separate `test:webgpu:parity` and `bench:webgpu` scripts; do not fold hardware benchmarks into ordinary unit tests.

1. Configure the installed Chrome channel and record its actual version. Record adapter information where exposed, enabled WebGPU features/limits, OS, TFJS version, model SHA-256, source/build provenance, and the complete frozen search profile. Do not assume bundled Chromium or headless execution matches the user's Chrome/GPU path.
2. Serve the built worker and test harness locally. Verify the emitted worker URL and production package entry points, so testing source imports cannot hide packaging defects.
3. In the page, require WebGPU initialization and assert the actual selected backend. Fail qualification if the model or required GPU capability is missing; never substitute WASM/CPU or silently skip.
4. Run the unchanged baseline twice first to establish repeatability. Run baseline and candidate in separate clean sessions with identical model/input bytes and random tapes, ensuring they never compete for the GPU simultaneously.
5. Return lossless typed-array bytes and explicit encodings of special scalar values to the Node comparator. Do not rely on ordinary JSON numbers to preserve negative zero or other special values. Compare the fixed-visit outputs and search traces defined earlier in this plan.
6. Run the seven alternating paired timing trials from Step 02 with diagnostics disabled. Measure inside the worker/search harness, not around Playwright navigation or evaluation calls. Count actual completed visit deltas, excluding initialization/root work as specified earlier.
7. Write machine-readable results with manifests and failures. Keep cold-start and warm throughput results separate. Use a dedicated available GPU runner for qualification; ordinary hosted CI without an appropriate adapter is not a substitute.

Reference: [Playwright browser channels](https://playwright.dev/docs/browsers).

### 25.4 fast-check: protect board, feature, and ownership refactors

Add `fast-check` as a pinned development dependency and use it through the existing Vitest runner. Add `test/boardParity.property.test.ts` and `test/featureParity.property.test.ts`; extend pipeline tests for ownership/lifetime cases. Start with a bounded fixed seed and 100 cases per property for local checks, with a larger explicitly invoked qualification run.

1. Generate valid game histories by applying legal commands to an independently retained reference implementation. Include pass, captures, ko, and supported board sizes/rules. Avoid generating arbitrary stone arrays as a replacement for valid histories.
2. Replay exactly the same commands against the candidate. After each move, compare stones, player, ko, relevant history, and full recomputed liberty maps. For apply/undo operations, compare against the captured pre-move state.
3. For feature preparation, compare every byte of spatial/global inputs and every ladder/area result, including all supported symmetries. Preserve ladder budgets and traversal order. Generate same stones with different history/ko to exercise cache-key separation.
4. For scratch pools and caches, generate interleaved sessions and slot reuse. Retain snapshots of earlier outputs, exercise eviction and deliberate hash collisions, then assert retained outputs did not mutate and exact collision checks reject nonmatching positions.
5. For protocol ownership, use small nonuniform fake neural outputs and a state-machine command sequence covering submit, complete, cancel, report, and dispose. Compare the reference reservation/commit rules; verify that transferring a report cannot detach arrays still owned by the live search.
6. Save failing seed, shrink path, and commands. Turn every minimized discovered failure into a named regression case. Shrinking must preserve command preconditions; never silently discard every difficult generated case.
7. Keep model-free properties separate from model-backed WebGPU qualification. Property tests supplement the exact full-search and native compatibility tests; they do not prove floating-point parity by themselves. If a reference bug appears, follow the separate correctness-baseline process instead of changing both sides to make the property pass.

References: [fast-check model-based testing](https://fast-check.dev/docs/advanced/model-based-testing/) and [configuration and reproducibility](https://fast-check.dev/docs/configuration/).

### 25.5 ONNX Runtime Web: deferred backend experiment

`onnxruntime-web` with its WebGPU entry point is a possible alternative inference runtime. GPU-resident tensors and graph capture may reduce overhead for supported static graphs. This is a research branch after the existing backend's host/lifetime improvements, not an accepted parity-preserving substitution. Do not add it to the production bundle or replace TFJS merely because its microbenchmark is faster.

1. First demonstrate from measurements that inference dispatch/kernel overhead remains a material bottleneck. If board/feature work dominates, prioritize the earlier host changes.
2. Inventory the parsed architecture and every required operator, including custom rotary/RMS behavior, GQA, pooling, and all output heads. Check the current WebGPU operator support before writing a converter. Direct binary loading currently exists; ONNX requires a separately validated conversion pipeline or equivalent model export.
3. Create an isolated experimental dependency/package and converter. Record source model digest, converter version/options, output graph digest, tensor layouts, and head semantics. Preserve the baseline's FP32 arithmetic intent; disable optional precision or graph rewrites where supported, but do not assume this guarantees identical math.
4. Start with one model and the exact shapes used by one frozen baseline profile. Compare input features and all raw policy/value/score/ownership outputs bitwise before invoking MCTS. Only after this passes compare decoded outputs and complete fixed-visit search traces. A tolerance pass or unchanged top move does not satisfy this plan.
5. Investigate GPU-resident tensors only with explicit producer/consumer ownership and compatible devices. Do not assume TFJS buffers can be handed directly to ORT. Test disposal, device loss, and readback behavior independently.
6. Attempt graph capture only after confirming its documented prerequisites: supported static shapes and all relevant compute kernels executing on WebGPU. Account separately for each actual batch shape and partial tail; never pad or change batch membership merely to enable capture under the strict baseline.
7. Benchmark conversion/model loading, warm-up/capture cost, peak retained memory, warm inference, and end-to-end visits/sec separately. Compare with the production flags and outputs enabled, not a reduced head-only workload.
8. Reject the experiment from the strict-parity release if any required exact comparison fails. A faster numerically different implementation can be reported as a separate future backend proposal, requiring an explicitly different parity contract. No expected speedup is promised.

References: [ORT WebGPU usage, graph capture, and GPU tensors](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html), [Web performance diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html), and [WebGPU execution provider](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html).

### 25.6 Integration order and completion criteria

Extend implementation commit 1 with seeded property-test infrastructure and commit 2 with the development-only diagnostic hooks and optional browser harness. Add targeted properties immediately before each corresponding board/cache/pool refactor. Keep DevTools/TFJS diagnostic runs separate from throughput measurements. Leave ONNX Runtime outside the 18-commit production sequence until measurements justify it and its exact-parity gates pass.

Completion means the selected tools produce reproducible failure cases and trustworthy fixed-visit measurements without adding production hot-path work. Documentation updates alone require no dependency installation, browser execution, or full test-suite run.

## 26. Clean-code architecture and refactoring specification

### 26.1 Allowed dependencies and public boundaries

Use the directory structure in section 4 as the target, with these enforced directions:

- `engine/board` owns board geometry and rule algorithms. It cannot import search, evaluation, model, worker, TensorFlow.js, or browser storage.
- `engine/features` imports board algorithms and feature types. It cannot import tree nodes, worker messages, or TFJS tensors.
- `engine/search` imports board/features, search types, and the evaluation contract. It cannot import TFJS, GPU objects, IndexedDB, worker transport, or client code.
- `engine/evaluation` owns feature batches and orchestration through an injected inference adapter. Keep its public contracts independent of TFJS; concrete tensor creation/disposal lives in `model/tfjs`.
- `model/parsedTypes.ts` contains only data declarations. Parser code depends on those declarations rather than importing the executable model implementation.
- `model/tfjs` owns tensor operations, GPU-facing resources, and backend-specific capability checks. It cannot import search policy or worker message handlers.
- `worker/runtime.ts` composes the concrete model adapter and search session. `worker/protocol.ts` contains transport data types only; `worker/serialization.ts` owns snapshots and transfers.
- `worker/client.ts` depends on protocol and browser-worker creation, never engine/model implementations. `worker/worker.ts` routes requests to runtime operations and sends responses.
- Public entry modules re-export compatibility facades and types. Internal code imports concrete internal modules directly instead of importing the package barrel and creating cycles.

Use `import type` where the dependency is type-only. Do not move domain logic into a general `utils` directory to evade these rules. Add a focused import-boundary check using the existing TypeScript compiler API when these directories are introduced: parse import/export declarations, resolve relative imports, and reject forbidden layer edges and runtime cycles. Run it on the source tree without loading model weights. Do not add a large architecture framework for this check.

### 26.2 Resource ownership must be visible in types and code

Replace implicit shared state with concrete owners, allocated once at the appropriate lifetime:

1. `BoardGeometry`: immutable dimensions, ordered neighbor tables, and symmetry maps. Share only immutable geometry for equal sizes.
2. `BoardContext`: geometry plus mutable board-algorithm scratch. A search owns its context; functions that mutate scratch receive that context explicitly. Document any helper whose returned view is invalidated by the next call.
3. `SearchSession`: root/tree state, random generator, frozen options, board context, feature cache, report scratch, and scheduler. It does not own the worker transport or dispose a shared loaded model.
4. `EvaluationSlot`: feature arrays, retained job records, actual job count, readback state, and an adapter-owned input-resource lease. A slot cannot be refilled until its last consumer releases it.
5. `ModelRuntime`: parsed model identity, inference adapter, model weights, and model-generation lifetime. It outlives individual searches and disposes model resources only after active operations drain.
6. `AnalysisSnapshot`: report-owned arrays safe to transfer. Borrowed tree/model arrays are never transferred. Naming an array `readonly` does not make its buffer transferable safely.

Document the owner and release point on each interface field that holds a tensor, typed-array view, buffer lease, or outstanding promise. Use explicit operations such as `acquireInputSlot`, `releaseInputSlot`, and `dispose`; avoid a generic resource manager that hides lifetimes. Keep lease objects stable across batches where safe instead of adding allocations to the hot path.

Represent evaluation-slot state as `idle`, `prepared`, `submitted`, `readable`, and `committing`, with cancellation/error draining paths. Define transitions in one scheduler module. Retain pending-operation references until every started read completes or rejects; cancellation is not permission to dispose resources still being read. Development assertions can validate transitions, but must be disabled in performance qualification.

### 26.3 Exact extraction procedure for every large module

Apply this procedure to `analyzeMcts.ts`, `fastBoard.ts`, `modelV8.ts`, and `worker.ts` in small slices:

1. Identify the functions to move, their callers, mutable globals, borrowed buffers, output aliases, and arithmetic-order assumptions. Record these in the destination module's short ownership comment.
2. Extract shared data types first. Preserve field types and defaults; do not replace JS-number accumulators with Float32 storage or reorder fields into a new packed representation during extraction.
3. Move the functions verbatim, retaining loop order, branch order, intermediate assignments, and rounding boundaries. Adjust imports only. Preserve old exports through a forwarding facade.
4. Pass formerly implicit dependencies explicitly. For non-hot setup paths use a named options object; for per-node/per-point helpers pass existing stable context references without constructing an options object per call.
5. Run the cheapest affected characterization tests and typecheck. For numerical/scheduling code, compare exact outputs and traces against the retained reference. Resolve accidental import cycles before proceeding.
6. Remove the old duplicate body once the new code serves the original callers. Keep intentionally different behavior in explicitly named compatibility policies; never merge differing defaults by accident.
7. Commit the extraction separately. Only then implement pooling, caching, reduced traversal, or lifetime optimization and measure it with the existing gates.

Do not pursue a maximum line count or split every short helper into a file. A module is cohesive when its functions share a responsibility and resource lifetime. Keep tightly coupled hot loops together when splitting would require repeated copying, dynamic dispatch, or tangled dependencies.

### 26.4 Concrete decomposition of the existing large files

**`analyzeMcts.ts`:** move Node/Edge declarations into `search/tree.ts`; move unchanged PUCT/FPU/noise formulas into `selection.ts`; move legality/prior/top-K logic into `expansion.ts`; move score utility into `utility.ts`; move feature batching and decode coordination into `evaluation/batch.ts`; move output/PV/ownership construction into `report.ts`. Keep root creation, run lifecycle, and tree ownership in `session.ts`. Place only collection/dispatch/commit sequencing in `scheduler.ts`. The old file becomes the public facade for the class and legacy function, each selecting its established compatibility policy.

**`fastBoard.ts`:** extract geometry/scratch first, followed by liberties, moves, area, and ladders. Moves and liberty routines may share low-level board types but must not form an import cycle; move their truly shared primitives into a small `board/primitives.ts` only if needed by both. Preserve exported wrappers around the default context. Internal search paths must use the session context. Keep neighbor order, ladder limits, rollback semantics, and algorithms unchanged during these moves.

**`modelV8.ts`:** move parsed descriptors first; separate input-slot management and output readback from graph construction; extract shared policy/value/score/ownership heads without changing operation order. Keep residual-block execution and architecture-specific graph logic together until there is a concrete duplicated unit to extract. Preserve custom rotary/RMS kernel selection and all current arithmetic boundaries. The public model class delegates to concrete functions; do not create a subclass hierarchy per layer type.

**`worker.ts`:** extract protocol types, then serialization, then model/backend lifetime into runtime. Keep routing as a small explicit switch over typed messages. Extract request scheduling into `worker/requestQueue.ts` only when moving the existing queue implementation: it owns request ordering/cancellation identities, while search's scheduler owns neural-batch ordering. Neither scheduler reaches into the other's internal collections. Move backend capability/resource probing behind runtime methods, retaining its current policy until the separate correctness/tuning stage.

### 26.5 Eliminate duplication without concealing behavior

- Keep exactly one neural-output decoder for the class, legacy, and asynchronous evaluation paths. It accepts raw arrays plus offsets/strides and writes caller-owned destinations. Preserve mixing formulas and float32 writes.
- Share model head construction only when the emitted operations and operation order are identical. A helper takes explicit requested heads; it must not compute extra ownership outputs merely to simplify callers.
- Use one canonical worker protocol and public result-type source with compatibility re-exports. Do not maintain independently edited copies of the same shape.
- Resolve defaults at each public boundary once, producing an immutable `ResolvedSearchOptions`. Legacy and class defaults remain explicitly separate where they differ. Internal modules do not repeat defaulting expressions or consult mutable flags mid-run.
- Group compatibility choices in a named profile with scheduling/report/root-initialization policies. Do not scatter checks such as `isLegacy` through board math or TFJS kernels. Only add policies for observed differences, not hypothetical future engines.
- Consolidate benchmark model loading, frozen profile construction, visit accounting, and manifest writing into shared harness helpers. Unit tests retain their lightweight fakes; they need not use a real-model loader.
- Distinguish model asset cache, board-feature cache, and completed-analysis cache in names and interfaces. They have different keys and correctness contracts and must not share a generic `getOrCompute` API that hides those distinctions.

### 26.6 Readability rules for the implementation

Use names that reveal lifetime and units: `activeJobCount`, `slotCapacity`, `completedVisitDelta`, `elapsedMs`, `borrowedOwnership`, and `snapshotOwnership`. Prefer an explicit loop to a compact chain that allocates or obscures traversal order. Keep constants for fixed semantic values near the owning algorithm; keep tunable configuration in resolved options. Preserve the original numeric literals and formulas when naming constants.

Replace untyped messages and broad casts at transport boundaries with discriminated unions and narrow validation. Validate external inputs once at the boundary, not repeatedly per node. Do not relax current validation or silently reinterpret malformed inputs during a cleanup commit. Treat backend-specific TFJS internals as one isolated adapter capability check rather than repeated casts throughout search code.

Comments should explain invariants: why rotary has two passes, why a buffer cannot be reused yet, why edge order is stable, or why two API modes remain distinct. Remove obsolete comments only after checking the current code. Avoid comments that merely narrate an assignment. Document scratch-view validity and mutation explicitly.

Remove dead private code after checking references across source/tests/scripts. Deprecate public exports or deep-import paths instead of silently removing them; preserve shims where compatibility requires it. Avoid reformatting unrelated files, introducing a new formatter, or enabling a large lint ruleset in the same change as numerical extraction.

### 26.7 Cleanup delivery sequence and required checks

Use these checkpoints within section 22's existing sequence:

1. Before extraction: retain reference manifests and exact comparators; inventory public exports and their callers.
2. During stages 3–5: create evaluation contracts, the shared decoder, explicit slot ownership, and adapter boundaries. Verify delayed reads, failures, cancellation, and original commit order.
3. During stages 6–10: use named pool counts/capacities, context-owned scratch, and feature-cache contracts. Verify retained views, cache collisions, symmetry bytes, and same-seed fixed-visit results.
4. During stages 11–12: finish board/search decomposition and compatibility facades. Enforce dependency directions and compare each public API to its own baseline.
5. During stages 13–16: isolate reporting and model lifetime details. Verify output snapshot ownership, tensor cleanup, and unchanged reduction order.
6. During stages 17–18: finish worker routing/runtime separation, parser types, package entry points, and documentation. Keep correctness repairs explicitly separate from extraction commits.

Add `docs/architecture.md` describing the dependency direction, model/session/slot lifetimes, and public compatibility modes. Add `docs/performance.md` describing the frozen benchmark profiles and parity gate commands. These documents should point to owning modules rather than duplicate implementation internals. Update them when a boundary changes; do not add walkthrough recordings or a demo application.

Final acceptance: public entry points still resolve; no forbidden runtime imports or runtime cycles remain; search cannot access TFJS tensors or worker transport directly; mutable search scratch is session/slot-owned; each transferred array has snapshot ownership; duplicate decoders/types/default resolution have a single authority; intentionally different API behavior is documented and tested. All affected exact-parity gates must pass, and organization-only refactors must satisfy the existing no-regression threshold. Record neutral cleanup as maintainability work, not as a visits/sec gain.

