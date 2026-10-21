# KataGo Web: Optimization & Parity Progress Record

## Initial Source Identity & Hashes (SHA-256)
- `src/engine/analyzeMcts.ts`: `97E23C090B530C73FF47F114C228E85C142F10B5DAB933CAC408226AD716F517`
- `src/engine/fastBoard.ts`:   `99B98AA5A568250DADDF9B7179101A6CB73B471B3168F1C55DABFF8A54134155`
- `src/model/modelV8.ts`:     `2FAF22AD27AA0C1E62FA706498CB22655DA2DB14F9B2CCFEB62737BFA17B2A2E`
- `src/worker/worker.ts`:     `DBE3CB76EF8E2820E907A73783BC934486110817BAF9D6784EDA9A8785CFAD0A`
- `src/worker/client.ts`:     `E53582310DE95EDA2CFC0D2633642BDB58F4571CEA048A16723F70893F7EB220`
- Baseline snapshot preserved in: `C:\Users\junwo\.gemini\antigravity\scratch\katago-web-baseline`

## Environment & Toolchain
- Platform: Windows x64, Node v24.x, TypeScript ~5.9.3, Vitest ^4.1.8, fast-check 4.10.2
- Pinned TF.js: `@tensorflow/tfjs: ^4.22.0`, `@tensorflow/tfjs-backend-wasm: ^4.22.0`, `@tensorflow/tfjs-backend-webgpu: ^4.22.0`
- Model available: `C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz`

## Stage Status (Section 22 Checklist)
- [x] Stage 1: Source/profile manifest, random/clock seams, exact comparators, reference harness
- [x] Stage 2: Correct WebGPU benchmark accounting, representative workloads, browser harness / diagnostic hooks
- [x] Stage 3: Shared decoder extraction (no formula changes)
- [x] Stage 4: Per-session/per-slot evaluation ownership & exceptional cleanup
- [x] Stage 5: Synchronous input allocation/forward start, preserve selection/commit schedule
- [x] Stage 6: Persistent job/path/recent-move/undo storage
- [x] Stage 7: Remove redundant full undo after harvesting history
- [x] Stage 8: Ladder scratch reuse & removal of duplicate group traversal
- [x] Stage 9: Exact bounded ladder/area feature cache
- [x] Stage 10: Direct symmetry-aware encoding (eliminated `spatialScratch` and extra 22-plane copy loops)
- [x] Stage 11: Board-context extraction with compatibility wrappers (`src/engine/board/`)
- [x] Stage 12: Search/math/report extraction preserving serial and prefetched modes (`src/engine/search/`)
- [x] Stage 13: Report scratch reuse & allocation-free ownership traversal (2-pass edge scan in `averageTreeOwnership`)
- [ ] Stage 14: Shared model heads & explicit dead-activation disposal
- [ ] Stage 15: Copy-only packed output readback experiment (gated)
- [ ] Stage 16: Score-table cold-start optimization (reported separately)
- [ ] Stage 17: Worker runtime extraction & versioned correctness repairs
- [ ] Stage 18: Parser/package/docs cleanup and final qualification

## Current Test Status
- `npm run typecheck`: PASS (0 errors)
- `npx vitest run test/boardParity.property.test.ts test/boardContext.test.ts test/featureCache.test.ts test/pipeline.test.ts test/parity/`: 21 passed (8 test files)
  - `test/parity/floatBits.test.ts`: 5 passed
  - `test/boardContext.test.ts`: 3 passed (neighbor ordering, stamp rollover near Int32 overflow, multi-context isolation)
  - `test/parity/featuresExact.test.ts`: 3 passed (all 8 symmetries, rules, players matching legacy encode-then-permute bitwise)
  - `test/parity/inferenceExact.test.ts`: 1 passed
  - `test/featureCache.test.ts`: 2 passed
  - `test/boardParity.property.test.ts`: 1 passed (100 random board states vs reference)
  - `test/parity/searchTrace.test.ts`: 2 passed (both `MctsSearch` class and standalone `analyzeMcts` function pass bitwise determinism)
  - `test/pipeline.test.ts`: 4 passed (concurrent run rejection, overlapping slot distinctness, cancellation drain)

## Next Action
- Implement Stage 14: Shared model heads & explicit dead-activation disposal in `src/model/modelV8.ts` and `src/model/parsedTypes.ts`.
