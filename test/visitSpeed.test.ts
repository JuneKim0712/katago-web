import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
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
  type BoardState,
} from '../src/index.js';

const baselinePath = resolve(process.cwd(), 'benchmarks/performance-baseline.json');
const baselineConfig = existsSync(baselinePath)
  ? JSON.parse(readFileSync(baselinePath, 'utf8'))
  : null;

const b10ModelCandidate = [
  process.env.KATAGO_TEST_MODEL,
  process.env.KATAGO_MODEL,
  'C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz',
  resolve('models/kata1-tf2-b10c384-s2941M-d5872M.bin.gz'),
].find((p) => p && existsSync(p));

const smallModelCandidate = [
  resolve('../baduk/public/models/katago-small.bin.gz'),
  resolve('../baduk/dist/models/katago-small.bin.gz'),
  resolve('models/katago-small.bin.gz'),
].find((p) => p && existsSync(p));

const selectedModelPath = b10ModelCandidate || smallModelCandidate || '';
const isB10Model = Boolean(b10ModelCandidate && selectedModelPath === b10ModelCandidate);
const modelKey = isB10Model ? 'b10c384' : 'small';
const baseline = baselineConfig?.[modelKey] ?? {
  minVisitsPerSec: 0.10,
  maxForwardMs: 12000,
  minFeatureOpsPerSec: 250000,
};

describe.skipIf(!selectedModelPath)('MCTS Visit-Speed & Performance Enforcement', () => {
  let model: KataGoModelV8Tf;

  beforeAll(async () => {
    try {
      const wasmDir = resolve(process.cwd(), 'node_modules/@tensorflow/tfjs-backend-wasm/dist');
      setWasmPaths(`${wasmDir}/`);
      await tf.setBackend('wasm');
    } catch {
      await tf.setBackend('cpu');
    }
    await tf.ready();

    const raw = readFileSync(selectedModelPath);
    const parsed = parseKataGoModelV8(gunzipSync(raw));
    model = new KataGoModelV8Tf(parsed, {
      fuseRotary: true,
      fuseRmsNorm: true,
      fuseTransformerProjections: true,
    });
  });

  afterAll(() => {
    model?.dispose();
  });

  it('enforces feature extraction throughput meets baseline', () => {
    const stones = new Uint8Array(361);
    for (let i = 0; i < 361; i += 3) stones[i] = 1;
    for (let i = 1; i < 361; i += 3) stones[i] = 2;
    const libertyMap = new Uint8Array(361);
    const outSpatial = new Float32Array(361 * 22);
    const outGlobal = new Float32Array(19);

    const WARMUP_ITERS = 5000;
    for (let i = 0; i < WARMUP_ITERS; i++) {
      fillInputsV7Fast({
        stones,
        koPoint: 180,
        currentPlayer: 'black',
        recentMoves: [{ move: 100, player: 'white' }, { move: 101, player: 'black' }],
        komi: 7.5,
        rules: 'japanese',
        libertyMap,
        outSpatial,
        outGlobal,
      });
    }

    const TEST_ITERS = 25000;
    const t0 = performance.now();
    for (let i = 0; i < TEST_ITERS; i++) {
      fillInputsV7Fast({
        stones,
        koPoint: 180,
        currentPlayer: 'black',
        recentMoves: [{ move: 100, player: 'white' }, { move: 101, player: 'black' }],
        komi: 7.5,
        rules: 'japanese',
        libertyMap,
        outSpatial,
        outGlobal,
      });
    }
    const elapsedMs = performance.now() - t0;
    const opsPerSec = Math.round(TEST_ITERS / (elapsedMs / 1000));
    const usPerOp = ((elapsedMs / TEST_ITERS) * 1000).toFixed(2);

    console.info(`\n[Perf:Features] ${opsPerSec.toLocaleString()} ops/sec (${usPerOp} µs/op) | Baseline >= ${baseline.minFeatureOpsPerSec.toLocaleString()}`);

    expect(opsPerSec).toBeGreaterThanOrEqual(baseline.minFeatureOpsPerSec);
  });

  it('enforces forward inference latency meets baseline', async () => {
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

    // Warmup pass
    const warmup = model.forwardPolicyValue(spatial, global);
    await Promise.all([warmup.policy.data(), warmup.value.data()]);
    tf.dispose(warmup);

    const times: number[] = [];
    const ITERS = 3;
    for (let i = 0; i < ITERS; i++) {
      const t0 = performance.now();
      const out = model.forwardPolicyValue(spatial, global);
      await Promise.all([out.policy.data(), out.value.data()]);
      times.push(performance.now() - t0);
      tf.dispose(out);
    }
    spatial.dispose();
    global.dispose();

    times.sort((a, b) => a - b);
    const medianMs = times[1]!;

    console.info(`[Perf:Forward] Batch=1 median latency: ${medianMs.toFixed(1)}ms (${(1000 / medianMs).toFixed(2)} evals/sec) | Baseline <= ${baseline.maxForwardMs}ms`);

    expect(medianMs).toBeLessThanOrEqual(baseline.maxForwardMs);
  }, 60000);

  it('enforces MCTS visit throughput (visits/sec) meets performance gate', async () => {
    const emptyBoard: BoardState = Array.from({ length: 19 }, () => Array(19).fill(null));

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

    const targetVisits = isB10Model ? 8 : 40;
    const batchSize = isB10Model ? 2 : 8;

    const t0 = performance.now();
    await search.run({ visits: targetVisits, maxTimeMs: 60000, batchSize });
    const elapsed = performance.now() - t0;

    const analysis = search.getAnalysisLight({ topK: 3, analysisPvLen: 2 });
    const visitsPerSec = analysis.rootVisits / (elapsed / 1000);

    const reportTable = [
      '=========================================================================',
      '                      KATAGO MCTS PERFORMANCE AUDIT                      ',
      '=========================================================================',
      `Model:             ${selectedModelPath.split(/[\/\\]/).pop()} (${isB10Model ? '10-Block 384c' : 'Small'})`,
      `Backend:           ${tf.getBackend()} (WASM)`,
      `Batch Size:        ${batchSize}`,
      `Total Visits:      ${analysis.rootVisits}`,
      `Search Time:       ${(elapsed / 1000).toFixed(2)}s (${elapsed.toFixed(1)}ms)`,
      `Measured Throughput: ${visitsPerSec.toFixed(2)} visits/sec`,
      `Required Baseline: >= ${baseline.minVisitsPerSec.toFixed(2)} visits/sec`,
      `Status:            ${visitsPerSec >= baseline.minVisitsPerSec ? 'PASS (Meets Performance Gate ✅)' : 'FAIL (Performance Regressed ❌)'}`,
      '=========================================================================',
    ].join('\n');

    console.info('\n' + reportTable + '\n');

    expect(analysis.rootVisits).toBeGreaterThanOrEqual(targetVisits);
    expect(visitsPerSec).toBeGreaterThanOrEqual(baseline.minVisitsPerSec);
  }, 120000);
});
