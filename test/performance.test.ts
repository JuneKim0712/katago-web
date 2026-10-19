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

const modelPath = [
  process.env.KATAGO_TEST_MODEL,
  process.env.KATAGO_MODEL,
  'C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz',
  resolve('models/kata1-tf2-b10c384-s2941M-d5872M.bin.gz'),
].find((p) => p && existsSync(p)) ?? '';

describe.skipIf(!existsSync(modelPath))('KataGo Performance Benchmark (kata1-b10c384)', () => {
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

    const raw = readFileSync(modelPath);
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

  it('measures forward inference latency on b10c384', async () => {
    const spatialSingle = new Float32Array(19 * 19 * 22);
    const globalSingle = new Float32Array(19);
    fillInputsV7Fast({
      stones: new Uint8Array(361),
      koPoint: -1,
      currentPlayer: 'black',
      recentMoves: [],
      komi: 7.5,
      outSpatial: spatialSingle,
      outGlobal: globalSingle,
    });

    const spatial = tf.tensor4d(spatialSingle, [1, 19, 19, 22]);
    const global = tf.tensor2d(globalSingle, [1, 19]);

    // Warmup
    const warmupOut = model.forwardPolicyValue(spatial, global);
    await warmupOut.policy.data();
    tf.dispose(warmupOut);

    // Measure 3 iterations
    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      const t0 = performance.now();
      const out = model.forwardPolicyValue(spatial, global);
      await Promise.all([out.policy.data(), out.value.data()]);
      times.push(performance.now() - t0);
      tf.dispose(out);
    }
    spatial.dispose();
    global.dispose();

    times.sort((a, b) => a - b);
    const medianMs = times[1];
    console.log(`\n[Performance] b10c384 batch=1 forward latency: ${medianMs.toFixed(1)}ms (${(1000 / medianMs).toFixed(1)} evals/sec)`);
    expect(medianMs).toBeGreaterThan(0);
    expect(medianMs).toBeLessThan(15000); // Sanity guard for CPU
  }, 45000);

  it('measures MCTS search visit speed on b10c384', async () => {
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

    const initialTensors = tf.memory().numTensors;
    const targetVisits = 6;

    const t0 = performance.now();
    await search.run({ visits: targetVisits, maxTimeMs: 60000, batchSize: 2 });
    const elapsed = performance.now() - t0;

    const analysis = search.getAnalysisLight({ topK: 3, analysisPvLen: 2 });
    const visitsPerSec = (analysis.rootVisits / (elapsed / 1000));

    console.log(`[Performance] MCTS search: ${analysis.rootVisits} visits in ${elapsed.toFixed(1)}ms (${visitsPerSec.toFixed(2)} visits/sec)`);
    console.log(`[Performance] Top move: (${analysis.moves[0]?.x}, ${analysis.moves[0]?.y}) prior: ${(analysis.moves[0]?.prior * 100).toFixed(1)}%`);

    expect(analysis.rootVisits).toBeGreaterThanOrEqual(targetVisits);
    expect(visitsPerSec).toBeGreaterThan(0.05);

    expect(Number.isFinite(analysis.rootWinRate)).toBe(true);
    expect(analysis.rootWinRate).toBeGreaterThan(0.3);
    expect(analysis.rootWinRate).toBeLessThan(0.7);
  }, 120000);
});
