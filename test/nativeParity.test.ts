import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { resolve } from 'node:path';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-wasm';
import fixture from './fixtures/katago-model17-native.json';
import {
  parseKataGoModelV8,
  KataGoModelV8Tf,
  MctsSearch,
  playMove,
  fillInputsV7Fast,
  type BoardState,
  type GameRules,
  type Move,
  type Player,
} from '../src/index.js';

const modelPath =
  process.env.KATAGO_TEST_MODEL ??
  ['C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz', 'models/kata1-tf2-b10c384-s2941M-d5872M.bin.gz'].find((p) =>
    existsSync(p)
  ) ??
  '';

describe.skipIf(!existsSync(modelPath))('native KataGo v1.18.1 raw-output parity (b10c384)', () => {
  let model: KataGoModelV8Tf;
  let parsedModel: ReturnType<typeof parseKataGoModelV8>;

  beforeAll(async () => {
    const compressed = readFileSync(modelPath);
    expect(createHash('sha256').update(compressed).digest('hex')).toBe(fixture.modelSha256);

    if (process.env.KATAGO_WEBGPU_MODULE) {
      const { create, globals } = await import(/* @vite-ignore */ pathToFileURL(process.env.KATAGO_WEBGPU_MODULE).href);
      Object.assign(globalThis, globals);
      Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: create(['backend=d3d12']) } });
      await import('@tensorflow/tfjs-backend-webgpu');
      if (!(await tf.setBackend('webgpu'))) throw new Error('WebGPU unavailable');
    } else {
      try {
        const wasmDir = resolve(process.cwd(), 'node_modules/@tensorflow/tfjs-backend-wasm/dist');
        setWasmPaths(`${wasmDir}/`);
        await tf.setBackend('wasm');
      } catch {
        await tf.setBackend('cpu');
      }
    }
    await tf.ready();

    parsedModel = parseKataGoModelV8(gunzipSync(compressed));
    model = new KataGoModelV8Tf(parsedModel, {
      fuseTransformerProjections: process.env.KATAGO_FUSED_PROJECTIONS === '1',
      fuseRotary: true,
      fuseRmsNorm: true,
    });
  });

  afterAll(() => {
    model?.dispose();
    if (process.env.KATAGO_WEBGPU_MODULE) {
      tf.removeBackend('webgpu');
      Reflect.deleteProperty(globalThis, 'navigator');
    }
  });

  // 1. Model Structure & Architecture Verification
  it('parses b10c384 modelVersion 17 architecture accurately', () => {
    expect(parsedModel.modelVersion).toBe(17);
    expect(parsedModel.modelName).toBe('b10c384h6nbttflrs');
    expect(parsedModel.policyOutChannels).toBe(2);
    expect(parsedModel.trunk.numBlocks).toBe(10);
    expect(parsedModel.trunk.trunkNumChannels).toBe(384);
    expect(parsedModel.trunk.tipRMSNorm?.channels).toBe(384);
    expect(parsedModel.trunk.tipActivation).toBe('silu');
  });

  // 2. Parity across All 9 Reference Game Positions
  for (const position of fixture.cases) {
    it(`parity on position: ${position.id} (${position.rules}, komi ${position.komi})`, async () => {
      const state = { stones: new Uint8Array(361), koPoint: -1 };
      const snapshot = (): BoardState =>
        Array.from({ length: 19 }, (_, y) =>
          Array.from({ length: 19 }, (_, x) => {
            const stone = state.stones[y * 19 + x];
            return stone === 1 ? 'black' : stone === 2 ? 'white' : null;
          })
        );

      const history: Move[] = [];
      const boards: BoardState[] = [snapshot()];

      for (const [player, coordinate] of position.moves) {
        const pass = coordinate === 'pass';
        const x = pass ? -1 : 'ABCDEFGHJKLMNOPQRST'.indexOf(coordinate[0]);
        const y = pass ? -1 : 19 - Number(coordinate.slice(1));
        playMove(state, pass ? 361 : y * 19 + x, player === 'black' ? 1 : 2, []);
        history.push({ x, y, player: player as Player });
        boards.push(snapshot());
      }

      const search = await MctsSearch.create({
        model,
        board: boards.at(-1)!,
        previousBoard: boards.at(-2),
        previousPreviousBoard: boards.at(-3),
        previousPreviousPreviousBoard: boards.at(-4),
        currentPlayer: position.currentPlayer as Player,
        moveHistory: history,
        koPoint: state.koPoint,
        komi: position.komi,
        rules: position.rules as GameRules,
        nnRandomize: false,
        conservativePass: false,
        maxChildren: 361,
        ownershipMode: 'root',
        wideRootNoise: 0,
        rootSymmetrySamples: 1,
      });

      const actual = search.getAnalysis({ topK: 10, analysisPvLen: 1, cloneBuffers: false });

      // Check winrate parity (< 0.02% difference)
      expect(Math.abs(actual.rootWinRate - position.native.whiteLoss), 'black win probability parity').toBeLessThan(
        0.0002
      );

      // Check score lead parity (< 0.01 points difference)
      expect(Math.abs(actual.rootScoreLead + position.native.whiteLead), 'score lead parity').toBeLessThan(0.01);

      // Check score mean parity (< 0.01 points difference)
      expect(
        Math.abs(actual.rootScoreSelfplay + position.native.whiteScoreSelfplay),
        'score selfplay mean parity'
      ).toBeLessThan(0.01);

      // Check score stdev parity
      const expectedVariance = Math.max(
        0,
        position.native.whiteScoreSelfplaySq - position.native.whiteScoreSelfplay ** 2
      );
      const expectedStdev = Math.sqrt(expectedVariance);
      expect(Math.abs(actual.rootScoreStdev - expectedStdev), 'score stdev parity').toBeLessThan(0.05);

      // Check policy array (361 moves + pass move)
      const expectedPolicy = [...position.native.policy, position.native.policyPass];
      let policyMaxError = 0;
      let policySumError = 0;
      let validPolicyCount = 0;
      let ownershipMaxError = 0;
      let ownershipSumError = 0;

      for (let i = 0; i < 362; i++) {
        const exp = expectedPolicy[i];
        if (exp === null) {
          expect(actual.policy[i], `illegal move ${i} must have negative policy`).toBeLessThan(0);
        } else {
          const err = Math.abs(actual.policy[i] - exp);
          policyMaxError = Math.max(policyMaxError, err);
          policySumError += err;
          validPolicyCount++;
        }

        if (i < 361) {
          const oErr = Math.abs(actual.ownership[i] + position.native.whiteOwnership[i]);
          ownershipMaxError = Math.max(ownershipMaxError, oErr);
          ownershipSumError += oErr;
        }
      }

      // Max absolute error thresholds
      expect(policyMaxError, 'max policy absolute error').toBeLessThan(0.0003);
      expect(policySumError / validPolicyCount, 'mean policy absolute error').toBeLessThan(0.00005);
      expect(ownershipMaxError, 'max ownership absolute error').toBeLessThan(0.002);
      expect(ownershipSumError / 361, 'mean ownership absolute error').toBeLessThan(0.0005);

      // Check pass move policy specifically
      expect(
        Math.abs(actual.policy[361] - position.native.policyPass),
        'pass move policy parity'
      ).toBeLessThan(0.0002);

      // Check top 5 policy move rankings 1:1 match native KataGo
      const nativeIndexedPolicy = position.native.policy
        .map((prob: number | null, idx: number) => ({ idx, prob: prob ?? -1 }))
        .sort((a: { prob: number }, b: { prob: number }) => b.prob - a.prob);

      const actualIndexedPolicy = Array.from(actual.policy.slice(0, 361))
        .map((prob: number, idx: number) => ({ idx, prob }))
        .sort((a, b) => b.prob - a.prob);

      for (let rank = 0; rank < 5; rank++) {
        expect(
          actualIndexedPolicy[rank].idx,
          `top move rank #${rank + 1} must match native KataGo`
        ).toBe(nativeIndexedPolicy[rank].idx);
      }

      // If this position has an active ko point, verify illegal ko rule enforcement
      if (state.koPoint >= 0) {
        expect(actual.policy[state.koPoint], 'active koPoint must be masked as illegal (-1)').toBeLessThan(0);
      }
    }, 60000);
  }

  // 3. Multi-Batch Forward Equivalence Test (batch=1 vs batch=4)
  it('produces identical outputs between single-sample and batched inference', async () => {
    const stones = new Uint8Array(361);
    stones[3 * 19 + 3] = 1; // Black at D16
    stones[15 * 19 + 15] = 2; // White at Q4

    const spatialSingle = new Float32Array(19 * 19 * 22);
    const globalSingle = new Float32Array(19);
    fillInputsV7Fast({
      stones,
      koPoint: -1,
      currentPlayer: 'black',
      recentMoves: [],
      komi: 7.5,
      outSpatial: spatialSingle,
      outGlobal: globalSingle,
    });

    const b1Spatial = tf.tensor4d(spatialSingle, [1, 19, 19, 22]);
    const b1Global = tf.tensor2d(globalSingle, [1, 19]);
    const b1Out = model.forwardPolicyValue(b1Spatial, b1Global);

    const b1Policy = await b1Out.policy.data();
    const b1Val = await b1Out.value.data();
    b1Spatial.dispose();
    b1Global.dispose();
    tf.dispose(b1Out);

    // Now test batch=4 (replicated 4 times)
    const b4SpatialData = new Float32Array(4 * 19 * 19 * 22);
    const b4GlobalData = new Float32Array(4 * 19);
    for (let b = 0; b < 4; b++) {
      b4SpatialData.set(spatialSingle, b * spatialSingle.length);
      b4GlobalData.set(globalSingle, b * globalSingle.length);
    }
    const b4Spatial = tf.tensor4d(b4SpatialData, [4, 19, 19, 22]);
    const b4Global = tf.tensor2d(b4GlobalData, [4, 19]);
    const b4Out = model.forwardPolicyValue(b4Spatial, b4Global);

    const b4Policy = await b4Out.policy.data();
    const b4Val = await b4Out.value.data();
    b4Spatial.dispose();
    b4Global.dispose();
    tf.dispose(b4Out);

    // Verify all 4 slices in the batch match batch=1 within numerical precision
    for (let b = 0; b < 4; b++) {
      for (let i = 0; i < 3; i++) {
        expect(Math.abs(b4Val[b * 3 + i] - b1Val[i])).toBeLessThan(1e-5);
      }
      const slicePolicy = b4Policy.subarray(b * 19 * 19 * 2, (b + 1) * 19 * 19 * 2);
      for (let i = 0; i < slicePolicy.length; i++) {
        expect(Math.abs(slicePolicy[i] - b1Policy[i])).toBeLessThan(1e-5);
      }
    }
  }, 90000);
});
