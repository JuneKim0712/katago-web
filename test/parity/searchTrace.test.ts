import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { MctsSearch, type KataGoModelV8Tf } from '../../src/index.js';
import { createSeededRandomSource } from '../../src/utils/random.js';
import { assertFloat32BitsEqual, assertNumbersEqual } from '../../src/utils/parityComparators.js';

function fakeSeededModel(seed: number) {
  const prng = createSeededRandomSource(seed);
  const forward = (spatial: tf.Tensor4D) => {
    const n = spatial.shape[0];
    const policyData = new Float32Array(n * 19 * 19 * 1);
    for (let i = 0; i < policyData.length; i++) {
      policyData[i] = (prng.next() - 0.5) * 2;
    }
    const policyPassData = new Float32Array(n * 1);
    for (let i = 0; i < policyPassData.length; i++) {
      policyPassData[i] = -5.0 + prng.next();
    }
    const valueData = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v0 = prng.next();
      const v1 = prng.next();
      const v2 = prng.next();
      const sum = v0 + v1 + v2;
      valueData[i * 3 + 0] = v0 / sum;
      valueData[i * 3 + 1] = v1 / sum;
      valueData[i * 3 + 2] = v2 / sum;
    }
    const scoreValueData = new Float32Array(n * 4);
    for (let i = 0; i < scoreValueData.length; i++) {
      scoreValueData[i] = (prng.next() - 0.5) * 10;
    }
    const ownershipData = new Float32Array(n * 361);
    for (let i = 0; i < ownershipData.length; i++) {
      ownershipData[i] = (prng.next() - 0.5) * 2;
    }

    return {
      policy: tf.tensor4d(policyData, [n, 19, 19, 1]),
      policyPass: tf.tensor2d(policyPassData, [n, 1]),
      value: tf.tensor2d(valueData, [n, 3]),
      scoreValue: tf.tensor2d(scoreValueData, [n, 4]),
      ownership: tf.tensor2d(ownershipData, [n, 361]),
    };
  };

  return {
    policyOutChannels: 1,
    modelVersion: 17,
    inputBuffers: { supportsWrites: false },
    forwardPolicyValue: forward,
    forward,
  } as unknown as KataGoModelV8Tf;
}

async function createSearch(model: KataGoModelV8Tf, seed: number) {
  const rand = createSeededRandomSource(seed);
  return MctsSearch.create({
    model,
    board: Array.from({ length: 19 }, () => Array(19).fill(null)),
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
    randomSource: rand,
  });
}

describe('searchTrace determinism', () => {
  it('produces bitwise identical analysis snapshots across identical runs', async () => {
    const modelA = fakeSeededModel(42);
    const searchA = await createSearch(modelA, 100);
    await searchA.run({ visits: 16, maxTimeMs: 1000, batchSize: 2 });
    const analysisA = searchA.getAnalysis({ topK: 5, analysisPvLen: 3 });

    const modelB = fakeSeededModel(42);
    const searchB = await createSearch(modelB, 100);
    await searchB.run({ visits: 16, maxTimeMs: 1000, batchSize: 2 });
    const analysisB = searchB.getAnalysis({ topK: 5, analysisPvLen: 3 });

    assertNumbersEqual(analysisA.rootVisits, analysisB.rootVisits, 'rootVisits');
    assertNumbersEqual(analysisA.rootWinRate, analysisB.rootWinRate, 'rootWinRate');
    assertNumbersEqual(analysisA.rootScoreLead, analysisB.rootScoreLead, 'rootScoreLead');
    assertNumbersEqual(analysisA.rootScoreSelfplay, analysisB.rootScoreSelfplay, 'rootScoreSelfplay');

    assertFloat32BitsEqual(analysisA.policy as Float32Array, analysisB.policy as Float32Array, 'policy');
    assertFloat32BitsEqual(analysisA.ownership as Float32Array, analysisB.ownership as Float32Array, 'ownership');

    expect(analysisA.moves.length).toBe(analysisB.moves.length);
    for (let i = 0; i < analysisA.moves.length; i++) {
      const ma = analysisA.moves[i]!;
      const mb = analysisB.moves[i]!;
      assertNumbersEqual(ma.x, mb.x, `moves[${i}].x`);
      assertNumbersEqual(ma.y, mb.y, `moves[${i}].y`);
      assertNumbersEqual(ma.visits, mb.visits, `moves[${i}].visits`);
      assertNumbersEqual(ma.winRate, mb.winRate, `moves[${i}].winRate`);
      assertNumbersEqual(ma.scoreLead, mb.scoreLead, `moves[${i}].scoreLead`);
      expect(ma.pv).toEqual(mb.pv);
    }
  });

  it('produces bitwise identical analysis snapshots across standalone analyzeMcts runs', async () => {
    const { analyzeMcts } = await import('../../src/engine/analyzeMcts.js');
    const board = Array.from({ length: 19 }, () => Array(19).fill(null));

    const modelA = fakeSeededModel(99);
    const randA = createSeededRandomSource(123);
    const resA = await analyzeMcts({
      model: modelA,
      board,
      currentPlayer: 'black',
      moveHistory: [],
      rules: 'japanese',
      komi: 6.5,
      nnRandomize: false,
      visits: 16,
      maxTimeMs: 1000,
      batchSize: 2,
      topK: 5,
      analysisPvLen: 3,
      randomSource: randA,
    });

    const modelB = fakeSeededModel(99);
    const randB = createSeededRandomSource(123);
    const resB = await analyzeMcts({
      model: modelB,
      board,
      currentPlayer: 'black',
      moveHistory: [],
      rules: 'japanese',
      komi: 6.5,
      nnRandomize: false,
      visits: 16,
      maxTimeMs: 1000,
      batchSize: 2,
      topK: 5,
      analysisPvLen: 3,
      randomSource: randB,
    });

    assertNumbersEqual(resA.rootVisits, resB.rootVisits, 'standalone rootVisits');
    assertNumbersEqual(resA.rootWinRate, resB.rootWinRate, 'standalone rootWinRate');
    assertNumbersEqual(resA.rootScoreLead, resB.rootScoreLead, 'standalone rootScoreLead');
    assertFloat32BitsEqual(resA.policy as Float32Array, resB.policy as Float32Array, 'standalone policy');
    assertFloat32BitsEqual(resA.ownership as Float32Array, resB.ownership as Float32Array, 'standalone ownership');
    expect(resA.moves.length).toBe(resB.moves.length);
    for (let i = 0; i < resA.moves.length; i++) {
      assertNumbersEqual(resA.moves[i]!.visits, resB.moves[i]!.visits, `standalone moves[${i}].visits`);
      assertNumbersEqual(resA.moves[i]!.winRate, resB.moves[i]!.winRate, `standalone moves[${i}].winRate`);
      expect(resA.moves[i]!.pv).toEqual(resB.moves[i]!.pv);
    }
  });
});
