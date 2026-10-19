import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { MctsSearch, type KataGoModelV8Tf } from '../src/index.js';

function fakeModel(delayMs = 5) {
  let pendingReads = 0;
  const boards: string[] = [];
  let onForward = () => {};
  const forward = (spatial: tf.Tensor4D) => {
    const n = spatial.shape[0], input = spatial.dataSync();
    for (let b = 0; b < n; b++) {
      const cells: string[] = [];
      for (let p = 0; p < 361; p++) {
        const at = (b * 361 + p) * 22;
        if (input[at + 1] || input[at + 2]) cells.push(`${p}:${input[at + 1]}:${input[at + 2]}`);
      }
      boards.push(cells.join(','));
    }
    const out = { policy: tf.zeros([n, 19, 19, 1]), policyPass: tf.fill([n, 1], -10), value: tf.zeros([n, 3]), scoreValue: tf.zeros([n, 4]) };
    for (const tensor of Object.values(out)) {
      const read = tensor.data.bind(tensor);
      tensor.data = async () => {
        pendingReads++;
        try { await new Promise(resolve => setTimeout(resolve, delayMs)); return await read(); }
        finally { pendingReads--; }
      };
    }
    onForward();
    return out;
  };
  return {
    model: { policyOutChannels: 1, modelVersion: 17, inputBuffers: { supportsWrites: false }, forwardPolicyValue: forward } as unknown as KataGoModelV8Tf,
    boards, pending: () => pendingReads, onForward: (fn: () => void) => { onForward = fn; },
  };
}

async function create(model: KataGoModelV8Tf) {
  return MctsSearch.create({ model, board: Array.from({ length: 19 }, () => Array(19).fill(null)), currentPlayer: 'black',
    moveHistory: [], rules: 'japanese', komi: 6.5, nnRandomize: false, conservativePass: false,
    maxChildren: 64, ownershipMode: 'none', wideRootNoise: 0, rootSymmetrySamples: 1 });
}

describe('pipelined search lifetime and counting', () => {
  it('keeps overlapping input batches distinct and respects the visit cap', async () => {
    const model = fakeModel(), search = await create(model.model);
    await search.run({ visits: 18, maxTimeMs: 1000, batchSize: 4 });
    expect(search.getAnalysisLight({ topK: 1, analysisPvLen: 1 }).rootVisits).toBe(18);
    expect(model.pending()).toBe(0);
    expect(new Set(model.boards.slice(1, 9)).size).toBe(8);
  });

  it('drains prefetched work at a time limit and can resume', async () => {
    const model = fakeModel(20), search = await create(model.model);
    const tensors = tf.memory().numTensors;
    await search.run({ visits: 1000, maxTimeMs: 25, batchSize: 4 });
    const visits = search.getAnalysisLight({ topK: 1, analysisPvLen: 1 }).rootVisits;
    expect(model.pending()).toBe(0);
    expect(tf.memory().numTensors).toBe(tensors);
    await search.run({ visits: visits + 8, maxTimeMs: 1000, batchSize: 4 });
    expect(search.getAnalysisLight({ topK: 1, analysisPvLen: 1 }).rootVisits).toBe(visits + 8);
    expect(model.pending()).toBe(0);
  });

  it('clears cancelled in-flight reservations before returning', async () => {
    const model = fakeModel(), search = await create(model.model);
    let cancelled = false;
    model.onForward(() => { cancelled = true; });
    expect(await search.run({ visits: 32, maxTimeMs: 1000, batchSize: 4, shouldAbort: () => cancelled })).toBe(true);
    expect(model.pending()).toBe(0);
    expect(search.getAnalysisLight({ topK: 1, analysisPvLen: 1 }).rootVisits).toBe(1);
    model.onForward(() => {});
    await search.run({ visits: 18, maxTimeMs: 1000, batchSize: 4 });
    expect(search.getAnalysisLight({ topK: 1, analysisPvLen: 1 }).rootVisits).toBe(18);
  });
});
