import { describe, it, expect } from 'vitest';
import { FakeEvaluator } from './fakeEvaluator.js';
import {
  type EvaluationBatch,
  EvaluationCancelledError,
} from '../../src/core-contract/evaluation.js';
import { NUM_SPATIAL_CHANNELS, NUM_GLOBAL_CHANNELS } from '../../src/core-runtime/features.js';

describe('Stage 4: Backend-Neutral Evaluation Contract v1', () => {
  it('adheres to capabilities contract', () => {
    const evaluator = new FakeEvaluator();
    expect(evaluator.capabilities.supportedLayouts).toContain('NHWC');
    expect(evaluator.capabilities.supportedLayouts).toContain('NCHW');
    expect(evaluator.capabilities.maxBatchSize).toBeGreaterThan(0);
    expect(evaluator.capabilities.supportsOwnership).toBe(true);
  });

  it('evaluates a single position batch returning well-formed policy and values', async () => {
    const evaluator = new FakeEvaluator();
    const batch: EvaluationBatch = {
      batchSize: 1,
      width: 19,
      height: 19,
      layout: 'NHWC',
      spatial: new Float32Array(19 * 19 * NUM_SPATIAL_CHANNELS),
      global: new Float32Array(NUM_GLOBAL_CHANNELS),
      generation: 1,
    };

    const result = await evaluator.evaluate(batch);

    expect(result.batchSize).toBe(1);
    expect(result.generation).toBe(1);
    expect(result.policy.length).toBe(362); // 361 + 1
    expect(result.values.length).toBe(1);
    expect(result.values[0].winrate).toBeCloseTo(0.52);
    expect(result.values[0].scoreLead).toBeCloseTo(1.2);
    expect(result.ownership?.length).toBe(361);

    // Verify policy is a normalized probability distribution summing to 1.0
    let sum = 0;
    for (let i = 0; i < 362; i++) {
      expect(result.policy[i]).toBeGreaterThan(0);
      sum += result.policy[i];
    }
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('correctly handles multi-item batches', async () => {
    const evaluator = new FakeEvaluator();
    const batchSize = 4;
    const batch: EvaluationBatch = {
      batchSize,
      width: 19,
      height: 19,
      layout: 'NHWC',
      spatial: new Float32Array(batchSize * 19 * 19 * NUM_SPATIAL_CHANNELS),
      global: new Float32Array(batchSize * NUM_GLOBAL_CHANNELS),
      generation: 1,
    };

    const result = await evaluator.evaluate(batch);

    expect(result.batchSize).toBe(4);
    expect(result.policy.length).toBe(4 * 362);
    expect(result.values.length).toBe(4);
    expect(result.ownership?.length).toBe(4 * 361);
    expect(evaluator.positionsEvaluated).toBe(4);
  });

  it('cancels evaluation for obsolete generations', async () => {
    const evaluator = new FakeEvaluator({ latencyMs: 20 });
    evaluator.currentGeneration = 1;

    const promise = evaluator.evaluate({
      batchSize: 1,
      width: 19,
      height: 19,
      layout: 'NHWC',
      spatial: new Float32Array(19 * 19 * NUM_SPATIAL_CHANNELS),
      global: new Float32Array(NUM_GLOBAL_CHANNELS),
      generation: 1,
    });

    // Advance generation while evaluation is pending
    evaluator.currentGeneration = 2;

    await expect(promise).rejects.toThrow(EvaluationCancelledError);
  });

  it('enforces lifecycle and prevents operations after disposal', async () => {
    const evaluator = new FakeEvaluator();
    evaluator.dispose();

    expect(evaluator.disposed).toBe(true);

    const batch: EvaluationBatch = {
      batchSize: 1,
      width: 19,
      height: 19,
      layout: 'NHWC',
      spatial: new Float32Array(19 * 19 * NUM_SPATIAL_CHANNELS),
      global: new Float32Array(NUM_GLOBAL_CHANNELS),
      generation: 1,
    };

    await expect(evaluator.evaluate(batch)).rejects.toThrow(/disposed/);
  });
});
