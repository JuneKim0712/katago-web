/**
 * Deterministic mock evaluator implementing Contract v1 EvaluationBackend.
 * Used for testing search tree integration, batching, cancellation, and lifecycle without a GPU.
 */

import {
  type EvaluationBackend,
  type EvaluationCapabilities,
  type EvaluationBatch,
  type EvaluationResult,
  type SingleEvaluationValue,
  EvaluationCancelledError,
} from '../../src/core-contract/evaluation.js';

export interface FakeEvaluatorOptions {
  latencyMs?: number;
  fixedWinrate?: number;
  fixedScoreLead?: number;
}

export class FakeEvaluator implements EvaluationBackend {
  readonly name = 'FakeEvaluator';
  readonly capabilities: EvaluationCapabilities = Object.freeze({
    supportedLayouts: ['NHWC', 'NCHW'],
    maxBatchSize: 64,
    supportsFp16: false,
    supportsOwnership: true,
    supportsIntermediateActivations: true,
  });

  currentGeneration = 0;
  callsCount = 0;
  batchesProcessed = 0;
  positionsEvaluated = 0;
  disposed = false;
  latencyMs = 0;

  fixedWinrate = 0.52;
  fixedScoreLead = 1.2;

  constructor(options: FakeEvaluatorOptions = {}) {
    if (options.latencyMs !== undefined) this.latencyMs = options.latencyMs;
    if (options.fixedWinrate !== undefined) this.fixedWinrate = options.fixedWinrate;
    if (options.fixedScoreLead !== undefined) this.fixedScoreLead = options.fixedScoreLead;
  }

  async warmup(_dimensions: { width: number; height: number; batchSize: number }): Promise<void> {
    if (this.disposed) throw new Error('Cannot warmup a disposed evaluator');
    // Simulated warmup
  }

  async evaluate(batch: EvaluationBatch): Promise<EvaluationResult> {
    if (this.disposed) {
      throw new Error('EvaluationBackend has been disposed');
    }

    if (batch.generation < this.currentGeneration) {
      throw new EvaluationCancelledError(batch.generation);
    }

    if (this.latencyMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.latencyMs));
      // Recheck generation after simulated latency
      if (batch.generation < this.currentGeneration) {
        throw new EvaluationCancelledError(batch.generation);
      }
    }

    this.callsCount++;
    this.batchesProcessed++;
    this.positionsEvaluated += batch.batchSize;

    const area = batch.width * batch.height;
    const policySize = area + 1; // area moves + 1 pass move
    const totalPolicyElements = batch.batchSize * policySize;
    const policy = new Float32Array(totalPolicyElements);

    const values: SingleEvaluationValue[] = [];
    const ownership = new Float32Array(batch.batchSize * area);

    for (let b = 0; b < batch.batchSize; b++) {
      const pOffset = b * policySize;
      const oOffset = b * area;

      // Construct a valid normalized probability distribution
      let policySum = 0;
      for (let loc = 0; loc < area; loc++) {
        const x = loc % batch.width;
        const y = Math.floor(loc / batch.width);
        // Center weight
        const dx = x - (batch.width - 1) / 2;
        const dy = y - (batch.height - 1) / 2;
        const distSq = dx * dx + dy * dy;
        const unnorm = Math.exp(-distSq / 50.0);
        policy[pOffset + loc] = unnorm;
        policySum += unnorm;

        // Slight ownership prior
        ownership[oOffset + loc] = Math.tanh(0.1 * (x - y));
      }

      // Pass move unnormalized weight
      const passUnnorm = 0.01;
      policy[pOffset + area] = passUnnorm;
      policySum += passUnnorm;

      // Normalize
      const invSum = 1.0 / policySum;
      for (let i = 0; i < policySize; i++) {
        policy[pOffset + i] *= invSum;
      }

      values.push(
        Object.freeze({
          winrate: this.fixedWinrate,
          scoreLead: this.fixedScoreLead,
          scoreMean: this.fixedScoreLead,
          scoreStdev: 12.5,
          utility: (this.fixedWinrate - 0.5) * 2.0,
          noResultProb: 0.0,
        })
      );
    }

    return Object.freeze({
      batchSize: batch.batchSize,
      generation: batch.generation,
      policy,
      values: Object.freeze(values),
      ownership,
    });
  }

  dispose(): void {
    this.disposed = true;
  }
}
