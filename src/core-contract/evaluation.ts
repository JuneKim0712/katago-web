/**
 * Contract v1: Backend-Neutral Evaluation Boundary.
 * Decouples the KataGo search tree from the neural network inference engine (TFJS, ONNX Runtime, etc.).
 */

import type { TensorLayout } from '../core-runtime/features.js';

export interface EvaluationCapabilities {
  readonly supportedLayouts: readonly TensorLayout[];
  readonly maxBatchSize: number;
  readonly supportsFp16: boolean;
  readonly supportsOwnership: boolean;
  readonly supportsIntermediateActivations: boolean;
}

export interface EvaluationBatch {
  readonly batchSize: number;
  readonly width: number;
  readonly height: number;
  readonly layout: TensorLayout;
  readonly spatial: Float32Array; // [batchSize, area * 22] or [batchSize, 22 * area]
  readonly global: Float32Array; // [batchSize, 19]
  readonly generation: number;
}

export interface SingleEvaluationValue {
  readonly winrate: number; // 0..1 from current player's perspective (or Black's perspective)
  readonly scoreLead: number;
  readonly scoreMean: number;
  readonly scoreStdev: number;
  readonly utility: number;
  readonly noResultProb: number;
}

export interface EvaluationResult {
  readonly batchSize: number;
  readonly generation: number;
  readonly policy: Float32Array; // [batchSize, 362], logits or softmax probabilities
  readonly values: readonly SingleEvaluationValue[];
  readonly ownership?: Float32Array; // [batchSize, 361]
  readonly intermediateActivations?: Readonly<Record<string, Float32Array>>;
}

export class EvaluationCancelledError extends Error {
  readonly generation: number;
  constructor(generation: number) {
    super(`Evaluation batch cancelled for obsolete generation ${generation}`);
    this.name = 'EvaluationCancelledError';
    this.generation = generation;
  }
}

export interface EvaluationBackend {
  readonly name: string;
  readonly capabilities: EvaluationCapabilities;

  /**
   * Evaluates a batch of positions. Throws EvaluationCancelledError if the batch generation is obsolete.
   */
  evaluate(batch: EvaluationBatch): Promise<EvaluationResult>;

  /**
   * Optional warmup method to allocate buffers, compile shaders/kernels, and verify pipeline readiness.
   */
  warmup?(dimensions: { width: number; height: number; batchSize: number }): Promise<void>;

  /**
   * Releases GPU/CPU memory and resources.
   */
  dispose(): Promise<void> | void;
}
