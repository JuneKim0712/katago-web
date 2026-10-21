import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { parseKataGoModelV8, KataGoModelV8Tf, MctsSearch } from '../dist/index.js';

export interface HarnessConfig {
  modelUrl: string;
  visits: number;
  batchSize: number;
}

export async function runWebGpuBenchmark(config: HarnessConfig): Promise<{
  backend: string;
  completedVisits: number;
  elapsedMs: number;
  visitsPerSec: number;
}> {
  await tf.setBackend('webgpu');
  await tf.ready();

  if (tf.getBackend() !== 'webgpu') {
    throw new Error(`WebGPU initialization failed. Current backend: ${tf.getBackend()}`);
  }

  const response = await fetch(config.modelUrl);
  const buffer = new Uint8Array(await response.arrayBuffer());
  const parsed = parseKataGoModelV8(buffer);
  const model = new KataGoModelV8Tf(parsed, {
    fuseRotary: true,
    fuseRmsNorm: true,
    fuseTransformerProjections: false,
  });

  const search = await MctsSearch.create({
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
  });

  const initialVisits = search.getAnalysisLight({ topK: 1, analysisPvLen: 1 }).rootVisits;
  const t0 = performance.now();
  await search.run({ visits: config.visits, maxTimeMs: 60000, batchSize: config.batchSize });
  const elapsedMs = performance.now() - t0;
  const analysis = search.getAnalysisLight({ topK: 1, analysisPvLen: 1 });
  const completedVisits = analysis.rootVisits - initialVisits;

  model.dispose();

  return {
    backend: tf.getBackend(),
    completedVisits,
    elapsedMs,
    visitsPerSec: completedVisits / (elapsedMs / 1000),
  };
}
