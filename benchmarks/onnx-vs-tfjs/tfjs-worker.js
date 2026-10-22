import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import pako from 'pako';
import { parseKataGoModelV8 } from '../../src/model/loadModelV8.js';
import { KataGoModelV8Tf } from '../../src/model/modelV8.js';

let model = null;

self.onmessage = async (e) => {
  const { action, id } = e.data;

  try {
    if (action === 'init') {
      const t0 = performance.now();
      await tf.setBackend('webgpu');
      await tf.ready();
      if (tf.getBackend() !== 'webgpu') {
        throw new Error(`Failed to initialize TFJS WebGPU: got ${tf.getBackend()}`);
      }

      const resp = await fetch(e.data.modelUrl);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching model: ${e.data.modelUrl}`);
      const compressed = new Uint8Array(await resp.arrayBuffer());
      const raw = pako.ungzip(compressed);
      const parsed = parseKataGoModelV8(raw);

      model = new KataGoModelV8Tf(parsed, {
        fuseRotary: true,
        fuseRmsNorm: true,
        fuseTransformerProjections: false, // Production default
      });
      const loadMs = performance.now() - t0;

      self.postMessage({ id, action: 'init_done', loadMs, backend: tf.getBackend() });
    } else if (action === 'run') {
      if (!model) throw new Error('Model not initialized');
      const { batchSize, warmupCount, iterationCount, spatial, global } = e.data;
      const spatialData = new Float32Array(spatial);
      const globalData = new Float32Array(global);

      // Warmup phase
      const tWarm0 = performance.now();
      for (let i = 0; i < warmupCount; i++) {
        const s = tf.tensor4d(spatialData, [batchSize, 19, 19, 22]);
        const g = tf.tensor2d(globalData, [batchSize, 19]);
        const out = model.forward(s, g);
        await Promise.all([
          out.policy.data(),
          out.policyPass.data(),
          out.value.data(),
          out.scoreValue.data(),
          out.ownership.data(),
        ]);
        tf.dispose(out);
        s.dispose();
        g.dispose();
      }
      const warmupMs = performance.now() - tWarm0;

      // Timed phase
      const timings = [];
      let lastOutputs = null;

      for (let i = 0; i < iterationCount; i++) {
        const start = performance.now();
        const s = tf.tensor4d(spatialData, [batchSize, 19, 19, 22]);
        const g = tf.tensor2d(globalData, [batchSize, 19]);
        const out = model.forward(s, g);
        const [policy, pass, val, score, own] = await Promise.all([
          out.policy.data(),
          out.policyPass.data(),
          out.value.data(),
          out.scoreValue.data(),
          out.ownership.data(),
        ]);
        // TFJS outputs are already in canonical layout: policy NHWC [N, 19, 19, 2], ownership NHWC [N, 19, 19, 1]
        const elapsedMs = performance.now() - start;
        tf.dispose(out);
        s.dispose();
        g.dispose();
        timings.push(elapsedMs);

        if (i === iterationCount - 1) {
          lastOutputs = {
            policy: Array.from(policy),
            policyPass: Array.from(pass),
            value: Array.from(val),
            scoreValue: Array.from(score),
            ownership: Array.from(own),
          };
        }
      }

      self.postMessage({
        id,
        action: 'run_done',
        warmupMs,
        timings,
        outputs: lastOutputs,
      });
    } else if (action === 'dispose') {
      if (model) {
        model.dispose();
        model = null;
      }
      self.postMessage({ id, action: 'dispose_done' });
    }
  } catch (err) {
    self.postMessage({ id, error: err.message, stack: err.stack });
  }
};
