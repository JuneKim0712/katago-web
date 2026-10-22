// Patch ORT WebGPU WGSL Pow vectorization mismatch bug before importing ORT
if (typeof GPUDevice !== 'undefined' && GPUDevice.prototype && GPUDevice.prototype.createShaderModule) {
  const origCreateShaderModule = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function(desc) {
    if (desc.code && desc.code.includes('pow_v(a, b);') && desc.code.includes('alias input_b_value_t = f32;')) {
      desc.code = desc.code.replace('output[global_idx]=pow_v(a, b);', 'output[global_idx]=pow_v(a, vec4<f32>(b));');
    }
    return origCreateShaderModule.apply(this, arguments);
  };
}

import * as ort from 'onnxruntime-web/webgpu';
import { nhwcToNchw, extractMask, nchwToNhwc } from './adapters.js';

let session = null;

self.onmessage = async (e) => {
  const { action, id } = e.data;

  try {
    if (action === 'init') {
      const t0 = performance.now();
      ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ['webgpu'],
      });
      const loadMs = performance.now() - t0;

      self.postMessage({
        id,
        action: 'init_done',
        loadMs,
        inputs: session.inputNames,
        outputs: session.outputNames,
      });
    } else if (action === 'run') {
      if (!session) throw new Error('Session not initialized');
      const { batchSize, warmupCount, iterationCount, spatial, global } = e.data;
      const spatialData = new Float32Array(spatial);
      const globalData = new Float32Array(global);

      // Warmup phase
      const tWarm0 = performance.now();
      for (let i = 0; i < warmupCount; i++) {
        const spatialNCHW = nhwcToNchw(spatialData, batchSize, 19, 19, 22);
        const maskData = extractMask(spatialData, batchSize, 19, 19, 22);
        const feeds = {
          InputSpatial: new ort.Tensor('float32', spatialNCHW, [batchSize, 22, 19, 19]),
          InputGlobal: new ort.Tensor('float32', globalData, [batchSize, 19, 1, 1]),
          InputMask: new ort.Tensor('float32', maskData, [batchSize, 1, 19, 19]),
        };
        const out = await session.run(feeds);
        const [pol, pass, val, score, own] = await Promise.all([
          out.OutputPolicy.getData ? out.OutputPolicy.getData() : Promise.resolve(out.OutputPolicy.data),
          out.OutputPolicyPass.getData ? out.OutputPolicyPass.getData() : Promise.resolve(out.OutputPolicyPass.data),
          out.OutputValue.getData ? out.OutputValue.getData() : Promise.resolve(out.OutputValue.data),
          out.OutputScoreValue.getData ? out.OutputScoreValue.getData() : Promise.resolve(out.OutputScoreValue.data),
          out.OutputOwnership.getData ? out.OutputOwnership.getData() : Promise.resolve(out.OutputOwnership.data),
        ]);
        nchwToNhwc(pol, batchSize, 2, 19, 19);
        feeds.InputSpatial.dispose?.();
        feeds.InputGlobal.dispose?.();
        feeds.InputMask.dispose?.();
      }
      const warmupMs = performance.now() - tWarm0;

      // Timed phase
      const timings = [];
      let lastOutputs = null;

      for (let i = 0; i < iterationCount; i++) {
        const start = performance.now();
        // 1. Adapt canonical input layout
        const spatialNCHW = nhwcToNchw(spatialData, batchSize, 19, 19, 22);
        const maskData = extractMask(spatialData, batchSize, 19, 19, 22);

        // 2. Create input tensors
        const feeds = {
          InputSpatial: new ort.Tensor('float32', spatialNCHW, [batchSize, 22, 19, 19]),
          InputGlobal: new ort.Tensor('float32', globalData, [batchSize, 19, 1, 1]),
          InputMask: new ort.Tensor('float32', maskData, [batchSize, 1, 19, 19]),
        };

        // 3. Execute session.run
        const out = await session.run(feeds);

        // 4. Await ALL selected output readbacks
        const [pol, pass, val, score, own] = await Promise.all([
          out.OutputPolicy.getData ? out.OutputPolicy.getData() : Promise.resolve(out.OutputPolicy.data),
          out.OutputPolicyPass.getData ? out.OutputPolicyPass.getData() : Promise.resolve(out.OutputPolicyPass.data),
          out.OutputValue.getData ? out.OutputValue.getData() : Promise.resolve(out.OutputValue.data),
          out.OutputScoreValue.getData ? out.OutputScoreValue.getData() : Promise.resolve(out.OutputScoreValue.data),
          out.OutputOwnership.getData ? out.OutputOwnership.getData() : Promise.resolve(out.OutputOwnership.data),
        ]);

        // 5. Normalize output layout to agreed representation: Policy NHWC [N, 19, 19, 2]
        const polNHWC = nchwToNhwc(pol, batchSize, 2, 19, 19);
        const elapsedMs = performance.now() - start;

        // 6. Dispose per-iteration resources
        feeds.InputSpatial.dispose?.();
        feeds.InputGlobal.dispose?.();
        feeds.InputMask.dispose?.();

        timings.push(elapsedMs);

        if (i === iterationCount - 1) {
          lastOutputs = {
            policy: Array.from(polNHWC),
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
      if (session) {
        await session.release();
        session = null;
      }
      self.postMessage({ id, action: 'dispose_done' });
    }
  } catch (err) {
    self.postMessage({ id, error: err.message, stack: err.stack });
  }
};
