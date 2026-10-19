import * as tf from '@tensorflow/tfjs';
import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import type { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

const KERNEL = 'KataGoTransformerRms';
if (!tf.getKernel(KERNEL, 'webgpu')) tf.registerKernel({
  kernelName: KERNEL, backendName: 'webgpu',
  kernelFunc: ({ inputs, attrs, backend }) => {
    const { X, Weight } = inputs;
    if (!X || !Weight || !attrs) throw new Error('Missing RMS normalization input');
    const epsilon = Math.fround(attrs.epsilon as number);
    const [batch, height, width, channels] = X.shape;
    const program: WebGPUProgram = {
      outputShape: X.shape, variableNames: ['X', 'Weight'],
      shaderKey: `kg_rms_full_${epsilon}_${channels}_${width}_${height}`,
      size: true, workgroupSize: [64, 1, 1], dispatchLayout: { x: [0, 1, 2] },
      dispatch: [batch * height * width, 1, 1],
      // Match TF.js ReduceProgram's lane sums and binary tree for C < 4096.
      // Shared FP32 storage separates squares, sums and mean rounding; in
      // particular, squaring must not contract into a fused multiply-add.
      getUserCode: () => `
      var<workgroup> squares: array<f32, ${channels}>;
      var<workgroup> sums: array<f32, 64>;
      var<workgroup> mean: f32;
      fn main(index: i32) {
        let row = index / 64;
        let lane = i32(localId.x);
        let x = row % ${width};
        let y = (row / ${width}) % ${height};
        let n = row / ${width * height};
        for (var c = lane; c < ${channels}; c += 64) {
          let value = getX(n, y, x, c);
          squares[c] = value * value;
        }
        workgroupBarrier();
        var total = 0.0;
        for (var c = lane; c < ${channels}; c += 64) { total += squares[c]; }
        sums[lane] = total;
        workgroupBarrier();
        var count = ${Math.min(channels, 64)};
        while (count > 1) {
          let half = (count + 1) / 2;
          if (lane < count / 2) { total += sums[lane + half]; sums[lane] = total; }
          count = half;
          workgroupBarrier();
        }
        if (lane == 0) { mean = total / ${channels}.0; }
        workgroupBarrier();
        let r = inverseSqrt(mean + ${epsilon.toExponential()}f);
        for (var c = lane; c < ${channels}; c += 64) {
          let normalized = getX(n, y, x, c) * r;
          setOutputAtIndex(row * ${channels} + c, normalized * getWeight(0, 0, 0, c));
        }
      }`,
    };
    return (backend as WebGPUBackend).runWebGPUProgram(program, [X, Weight], 'float32');
  },
});

/** Use the fused path only for validated 64-channel groups within shared-memory limits. */
export function transformerRmsWebGpu(x: tf.Tensor4D, weight: tf.Tensor4D, epsilon: number): tf.Tensor4D {
  if (x.shape[3] > 2048 || x.shape[3] % 64 !== 0 || tf.getBackend() !== 'webgpu') {
    return tf.tidy(() => tf.mul(tf.mul(x, tf.rsqrt(tf.add(tf.mean(tf.square(x), -1, true), epsilon))), weight) as tf.Tensor4D);
  }
  return tf.engine().runKernel(KERNEL, { X: x, Weight: weight }, { epsilon });
}
