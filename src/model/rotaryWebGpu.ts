import * as tf from '@tensorflow/tfjs';
import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import type { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

const KERNEL = 'KataGoProjectedRotary';

function program(shape: number[], names: string[], key: string, code: string): WebGPUProgram {
  return {
    outputShape: shape, variableNames: names, shaderKey: key,
    size: true, workgroupSize: [128, 1, 1], dispatchLayout: { x: [0, 1, 2, 3] },
    dispatch: [Math.ceil(shape.reduce((a, b) => a * b, 1) / 128), 1, 1],
    getUserCode: () => `fn main(index: i32) { if (index < uniforms.size) { ${code} } }`,
  };
}

if (!tf.getKernel(KERNEL, 'webgpu')) tf.registerKernel({
  kernelName: KERNEL, backendName: 'webgpu',
  kernelFunc: ({ inputs, attrs, backend }) => {
    const { X, Cos, Sin } = inputs;
    if (!X || !Cos || !Sin) throw new Error('Missing rotary input');
    const { batch, heads, sequence, dim } = attrs as { batch: number; heads: number; sequence: number; dim: number };
    const gpu = backend as WebGPUBackend;
    const tableHeads = Cos.shape[1];
    const key = `${batch}_${heads}_${sequence}_${dim}_${tableHeads}`;
    // Keep products in an intermediate FP32 buffer. A single multiply-add shader
    // could contract operations and change rounding compared with the original.
    const products = gpu.runWebGPUProgram(program([batch, heads, sequence, dim * 2], ['X', 'Cos', 'Sin'], `kg_rope_products_${key}`, `
      let component = index % 2;
      let outputIndex = index / 2;
      let channel = outputIndex % ${dim};
      let position = (outputIndex / ${dim}) % ${sequence};
      let head = (outputIndex / ${dim * sequence}) % ${heads};
      let batch = outputIndex / ${dim * sequence * heads};
      let pair = channel / 2;
      let source = getX(batch * ${sequence} + position, head * ${dim} + pair * 2 + component);
      let tableHead = head / ${heads / tableHeads};
      var factor: f32;
      if ((channel % 2) == component) { factor = getCos(0, tableHead, position, pair); }
      else { factor = getSin(0, tableHead, position, pair); }
      setOutputAtIndex(index, source * factor);
    `), [X, Cos, Sin], 'float32');
    try {
      return gpu.runWebGPUProgram(program([batch, heads, sequence, dim], ['Products'], `kg_rope_sum_${key}`, `
        let channel = index % ${dim};
        let position = (index / ${dim}) % ${sequence};
        let head = (index / ${dim * sequence}) % ${heads};
        let batch = index / ${dim * sequence * heads};
        let a = getProducts(batch, head, position, channel * 2);
        let b = getProducts(batch, head, position, channel * 2 + 1);
        if ((channel % 2) == 0) { setOutputAtIndex(index, a - b); }
        else { setOutputAtIndex(index, a + b); }
      `), [products], 'float32');
    } finally { gpu.disposeData(products.dataId); }
  },
});

/** Projected [batch*sequence, heads*dim] to rotated [batch, heads, sequence, dim]. */
export function projectedRotaryWebGpu(x: tf.Tensor2D, cos: tf.Tensor4D, sin: tf.Tensor4D, batch: number, heads: number, sequence: number, dim: number): tf.Tensor4D {
  if (dim % 2 !== 0 || heads % cos.shape[1] !== 0) throw new Error('Invalid rotary dimensions');
  return tf.engine().runKernel(KERNEL, { X: x, Cos: cos, Sin: sin }, { batch, heads, sequence, dim });
}
