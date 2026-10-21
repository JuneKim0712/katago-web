import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { assertFloat32BitsEqual } from '../../src/utils/parityComparators.js';

describe('inferenceExact bitwise parity', () => {
  it('produces bitwise identical outputs for identical graph evaluations', async () => {
    const input = tf.tensor4d(new Float32Array(19 * 19 * 22).fill(0.125), [1, 19, 19, 22]);
    const weights = tf.tensor4d(new Float32Array(3 * 3 * 22 * 32).fill(0.0625), [3, 3, 22, 32]);

    const conv1 = tf.conv2d(input, weights, 1, 'same');
    const out1 = (await conv1.data()) as Float32Array;

    const conv2 = tf.conv2d(input, weights, 1, 'same');
    const out2 = (await conv2.data()) as Float32Array;

    assertFloat32BitsEqual(out1, out2, 'conv2d deterministic parity');

    tf.dispose([input, weights, conv1, conv2]);
  });
});
