import { describe, expect, it } from 'vitest';
import {
  assertFloat32BitsEqual,
  assertNumbersEqual,
  assertFloat64BitsEqual,
  encodeFloat32ToBinary,
  decodeBinaryToFloat32,
} from '../../src/utils/parityComparators.js';

describe('Parity Comparators', () => {
  it('detects identical float32 arrays including signed zero', () => {
    const a = new Float32Array([1.0, 2.5, -0.0, 0.0, 1e-10]);
    const b = new Float32Array([1.0, 2.5, -0.0, 0.0, 1e-10]);
    expect(() => assertFloat32BitsEqual(a, b)).not.toThrow();

    // Differing signed zeros should throw
    const c = new Float32Array([1.0, 2.5, 0.0, 0.0, 1e-10]);
    expect(() => assertFloat32BitsEqual(a, c)).toThrow(/Bitwise mismatch at index 2/);
  });

  it('handles subarray offsets correctly', () => {
    const buffer = new ArrayBuffer(64);
    const a = new Float32Array(buffer, 8, 4);
    const b = new Float32Array(4);
    a.set([1, 2, 3, 4]);
    b.set([1, 2, 3, 4]);
    expect(() => assertFloat32BitsEqual(a, b)).not.toThrow();

    b[2] = 3.000001;
    expect(() => assertFloat32BitsEqual(a, b)).toThrow(/Bitwise mismatch at index 2/);
  });

  it('detects length differences', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([1, 2]);
    expect(() => assertFloat32BitsEqual(a, b)).toThrow(/Length mismatch/);
  });

  it('handles Object.is exact number comparison', () => {
    expect(() => assertNumbersEqual(0, 0)).not.toThrow();
    expect(() => assertNumbersEqual(-0, -0)).not.toThrow();
    expect(() => assertNumbersEqual(NaN, NaN)).not.toThrow();
    expect(() => assertNumbersEqual(0, -0)).toThrow(/Number mismatch/);
  });

  it('lossless roundtrip binary encoding', () => {
    const orig = new Float32Array([0, -0, 1.5, -3.14159, 1e30, -1e-30]);
    const bin = encodeFloat32ToBinary(orig);
    const restored = decodeBinaryToFloat32(bin);
    assertFloat32BitsEqual(orig, restored);
  });
});
