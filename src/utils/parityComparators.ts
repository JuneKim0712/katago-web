/**
 * Lossless bitwise parity assertion utilities for Float32 and Float64.
 */

export function assertFloat32BitsEqual(
  actual: Float32Array,
  expected: Float32Array,
  context = 'Float32Array'
): void {
  if (actual.length !== expected.length) {
    throw new Error(
      `[${context}] Length mismatch: actual=${actual.length}, expected=${expected.length}`
    );
  }

  // Create Uint32 views respecting byteOffset and byteLength
  const actualU32 = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedU32 = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);

  for (let i = 0; i < actual.length; i++) {
    const actBits = actualU32[i];
    const expBits = expectedU32[i];
    if (actBits !== expBits) {
      // Allow identical NaNs if bits match, otherwise fail
      const actVal = actual[i];
      const expVal = expected[i];
      throw new Error(
        `[${context}] Bitwise mismatch at index ${i}: ` +
        `actual=0x${actBits.toString(16).padStart(8, '0')} (${actVal}), ` +
        `expected=0x${expBits.toString(16).padStart(8, '0')} (${expVal})`
      );
    }
  }
}

export function assertNumbersEqual(actual: number, expected: number, context = 'number'): void {
  if (!Object.is(actual, expected)) {
    throw new Error(
      `[${context}] Number mismatch: actual=${actual} (${1 / actual === -Infinity ? '-0' : actual}), ` +
      `expected=${expected} (${1 / expected === -Infinity ? '-0' : expected})`
    );
  }
}

export function assertFloat64BitsEqual(
  actual: Float64Array | number[],
  expected: Float64Array | number[],
  context = 'Float64'
): void {
  if (actual.length !== expected.length) {
    throw new Error(
      `[${context}] Length mismatch: actual=${actual.length}, expected=${expected.length}`
    );
  }

  const actArr = actual instanceof Float64Array ? actual : new Float64Array(actual);
  const expArr = expected instanceof Float64Array ? expected : new Float64Array(expected);

  const actU32 = new Uint32Array(actArr.buffer, actArr.byteOffset, actArr.length * 2);
  const expU32 = new Uint32Array(expArr.buffer, expArr.byteOffset, expArr.length * 2);

  for (let i = 0; i < actU32.length; i++) {
    if (actU32[i] !== expU32[i]) {
      const idx = Math.floor(i / 2);
      throw new Error(
        `[${context}] Float64 bitwise mismatch at index ${idx}: ` +
        `actual=${actArr[idx]}, expected=${expArr[idx]}`
      );
    }
  }
}

/**
 * Lossless binary serialization of Float32Array to byte buffer.
 */
export function encodeFloat32ToBinary(arr: Float32Array): Uint8Array {
  const bytes = new Uint8Array(arr.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < arr.length; i++) {
    view.setFloat32(i * 4, arr[i], true); // little-endian
  }
  return bytes;
}

/**
 * Lossless binary deserialization of Float32Array from byte buffer.
 */
export function decodeBinaryToFloat32(bytes: Uint8Array): Float32Array {
  const arr = new Float32Array(bytes.byteLength / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < arr.length; i++) {
    arr[i] = view.getFloat32(i * 4, true);
  }
  return arr;
}
