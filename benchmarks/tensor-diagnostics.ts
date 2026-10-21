import * as tf from '@tensorflow/tfjs';

export interface MemorySnapshot {
  numTensors: number;
  numDataBuffers: number;
  numBytes: number;
  unreliable?: boolean;
  reasons?: string[];
}

export function captureMemorySnapshot(): MemorySnapshot {
  const mem = tf.memory();
  return {
    numTensors: mem.numTensors,
    numDataBuffers: mem.numDataBuffers,
    numBytes: mem.numBytes,
    unreliable: mem.unreliable,
    reasons: mem.reasons,
  };
}

export async function profileForwardExecution<T>(fn: () => Promise<T> | T): Promise<{
  result: T;
  profile: tf.ProfileInfo;
}> {
  let result!: T;
  const profile = await tf.profile(async () => {
    result = await fn();
  });
  return { result, profile };
}
