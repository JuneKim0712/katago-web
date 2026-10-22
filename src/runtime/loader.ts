import { OrtWebGpuSession } from '../inference/ort/session';

export interface KatagoWasmModule {
  HEAPU8: Uint8Array;
  _katago_init(): number;
  _katago_load_model(
    namePtr: number,
    jsonPtr: number,
    maxBatchSize: number,
    nnXLen: number,
    nnYLen: number,
    numThreads: number
  ): number;
  _katago_post_analyze(queryPtr: number): number;
  _katago_poll_response(): number;
  _katago_free_response(ptr: number): void;
  _katago_cancel(requestIdPtr: number): number;
  _katago_reset(): number;
  _katago_destroy(): number;
  _katago_get_slot_desc_ptr(slotId: number): number;
  stringToUTF8(str: string, outPtr: number, maxBytesToWrite: number): void;
  UTF8ToString(ptr: number): string;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
}

export interface KatagoNativeInstance {
  init(): void;
  loadModel(
    name: string,
    modelJson: string,
    maxBatchSize: number,
    nnXLen: number,
    nnYLen: number,
    numThreads: number
  ): void;
  postAnalyze(queryJson: string): number;
  pollResponse(): string | null;
  cancel(requestId: string): void;
  reset(): void;
  destroy(): void;
}

export async function createKatagoWasmInstance(
  wasmModuleFactory: (options: Record<string, unknown>) => Promise<KatagoWasmModule>,
  ortSession: OrtWebGpuSession
): Promise<KatagoNativeInstance> {
  let wasmMod: KatagoWasmModule | null = null;

  // JSPI bridge handler called on coordinator when evaluator pthreads suspend
  const onPthreadEvalBatch = async (
    slotId: number,
    threadId: number,
    generation?: number,
    seqLow?: number,
    seqHigh?: number
  ): Promise<number> => {
    if (!wasmMod) return -1;
    try {
      const descOffset = wasmMod._katago_get_slot_desc_ptr(slotId);
      if (descOffset === 0) {
        return -2;
      }

      // Memory buffer growth detection: always pass latest wasmMod.HEAPU8.buffer
      await ortSession.runBatch(wasmMod.HEAPU8.buffer, descOffset, generation);

      if (threadId !== 0) {
        const worker = (wasmMod as any).PThread?.pthreads?.[threadId];
        if (worker) {
          worker.postMessage({
            target: 'setimmediate',
            cmd: 'eval_response',
            slot_id: slotId,
            generation,
            seq_low: seqLow,
            seq_high: seqHigh,
            result: 0,
          });
        }
      }
      return 0;
    } catch (err) {
      if (threadId !== 0) {
        const worker = (wasmMod as any).PThread?.pthreads?.[threadId];
        if (worker) {
          worker.postMessage({
            target: 'setimmediate',
            cmd: 'eval_response',
            slot_id: slotId,
            generation,
            seq_low: seqLow,
            seq_high: seqHigh,
            result: -3,
          });
        }
      }
      return -3;
    }
  };

  const moduleOptions: Record<string, unknown> = {
    onPthreadEvalBatch,
    js_eval_batch_async: onPthreadEvalBatch,
  };

  wasmMod = await wasmModuleFactory(moduleOptions);

  function allocateUtf8(str: string): number {
    if (!wasmMod) throw new Error('Wasm not initialized');
    const byteLen = new TextEncoder().encode(str).length + 1;
    const ptr = wasmMod._malloc(byteLen);
    wasmMod.stringToUTF8(str, ptr, byteLen);
    return ptr;
  }

  return {
    init(): void {
      if (!wasmMod) throw new Error('Wasm not initialized');
      const res = wasmMod._katago_init();
      if (res !== 0) throw new Error(`katago_init failed with code ${res}`);
    },

    loadModel(
      name: string,
      modelJson: string,
      maxBatchSize: number,
      nnXLen: number,
      nnYLen: number,
      numThreads: number
    ): void {
      if (!wasmMod) throw new Error('Wasm not initialized');
      const namePtr = allocateUtf8(name);
      const jsonPtr = allocateUtf8(modelJson);
      try {
        const res = wasmMod._katago_load_model(namePtr, jsonPtr, maxBatchSize, nnXLen, nnYLen, numThreads);
        if (res !== 0) throw new Error(`katago_load_model failed with code ${res}`);
      } finally {
        wasmMod._free(namePtr);
        wasmMod._free(jsonPtr);
      }
    },

    postAnalyze(queryJson: string): number {
      if (!wasmMod) throw new Error('Wasm not initialized');
      const queryPtr = allocateUtf8(queryJson);
      try {
        return wasmMod._katago_post_analyze(queryPtr);
      } finally {
        wasmMod._free(queryPtr);
      }
    },

    pollResponse(): string | null {
      if (!wasmMod) return null;
      const ptr = wasmMod._katago_poll_response();
      if (ptr === 0) return null;
      try {
        return wasmMod.UTF8ToString(ptr);
      } finally {
        wasmMod._katago_free_response(ptr);
      }
    },

    cancel(requestId: string): void {
      if (!wasmMod) return;
      const idPtr = allocateUtf8(requestId);
      try {
        wasmMod._katago_cancel(idPtr);
      } finally {
        wasmMod._free(idPtr);
      }
    },

    reset(): void {
      if (!wasmMod) return;
      wasmMod._katago_reset();
    },

    destroy(): void {
      if (!wasmMod) return;
      wasmMod._katago_destroy();
    },
  };
}
