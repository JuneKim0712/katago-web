// WebGPU debugging and WGSL Pow vectorization fix
if (typeof GPU !== 'undefined' && GPU.prototype && GPU.prototype.requestAdapter) {
  const origRequestAdapter = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function(options?: any) {
    const adapter = await origRequestAdapter.call(this, options || { powerPreference: 'high-performance' });
    if (!adapter) return null;
    const origDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async function(desc?: any) {
      const limits: Record<string, number> = {};
      for (const k in adapter.limits) {
        if (typeof (adapter.limits as any)[k] === 'number') limits[k] = (adapter.limits as any)[k];
      }
      const features = Array.from(adapter.features).sort();
      const device = await origDevice({
        requiredFeatures: features as any,
        requiredLimits: limits,
      });
      device.addEventListener('uncapturederror', (e: any) => {
        console.error('[WEBGPU UNCAPTURED ERROR]', e.error?.message || e.error);
      });
      device.lost.then((info: any) => {
        console.error('[WEBGPU DEVICE LOST]', info.reason, info.message);
      });
      return device;
    };
    return adapter;
  };
}

if (typeof GPUDevice !== 'undefined' && GPUDevice.prototype && GPUDevice.prototype.createShaderModule) {
  const origCreateShaderModule = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function(desc: any) {
    if (desc.code && desc.code.includes('pow_v(a, b);') && desc.code.includes('alias input_b_value_t = f32;')) {
      console.log('[WEBGPU WGSL PATCH] Applied pow_v patch to shader module');
      desc.code = desc.code.replaceAll('output[global_idx]=pow_v(a, b);', 'output[global_idx]=pow_v(a, vec4<f32>(b));');
    }
    return origCreateShaderModule.apply(this, arguments as any);
  };
}

import * as ort from 'onnxruntime-web';
import {
  DESCRIPTOR_OFFSETS,
  KatagoBridgeErrorCode,
  KatagoSlotStatus,
  readSlotDescriptor,
  SlotDescriptorView,
  validateBufferBounds,
} from '../../runtime/abi';

export interface OrtDiagnostics {
  provider: string;
  disableCpuFallback: boolean;
  inputNames: readonly string[];
  outputNames: readonly string[];
  executionTimeMs: number;
  totalInferences: number;
}

export class OrtWebGpuSession {
  private session: ort.InferenceSession | null = null;
  private readonly inputNames: string[] = [];
  private readonly outputNames: string[] = [];
  private totalInferences = 0;
  private lastExecutionTimeMs = 0;
  private isBroken = false;

  private constructor(
    session: ort.InferenceSession,
    private readonly modelBytesLength: number
  ) {
    this.session = session;
    this.inputNames = [...session.inputNames];
    this.outputNames = [...session.outputNames];
  }

  public static async create(
    modelBytes: Uint8Array,
    options: {
      disableCpuFallback?: boolean;
      preferredLayout?: 'NCHW' | 'NHWC';
    } = {}
  ): Promise<OrtWebGpuSession> {
    const disableCpuFallback = options.disableCpuFallback !== false;

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: [
        {
          name: 'webgpu',
          preferredLayout: options.preferredLayout || 'NCHW',
        },
      ],
      graphOptimizationLevel: 'all',
    };

    if (disableCpuFallback) {
      sessionOptions.extra = {
        session: {
          disable_cpu_ep_fallback: '1',
          use_device_allocator_for_initializers: '1',
        },
      };
    }

    try {
      const session = await ort.InferenceSession.create(modelBytes, sessionOptions);
      return new OrtWebGpuSession(session, modelBytes.byteLength);
    } catch (err: unknown) {
      // Never conceal fallback or failure
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Failed to create ORT WebGPU session (disable_cpu_ep_fallback=${disableCpuFallback}): ${msg}`);
    }
  }

  public async warmup(batchSize = 1, nnXLen = 19, nnYLen = 19): Promise<void> {
    if (!this.session) throw new Error('Session not initialized');
    console.log(`[ORT] Starting warmup inference (batchSize=${batchSize}, ${nnXLen}x${nnYLen})...`);
    const maskData = new Float32Array(batchSize * 1 * nnYLen * nnXLen);
    const spatialData = new Float32Array(batchSize * 22 * nnYLen * nnXLen);
    const globalData = new Float32Array(batchSize * 19 * 1 * 1);
    const feeds: Record<string, ort.Tensor> = {
      InputMask: new ort.Tensor('float32', maskData, [batchSize, 1, nnYLen, nnXLen]),
      InputSpatial: new ort.Tensor('float32', spatialData, [batchSize, 22, nnYLen, nnXLen]),
      InputGlobal: new ort.Tensor('float32', globalData, [batchSize, 19, 1, 1]),
    };
    const t0 = performance.now();
    const out = await this.session.run(feeds);
    console.log(`[ORT] Warmup inference completed in ${(performance.now() - t0).toFixed(1)} ms!`);
    for (const tensor of Object.values(out)) {
      (tensor as any)?.dispose?.();
    }
    (feeds.InputMask as any)?.dispose?.();
    (feeds.InputSpatial as any)?.dispose?.();
    (feeds.InputGlobal as any)?.dispose?.();
  }

  public async runBatch(
    memory: ArrayBufferLike,
    descOffset: number,
    expectedGeneration?: number
  ): Promise<void> {
    if (!this.session) {
      throw new Error('ORT session is disposed or not initialized');
    }
    if (this.isBroken) {
      throw new Error('ORT session is broken from previous timeout/error; must be recreated');
    }

    const desc: SlotDescriptorView = readSlotDescriptor(memory, descOffset);
    const u32 = new Uint32Array(memory, descOffset, 168 / 4);

    if (expectedGeneration !== undefined && desc.engineGeneration !== expectedGeneration) {
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.STALE_GENERATION);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      throw new Error(`Stale generation in descriptor: expected ${expectedGeneration}, got ${desc.engineGeneration}`);
    }

    const batchSize = desc.batchSize;
    if (batchSize <= 0) {
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.INVALID_DESCRIPTOR);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      throw new Error(`Invalid batchSize in slot descriptor: ${batchSize}`);
    }

    const H = desc.nnYLen;
    const W = desc.nnXLen;
    const memLength = memory.byteLength;

    // Validate input tensor bounds
    if (!validateBufferBounds(desc.maskOffset, desc.maskBytes, memLength, 4) ||
        !validateBufferBounds(desc.spatialOffset, desc.spatialBytes, memLength, 4) ||
        !validateBufferBounds(desc.globalOffset, desc.globalBytes, memLength, 4)) {
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.BUFFER_OVERFLOW);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      throw new Error('Input tensor buffer offset out of memory bounds');
    }

    // Clone from SharedArrayBuffer to ensure zero-offset standard ArrayBuffers for WebGPU queue.writeBuffer
    const maskData = new Float32Array(new Float32Array(memory, desc.maskOffset, desc.maskBytes / 4));
    const spatialData = new Float32Array(new Float32Array(memory, desc.spatialOffset, desc.spatialBytes / 4));
    const globalData = new Float32Array(new Float32Array(memory, desc.globalOffset, desc.globalBytes / 4));

    const maskTensor = new ort.Tensor('float32', maskData, [batchSize, 1, H, W]);
    const spatialTensor = new ort.Tensor('float32', spatialData, [batchSize, desc.numSpatialChannels, H, W]);
    const globalTensor = new ort.Tensor('float32', globalData, [batchSize, desc.numGlobalChannels, 1, 1]);

    const feeds: Record<string, ort.Tensor> = {
      InputMask: maskTensor,
      InputSpatial: spatialTensor,
      InputGlobal: globalTensor,
    };

    if (desc.numMetaChannels > 0 && desc.metaBytes > 0) {
      if (!validateBufferBounds(desc.metaOffset, desc.metaBytes, memLength, 4)) {
        Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.BUFFER_OVERFLOW);
        Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
        throw new Error('Meta tensor buffer offset out of memory bounds');
      }
      const metaData = new Float32Array(new Float32Array(memory, desc.metaOffset, desc.metaBytes / 4));
      feeds.InputMeta = new ort.Tensor('float32', metaData, [batchSize, desc.numMetaChannels, 1, 1]);
    }

    // Set slot status to RUNNING
    Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.RUNNING);

    const startTime = performance.now();
    let results: ort.InferenceSession.OnnxValueMapType;
    try {
      const timeout = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('ORT session.run TIMED OUT after 5000ms')), 5000)
      );
      results = await Promise.race([this.session.run(feeds), timeout]);
    } catch (err: unknown) {
      this.isBroken = true;
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.INFERENCE_FAILED);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`ORT WebGPU inference failed: ${msg}`);
    }
    this.lastExecutionTimeMs = performance.now() - startTime;
    this.totalInferences += batchSize;

    // Retrieve tensor data
    const [policyPassData, policyData, valueData, scoreValueData, ownershipData] = await Promise.all([
      (results.OutputPolicyPass as any)?.getData ? (results.OutputPolicyPass as any).getData() : Promise.resolve(results.OutputPolicyPass?.data),
      (results.OutputPolicy as any)?.getData ? (results.OutputPolicy as any).getData() : Promise.resolve(results.OutputPolicy?.data),
      (results.OutputValue as any)?.getData ? (results.OutputValue as any).getData() : Promise.resolve(results.OutputValue?.data),
      (results.OutputScoreValue as any)?.getData ? (results.OutputScoreValue as any).getData() : Promise.resolve(results.OutputScoreValue?.data),
      (results.OutputOwnership as any)?.getData ? (results.OutputOwnership as any).getData() : Promise.resolve(results.OutputOwnership?.data),
    ]);

    const policyPass = policyPassData as Float32Array | undefined;
    const policy = policyData as Float32Array | undefined;
    const value = valueData as Float32Array | undefined;
    const scoreValue = scoreValueData as Float32Array | undefined;
    const ownership = ownershipData as Float32Array | undefined;

    if (!policyPass || !policy || !value || !scoreValue || !ownership) {
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.INFERENCE_FAILED);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      throw new Error('ORT WebGPU inference returned incomplete output heads');
    }

    // Verify shapes and finiteness
    if (policyPass.length < batchSize * desc.numPolicyChannels ||
        policy.length < batchSize * desc.numPolicyChannels * H * W ||
        value.length < batchSize * desc.numValueChannels ||
        scoreValue.length < batchSize * desc.numScoreValueChannels ||
        ownership.length < batchSize * desc.numOwnershipChannels * H * W) {
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.INFERENCE_FAILED);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      throw new Error('Output tensor dimensions do not match descriptor expectations');
    }

    if (!Number.isFinite(policy[0]) || !Number.isFinite(value[0]) || !Number.isFinite(scoreValue[0])) {
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.INFERENCE_FAILED);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      throw new Error('Inference produced non-finite floating point values');
    }

    if (!validateBufferBounds(desc.policyPassOffset, desc.policyPassBytes, memLength, 4) ||
        !validateBufferBounds(desc.policyOffset, desc.policyBytes, memLength, 4) ||
        !validateBufferBounds(desc.valueOffset, desc.valueBytes, memLength, 4) ||
        !validateBufferBounds(desc.scoreValueOffset, desc.scoreValueBytes, memLength, 4) ||
        !validateBufferBounds(desc.ownershipOffset, desc.ownershipBytes, memLength, 4)) {
      Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.BUFFER_OVERFLOW);
      Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.ERROR);
      throw new Error('Output tensor buffer offset out of memory bounds');
    }

    const dstPolicyPass = new Float32Array(memory, desc.policyPassOffset, desc.policyPassBytes / 4);
    dstPolicyPass.set(policyPass.subarray(0, dstPolicyPass.length));

    const dstPolicy = new Float32Array(memory, desc.policyOffset, desc.policyBytes / 4);
    dstPolicy.set(policy.subarray(0, dstPolicy.length));

    const dstValue = new Float32Array(memory, desc.valueOffset, desc.valueBytes / 4);
    dstValue.set(value.subarray(0, dstValue.length));

    const dstScoreValue = new Float32Array(memory, desc.scoreValueOffset, desc.scoreValueBytes / 4);
    dstScoreValue.set(scoreValue.subarray(0, dstScoreValue.length));

    const dstOwnership = new Float32Array(memory, desc.ownershipOffset, desc.ownershipBytes / 4);
    dstOwnership.set(ownership.subarray(0, dstOwnership.length));

    // Dispose tensors to free GPU resources
    for (const tensor of Object.values(results)) {
      (tensor as any)?.dispose?.();
    }
    (feeds.InputMask as any)?.dispose?.();
    (feeds.InputSpatial as any)?.dispose?.();
    (feeds.InputGlobal as any)?.dispose?.();
    (feeds.InputMeta as any)?.dispose?.();

    // Transition to DONE via Atomics store
    Atomics.store(u32, DESCRIPTOR_OFFSETS.ERROR_CODE / 4, KatagoBridgeErrorCode.OK);
    Atomics.store(u32, DESCRIPTOR_OFFSETS.STATUS / 4, KatagoSlotStatus.DONE);
  }

  public getDiagnostics(): OrtDiagnostics {
    return {
      provider: 'webgpu',
      disableCpuFallback: true,
      inputNames: this.inputNames,
      outputNames: this.outputNames,
      executionTimeMs: this.lastExecutionTimeMs,
      totalInferences: this.totalInferences,
    };
  }

  public async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }
}
