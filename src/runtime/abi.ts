/**
 * Authoritative TypeScript view of the Katago Web Bridge ABI.
 * Must match native/bridge/bridge_abi.h byte-for-byte.
 */

export const KATAGO_BRIDGE_ABI_VERSION = 1;
export const BRIDGE_DESCRIPTOR_BYTES = 168;

export enum KatagoSlotStatus {
  FREE = 0,
  WRITING = 1,
  READY = 2,
  RUNNING = 3,
  DONE = 4,
  ERROR = 5,
}

export enum KatagoBridgeErrorCode {
  OK = 0,
  INVALID_DESCRIPTOR = 1,
  UNSUPPORTED_VERSION = 2,
  STALE_GENERATION = 3,
  CANCELLED = 4,
  INFERENCE_FAILED = 5,
  BUFFER_OVERFLOW = 6,
  DEVICE_LOST = 7,
}

export enum KatagoTensorLayout {
  NCHW = 0,
  NHWC = 1,
}

// Field byte offsets in BridgeSlotDescriptor
export const DESCRIPTOR_OFFSETS = {
  ABI_VERSION: 0,
  ENGINE_GENERATION: 4,
  REQUEST_SEQUENCE_LOW: 8,
  REQUEST_SEQUENCE_HIGH: 12,
  SLOT_ID: 16,
  MODEL_ID: 20,
  STATUS: 24,
  CANCEL_GENERATION: 28,

  ERROR_CODE: 32,
  ERROR_MSG_OFFSET: 36,
  ERROR_MSG_CAPACITY: 40,
  RESERVED0: 44,

  BATCH_SIZE: 48,
  MAX_BATCH_SIZE: 52,
  NN_X_LEN: 56,
  NN_Y_LEN: 60,
  NUM_SPATIAL_CHANNELS: 64,
  NUM_GLOBAL_CHANNELS: 68,
  NUM_META_CHANNELS: 72,
  NUM_POLICY_CHANNELS: 76,

  NUM_VALUE_CHANNELS: 80,
  NUM_SCORE_VALUE_CHANNELS: 84,
  NUM_OWNERSHIP_CHANNELS: 88,
  SPATIAL_LAYOUT: 92,

  MASK_OFFSET: 96,
  MASK_BYTES: 100,
  SPATIAL_OFFSET: 104,
  SPATIAL_BYTES: 108,
  GLOBAL_OFFSET: 112,
  GLOBAL_BYTES: 116,
  META_OFFSET: 120,
  META_BYTES: 124,

  POLICY_PASS_OFFSET: 128,
  POLICY_PASS_BYTES: 132,
  POLICY_OFFSET: 136,
  POLICY_BYTES: 140,
  VALUE_OFFSET: 144,
  VALUE_BYTES: 148,
  SCORE_VALUE_OFFSET: 152,
  SCORE_VALUE_BYTES: 156,
  OWNERSHIP_OFFSET: 160,
  OWNERSHIP_BYTES: 164,
} as const;

export interface SlotDescriptorView {
  readonly abiVersion: number;
  readonly engineGeneration: number;
  readonly requestSequence: bigint;
  readonly slotId: number;
  readonly modelId: number;
  readonly status: KatagoSlotStatus;
  readonly cancelGeneration: number;

  readonly errorCode: KatagoBridgeErrorCode;
  readonly errorMsgOffset: number;
  readonly errorMsgCapacity: number;

  readonly batchSize: number;
  readonly maxBatchSize: number;
  readonly nnXLen: number;
  readonly nnYLen: number;
  readonly numSpatialChannels: number;
  readonly numGlobalChannels: number;
  readonly numMetaChannels: number;
  readonly numPolicyChannels: number;
  readonly numValueChannels: number;
  readonly numScoreValueChannels: number;
  readonly numOwnershipChannels: number;
  readonly spatialLayout: KatagoTensorLayout;

  readonly maskOffset: number;
  readonly maskBytes: number;
  readonly spatialOffset: number;
  readonly spatialBytes: number;
  readonly globalOffset: number;
  readonly globalBytes: number;
  readonly metaOffset: number;
  readonly metaBytes: number;

  readonly policyPassOffset: number;
  readonly policyPassBytes: number;
  readonly policyOffset: number;
  readonly policyBytes: number;
  readonly valueOffset: number;
  readonly valueBytes: number;
  readonly scoreValueOffset: number;
  readonly scoreValueBytes: number;
  readonly ownershipOffset: number;
  readonly ownershipBytes: number;
}

/**
 * Validates non-wrapping unsigned 32-bit bounds in Wasm memory.
 */
export function validateBufferBounds(
  offset: number,
  byteLength: number,
  memoryByteLength: number,
  alignment = 4
): boolean {
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(byteLength)) {
    return false;
  }
  if (offset < 0 || byteLength < 0) {
    return false;
  }
  if ((offset & (alignment - 1)) !== 0) {
    return false;
  }
  // Safe unsigned addition without bitwise 32-bit truncation
  const end = offset + byteLength;
  if (end > memoryByteLength) {
    return false;
  }
  return true;
}

/**
 * Reads and validates a slot descriptor from Wasm memory.
 */
export function readSlotDescriptor(
  memory: ArrayBufferLike,
  descriptorOffset: number
): SlotDescriptorView {
  const memLength = memory.byteLength;
  if (!validateBufferBounds(descriptorOffset, BRIDGE_DESCRIPTOR_BYTES, memLength, 4)) {
    throw new Error(`Descriptor offset ${descriptorOffset} out of memory bounds (${memLength})`);
  }

  const u32 = new Uint32Array(memory, descriptorOffset, BRIDGE_DESCRIPTOR_BYTES / 4);
  const i32 = new Int32Array(memory, descriptorOffset, BRIDGE_DESCRIPTOR_BYTES / 4);

  const abiVersion = u32[DESCRIPTOR_OFFSETS.ABI_VERSION / 4];
  if (abiVersion !== KATAGO_BRIDGE_ABI_VERSION) {
    throw new Error(`Bridge ABI version mismatch: expected ${KATAGO_BRIDGE_ABI_VERSION}, got ${abiVersion}`);
  }

  const engineGeneration = u32[DESCRIPTOR_OFFSETS.ENGINE_GENERATION / 4];
  const seqLow = BigInt(u32[DESCRIPTOR_OFFSETS.REQUEST_SEQUENCE_LOW / 4]);
  const seqHigh = BigInt(u32[DESCRIPTOR_OFFSETS.REQUEST_SEQUENCE_HIGH / 4]);
  const requestSequence = (seqHigh << 32n) | seqLow;

  const slotId = u32[DESCRIPTOR_OFFSETS.SLOT_ID / 4];
  const modelId = u32[DESCRIPTOR_OFFSETS.MODEL_ID / 4];
  const status = Atomics.load(i32, DESCRIPTOR_OFFSETS.STATUS / 4) as KatagoSlotStatus;
  const cancelGeneration = u32[DESCRIPTOR_OFFSETS.CANCEL_GENERATION / 4];

  const errorCode = u32[DESCRIPTOR_OFFSETS.ERROR_CODE / 4] as KatagoBridgeErrorCode;
  const errorMsgOffset = u32[DESCRIPTOR_OFFSETS.ERROR_MSG_OFFSET / 4];
  const errorMsgCapacity = u32[DESCRIPTOR_OFFSETS.ERROR_MSG_CAPACITY / 4];

  const batchSize = u32[DESCRIPTOR_OFFSETS.BATCH_SIZE / 4];
  const maxBatchSize = u32[DESCRIPTOR_OFFSETS.MAX_BATCH_SIZE / 4];
  const nnXLen = u32[DESCRIPTOR_OFFSETS.NN_X_LEN / 4];
  const nnYLen = u32[DESCRIPTOR_OFFSETS.NN_Y_LEN / 4];
  const numSpatialChannels = u32[DESCRIPTOR_OFFSETS.NUM_SPATIAL_CHANNELS / 4];
  const numGlobalChannels = u32[DESCRIPTOR_OFFSETS.NUM_GLOBAL_CHANNELS / 4];
  const numMetaChannels = u32[DESCRIPTOR_OFFSETS.NUM_META_CHANNELS / 4];
  const numPolicyChannels = u32[DESCRIPTOR_OFFSETS.NUM_POLICY_CHANNELS / 4];
  const numValueChannels = u32[DESCRIPTOR_OFFSETS.NUM_VALUE_CHANNELS / 4];
  const numScoreValueChannels = u32[DESCRIPTOR_OFFSETS.NUM_SCORE_VALUE_CHANNELS / 4];
  const numOwnershipChannels = u32[DESCRIPTOR_OFFSETS.NUM_OWNERSHIP_CHANNELS / 4];
  const spatialLayout = u32[DESCRIPTOR_OFFSETS.SPATIAL_LAYOUT / 4] as KatagoTensorLayout;

  const maskOffset = u32[DESCRIPTOR_OFFSETS.MASK_OFFSET / 4];
  const maskBytes = u32[DESCRIPTOR_OFFSETS.MASK_BYTES / 4];
  const spatialOffset = u32[DESCRIPTOR_OFFSETS.SPATIAL_OFFSET / 4];
  const spatialBytes = u32[DESCRIPTOR_OFFSETS.SPATIAL_BYTES / 4];
  const globalOffset = u32[DESCRIPTOR_OFFSETS.GLOBAL_OFFSET / 4];
  const globalBytes = u32[DESCRIPTOR_OFFSETS.GLOBAL_BYTES / 4];
  const metaOffset = u32[DESCRIPTOR_OFFSETS.META_OFFSET / 4];
  const metaBytes = u32[DESCRIPTOR_OFFSETS.META_BYTES / 4];

  const policyPassOffset = u32[DESCRIPTOR_OFFSETS.POLICY_PASS_OFFSET / 4];
  const policyPassBytes = u32[DESCRIPTOR_OFFSETS.POLICY_PASS_BYTES / 4];
  const policyOffset = u32[DESCRIPTOR_OFFSETS.POLICY_OFFSET / 4];
  const policyBytes = u32[DESCRIPTOR_OFFSETS.POLICY_BYTES / 4];
  const valueOffset = u32[DESCRIPTOR_OFFSETS.VALUE_OFFSET / 4];
  const valueBytes = u32[DESCRIPTOR_OFFSETS.VALUE_BYTES / 4];
  const scoreValueOffset = u32[DESCRIPTOR_OFFSETS.SCORE_VALUE_OFFSET / 4];
  const scoreValueBytes = u32[DESCRIPTOR_OFFSETS.SCORE_VALUE_BYTES / 4];
  const ownershipOffset = u32[DESCRIPTOR_OFFSETS.OWNERSHIP_OFFSET / 4];
  const ownershipBytes = u32[DESCRIPTOR_OFFSETS.OWNERSHIP_BYTES / 4];

  // Validate all offsets against memory boundaries
  const checks = [
    ['mask', maskOffset, maskBytes],
    ['spatial', spatialOffset, spatialBytes],
    ['global', globalOffset, globalBytes],
    ['policyPass', policyPassOffset, policyPassBytes],
    ['policy', policyOffset, policyBytes],
    ['value', valueOffset, valueBytes],
    ['scoreValue', scoreValueOffset, scoreValueBytes],
    ['ownership', ownershipOffset, ownershipBytes],
  ] as const;

  if (numMetaChannels > 0) {
    if (!validateBufferBounds(metaOffset, metaBytes, memLength, 4)) {
      throw new Error(`Buffer 'meta' out of bounds: offset=${metaOffset}, bytes=${metaBytes}`);
    }
  }

  for (const [name, off, len] of checks) {
    if (!validateBufferBounds(off, len, memLength, 4)) {
      throw new Error(`Buffer '${name}' out of bounds: offset=${off}, bytes=${len}, memLength=${memLength}`);
    }
  }

  return {
    abiVersion,
    engineGeneration,
    requestSequence,
    slotId,
    modelId,
    status,
    cancelGeneration,
    errorCode,
    errorMsgOffset,
    errorMsgCapacity,
    batchSize,
    maxBatchSize,
    nnXLen,
    nnYLen,
    numSpatialChannels,
    numGlobalChannels,
    numMetaChannels,
    numPolicyChannels,
    numValueChannels,
    numScoreValueChannels,
    numOwnershipChannels,
    spatialLayout,
    maskOffset,
    maskBytes,
    spatialOffset,
    spatialBytes,
    globalOffset,
    globalBytes,
    metaOffset,
    metaBytes,
    policyPassOffset,
    policyPassBytes,
    policyOffset,
    policyBytes,
    valueOffset,
    valueBytes,
    scoreValueOffset,
    scoreValueBytes,
    ownershipOffset,
    ownershipBytes,
  };
}

/**
 * Transitions slot state using Atomics with release ordering.
 */
export function setSlotStatus(
  memory: ArrayBufferLike,
  descriptorOffset: number,
  expectedStatus: KatagoSlotStatus,
  newStatus: KatagoSlotStatus
): boolean {
  const i32 = new Int32Array(memory, descriptorOffset + DESCRIPTOR_OFFSETS.STATUS, 1);
  const actual = Atomics.compareExchange(i32, 0, expectedStatus, newStatus);
  return actual === expectedStatus;
}
