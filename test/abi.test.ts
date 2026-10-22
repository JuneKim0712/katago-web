import { describe, it, expect } from 'vitest';
import {
  KATAGO_BRIDGE_ABI_VERSION,
  BRIDGE_DESCRIPTOR_BYTES,
  DESCRIPTOR_OFFSETS,
  KatagoSlotStatus,
  KatagoBridgeErrorCode,
  KatagoTensorLayout,
  validateBufferBounds,
  readSlotDescriptor,
  setSlotStatus,
} from '../src/runtime/abi.js';

describe('Bridge ABI & Memory Ownership', () => {
  it('has exact byte length and expected field offsets', () => {
    expect(BRIDGE_DESCRIPTOR_BYTES).toBe(168);
    expect(DESCRIPTOR_OFFSETS.ABI_VERSION).toBe(0);
    expect(DESCRIPTOR_OFFSETS.ENGINE_GENERATION).toBe(4);
    expect(DESCRIPTOR_OFFSETS.STATUS).toBe(24);
    expect(DESCRIPTOR_OFFSETS.BATCH_SIZE).toBe(48);
    expect(DESCRIPTOR_OFFSETS.SPATIAL_LAYOUT).toBe(92);
    expect(DESCRIPTOR_OFFSETS.MASK_OFFSET).toBe(96);
    expect(DESCRIPTOR_OFFSETS.OWNERSHIP_BYTES).toBe(164);
  });

  it('validates buffer bounds without signed wraparound', () => {
    const memSize = 1024 * 1024; // 1MB

    expect(validateBufferBounds(0, 100, memSize, 4)).toBe(true);
    expect(validateBufferBounds(memSize - 100, 100, memSize, 4)).toBe(true);

    // Negative offset or length
    expect(validateBufferBounds(-4, 100, memSize, 4)).toBe(false);
    expect(validateBufferBounds(100, -4, memSize, 4)).toBe(false);

    // Unaligned offset
    expect(validateBufferBounds(3, 100, memSize, 4)).toBe(false);

    // Out of memory bounds
    expect(validateBufferBounds(memSize - 50, 100, memSize, 4)).toBe(false);

    // Potential 32-bit overflow (e.g. 0xFFFFFFF0 + 0x20)
    expect(validateBufferBounds(0xfffffff0, 0x20, memSize, 4)).toBe(false);
  });

  it('correctly reads valid slot descriptor from memory', () => {
    const mem = new SharedArrayBuffer(4096);
    const u32 = new Uint32Array(mem);

    // Fill valid descriptor
    u32[DESCRIPTOR_OFFSETS.ABI_VERSION / 4] = KATAGO_BRIDGE_ABI_VERSION;
    u32[DESCRIPTOR_OFFSETS.ENGINE_GENERATION / 4] = 42;
    u32[DESCRIPTOR_OFFSETS.REQUEST_SEQUENCE_LOW / 4] = 100;
    u32[DESCRIPTOR_OFFSETS.REQUEST_SEQUENCE_HIGH / 4] = 0;
    u32[DESCRIPTOR_OFFSETS.SLOT_ID / 4] = 0;
    u32[DESCRIPTOR_OFFSETS.STATUS / 4] = KatagoSlotStatus.READY;

    u32[DESCRIPTOR_OFFSETS.BATCH_SIZE / 4] = 1;
    u32[DESCRIPTOR_OFFSETS.MAX_BATCH_SIZE / 4] = 1;
    u32[DESCRIPTOR_OFFSETS.NN_X_LEN / 4] = 19;
    u32[DESCRIPTOR_OFFSETS.NN_Y_LEN / 4] = 19;
    u32[DESCRIPTOR_OFFSETS.NUM_SPATIAL_CHANNELS / 4] = 22;
    u32[DESCRIPTOR_OFFSETS.NUM_GLOBAL_CHANNELS / 4] = 19;
    u32[DESCRIPTOR_OFFSETS.NUM_POLICY_CHANNELS / 4] = 2;
    u32[DESCRIPTOR_OFFSETS.NUM_VALUE_CHANNELS / 4] = 3;
    u32[DESCRIPTOR_OFFSETS.NUM_SCORE_VALUE_CHANNELS / 4] = 6;
    u32[DESCRIPTOR_OFFSETS.NUM_OWNERSHIP_CHANNELS / 4] = 1;

    // Buffer offsets and lengths within 1024 bytes
    const base = 200;
    u32[DESCRIPTOR_OFFSETS.MASK_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.MASK_BYTES / 4] = 19 * 19 * 4;
    u32[DESCRIPTOR_OFFSETS.SPATIAL_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.SPATIAL_BYTES / 4] = 100;
    u32[DESCRIPTOR_OFFSETS.GLOBAL_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.GLOBAL_BYTES / 4] = 76;
    u32[DESCRIPTOR_OFFSETS.POLICY_PASS_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.POLICY_PASS_BYTES / 4] = 8;
    u32[DESCRIPTOR_OFFSETS.POLICY_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.POLICY_BYTES / 4] = 100;
    u32[DESCRIPTOR_OFFSETS.VALUE_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.VALUE_BYTES / 4] = 12;
    u32[DESCRIPTOR_OFFSETS.SCORE_VALUE_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.SCORE_VALUE_BYTES / 4] = 24;
    u32[DESCRIPTOR_OFFSETS.OWNERSHIP_OFFSET / 4] = base;
    u32[DESCRIPTOR_OFFSETS.OWNERSHIP_BYTES / 4] = 100;

    const desc = readSlotDescriptor(mem, 0);
    expect(desc.abiVersion).toBe(1);
    expect(desc.engineGeneration).toBe(42);
    expect(desc.requestSequence).toBe(100n);
    expect(desc.status).toBe(KatagoSlotStatus.READY);
    expect(desc.nnXLen).toBe(19);
    expect(desc.numSpatialChannels).toBe(22);
  });

  it('rejects version mismatch and out-of-bounds buffers', () => {
    const mem = new SharedArrayBuffer(1024);
    const u32 = new Uint32Array(mem);

    // Bad version
    u32[DESCRIPTOR_OFFSETS.ABI_VERSION / 4] = 999;
    expect(() => readSlotDescriptor(mem, 0)).toThrow(/version mismatch/);

    // Valid version, but buffer out of bounds
    u32[DESCRIPTOR_OFFSETS.ABI_VERSION / 4] = KATAGO_BRIDGE_ABI_VERSION;
    u32[DESCRIPTOR_OFFSETS.MASK_OFFSET / 4] = 1020;
    u32[DESCRIPTOR_OFFSETS.MASK_BYTES / 4] = 100; // 1020 + 100 > 1024
    expect(() => readSlotDescriptor(mem, 0)).toThrow(/out of bounds/);
  });

  it('transitions slot status atomically', () => {
    const mem = new SharedArrayBuffer(1024);
    const u32 = new Uint32Array(mem);
    u32[DESCRIPTOR_OFFSETS.STATUS / 4] = KatagoSlotStatus.READY;

    // Transition from READY to RUNNING
    const success = setSlotStatus(mem, 0, KatagoSlotStatus.READY, KatagoSlotStatus.RUNNING);
    expect(success).toBe(true);

    // Stale transition (expected READY, but is now RUNNING) should fail
    const staleSuccess = setSlotStatus(mem, 0, KatagoSlotStatus.READY, KatagoSlotStatus.DONE);
    expect(staleSuccess).toBe(false);

    // Transition from RUNNING to DONE
    const doneSuccess = setSlotStatus(mem, 0, KatagoSlotStatus.RUNNING, KatagoSlotStatus.DONE);
    expect(doneSuccess).toBe(true);
  });
});
