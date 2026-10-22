import { describe, it, expect } from 'vitest';
import {
  validateIdentifier,
  validateDimensions,
  validatePoint,
  validateMove,
  validateKomi,
  resolveRules,
  validateAnalyzeRequest,
} from '../src/api/validation';
import { Capabilities, AnalyzeRequest } from '../src/api/types';
import { BASELINE_B10C384_MANIFEST, validateManifest, computeModelFingerprint } from '../src/models/manifest';

const mockCaps: Capabilities = {
  apiVersion: 1,
  packageVersion: '1.0.0',
  upstreamVersion: '1.18.1',
  upstreamCommit: '92ee95c0a4b25fec214da00951ab69e97e207729',
  status: 'available',
  unavailableReasons: [],
  supportedAccelerators: ['webgpu'],
  supportedModelFormats: ['onnx'],
  supportedRulePresets: ['japanese', 'korean', 'chinese'],
  maxBoardWidth: 19,
  maxBoardHeight: 19,
  supportsRectangularBoards: true,
  supportsSetupPositions: true,
  supportsRootMoveRestriction: true,
  supportsCandidateOwnership: false,
  supportsOwnershipStdev: false,
  crossOriginIsolationRequired: true,
  maxVisits: 100000,
  maxTimeMs: 60000,
  maxBatchSize: 16,
  maxSearchThreads: 16,
  maxPvLength: 50,
};

describe('Contract v1 Validation', () => {
  it('validates identifiers strictly', () => {
    expect(validateIdentifier('req-1', 'requestId')).toBe('req-1');
    expect(() => validateIdentifier('', 'requestId')).toThrow();
    expect(() => validateIdentifier('   ', 'requestId')).toThrow();
    expect(() => validateIdentifier('a'.repeat(129), 'requestId')).toThrow();
  });

  it('validates board dimensions and bounds', () => {
    expect(validateDimensions({ width: 19, height: 19 }, mockCaps)).toEqual({ width: 19, height: 19 });
    expect(validateDimensions({ width: 9, height: 9 }, mockCaps)).toEqual({ width: 9, height: 9 });
    expect(validateDimensions({ width: 13, height: 19 }, mockCaps)).toEqual({ width: 13, height: 19 });

    expect(() => validateDimensions({ width: 1, height: 19 }, mockCaps)).toThrow();
    expect(() => validateDimensions({ width: 25, height: 19 }, mockCaps)).toThrow();
    expect(() => validateDimensions({ width: 19.5, height: 19 }, mockCaps)).toThrow();
  });

  it('validates point coordinates with non-negative checks and -0 rejection', () => {
    const dims = { width: 19, height: 19 };
    expect(validatePoint({ x: 0, y: 0 }, dims)).toEqual({ x: 0, y: 0 });
    expect(validatePoint({ x: 18, y: 18 }, dims)).toEqual({ x: 18, y: 18 });

    expect(() => validatePoint({ x: -1, y: 5 }, dims)).toThrow();
    expect(() => validatePoint({ x: 19, y: 5 }, dims)).toThrow();
    expect(() => validatePoint({ x: -0, y: 5 }, dims)).toThrow();
  });

  it('validates moves including pass and coordinate play', () => {
    const dims = { width: 19, height: 19 };
    expect(validateMove({ kind: 'pass' }, dims)).toEqual({ kind: 'pass' });
    expect(validateMove({ kind: 'play', point: { x: 3, y: 3 } }, dims)).toEqual({
      kind: 'play',
      point: { x: 3, y: 3 },
    });
    expect(() => validateMove({ kind: 'unknown' }, dims)).toThrow();
  });

  it('validates komi values', () => {
    expect(validateKomi(7.5)).toBe(7.5);
    expect(validateKomi(0)).toBe(0);
    expect(validateKomi(-5.5)).toBe(-5.5);
    expect(() => validateKomi(7.25)).toThrow(); // not integer or half-integer
    expect(() => validateKomi(NaN)).toThrow();
    expect(() => validateKomi(Infinity)).toThrow();
  });

  it('resolves rule presets faithfully', () => {
    const jpn = resolveRules({ kind: 'preset', name: 'japanese' }, mockCaps);
    expect(jpn.scoring).toBe('territory');
    expect(jpn.ko).toBe('simple');

    const chn = resolveRules({ kind: 'preset', name: 'chinese' }, mockCaps);
    expect(chn.scoring).toBe('area');

    expect(() => resolveRules({ kind: 'preset', name: 'unknown' as any }, mockCaps)).toThrow();
  });

  it('validates analyze requests and prevents duplicate IDs', () => {
    const activeIds = new Set<string>();

    const validReq: AnalyzeRequest = {
      apiVersion: 1,
      requestId: 'test-req-1',
      position: {
        kind: 'game',
        dimensions: { width: 19, height: 19 },
        initialStones: [],
        initialPlayer: 'black',
        moves: [],
        rules: { kind: 'preset', name: 'chinese' },
        komi: 7.5,
        handicap: { countOverride: null, inferFromInitialBlackMoves: false },
      },
      search: {
        profile: 'upstream-analysis-v1',
        limits: { kind: 'visits', maxVisits: 100 },
        seed: 'test-seed',
        batchSize: 4,
        searchThreads: 2,
        conservativePass: false,
        wideRootNoise: 0.04,
        rootSymmetrySamples: 1,
        allowedRootMoves: null,
      },
      output: {
        maxCandidates: 5,
        pvLength: 10,
        includePolicy: true,
        ownership: 'root',
        includeOwnershipStdev: false,
        includeCandidateOwnership: false,
        progressIntervalMs: 100,
      },
    };

    const validated = validateAnalyzeRequest(validReq, mockCaps, activeIds);
    expect(validated.requestId).toBe('test-req-1');

    activeIds.add('test-req-1');
    expect(() => validateAnalyzeRequest(validReq, mockCaps, activeIds)).toThrow(/already been used/);
  });
});

describe('Model Manifest Validation', () => {
  it('validates baseline model manifest structure', () => {
    const validated = validateManifest(BASELINE_B10C384_MANIFEST);
    expect(validated.modelId).toBe('b10c384-default');
    expect(validated.variants.length).toBeGreaterThan(0);
    expect(validated.variants[0].artifacts[0].path).toBe('model.onnx');
  });

  it('rejects path traversal in artifact paths', () => {
    const badManifest = {
      schemaVersion: 1,
      modelId: 'bad-model',
      sourceModelSha256: '00'.repeat(32),
      variants: [
        {
          variantId: 'bad-variant',
          format: 'onnx',
          entryPath: '../model.onnx',
          artifacts: [
            {
              path: '../model.onnx',
              sha256: '00'.repeat(32),
              byteLength: 100,
              source: { kind: 'url', url: 'http://example.com' },
            },
          ],
        },
      ],
    };

    expect(() => validateManifest(badManifest)).toThrow(/Illegal artifact path/);
  });

  it('computes model fingerprint deterministically', () => {
    const fp1 = computeModelFingerprint(BASELINE_B10C384_MANIFEST, 'onnx-webgpu-default');
    const fp2 = computeModelFingerprint(BASELINE_B10C384_MANIFEST, 'onnx-webgpu-default');
    expect(fp1).toBe(fp2);
    expect(fp1).toContain('b10c384-default');
  });
});
