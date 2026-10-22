import { describe, it, expect } from 'vitest';
import {
  mapAnalyzeRequestToNative,
  pointToKataGoCoord,
  moveToString,
  getNextPlayer,
} from '../src/worker/nativeRequest';
import { AnalyzeRequest, Capabilities } from '../src/api/types';

const MOCK_CAPS: Capabilities = {
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
  maxVisits: 1000000,
  maxTimeMs: 3600000,
  maxBatchSize: 16,
  maxSearchThreads: 16,
  maxPvLength: 100,
};

describe('nativeRequest mapping (Stage R1 Qualification)', () => {
  it('correctly maps Go coordinates skipping column I', () => {
    // A1 = (0, 18) on 19x19
    expect(pointToKataGoCoord({ x: 0, y: 18 }, 19)).toBe('A1');
    // H1 = (7, 18)
    expect(pointToKataGoCoord({ x: 7, y: 18 }, 19)).toBe('H1');
    // J1 = (8, 18) -> skips I!
    expect(pointToKataGoCoord({ x: 8, y: 18 }, 19)).toBe('J1');
    // Q16 = (15, 3) -> col 15 + 1 = 16 = 'Q', row = 19 - 3 = 16
    expect(pointToKataGoCoord({ x: 15, y: 3 }, 19)).toBe('Q16');
    // T19 = (18, 0)
    expect(pointToKataGoCoord({ x: 18, y: 0 }, 19)).toBe('T19');
  });

  it('correctly formats pass moves', () => {
    expect(moveToString({ kind: 'pass' }, 19)).toBe('pass');
    expect(moveToString({ kind: 'play', point: { x: 3, y: 3 } }, 19)).toBe('D16');
  });

  it('computes next player correctly from game move history', () => {
    expect(getNextPlayer({
      kind: 'game',
      dimensions: { width: 19, height: 19 },
      initialStones: [],
      initialPlayer: 'black',
      moves: [],
      rules: { kind: 'preset', name: 'japanese' },
      komi: 6.5,
      handicap: { countOverride: null, inferFromInitialBlackMoves: false },
    })).toBe('black');

    expect(getNextPlayer({
      kind: 'game',
      dimensions: { width: 19, height: 19 },
      initialStones: [],
      initialPlayer: 'black',
      moves: [{ player: 'black', move: { kind: 'play', point: { x: 15, y: 3 } } }],
      rules: { kind: 'preset', name: 'japanese' },
      komi: 6.5,
      handicap: { countOverride: null, inferFromInitialBlackMoves: false },
    })).toBe('white');
  });

  it('maps complete contract v1 AnalyzeRequest to native KataGo query', () => {
    const req: AnalyzeRequest = {
      apiVersion: 1,
      requestId: 'test-req-42',
      position: {
        kind: 'game',
        dimensions: { width: 19, height: 19 },
        initialStones: [{ player: 'black', point: { x: 3, y: 3 } }],
        initialPlayer: 'black',
        moves: [
          { player: 'white', move: { kind: 'play', point: { x: 15, y: 15 } } },
          { player: 'black', move: { kind: 'pass' } },
        ],
        rules: { kind: 'preset', name: 'chinese' },
        komi: 7.5,
        handicap: { countOverride: null, inferFromInitialBlackMoves: false },
      },
      search: {
        profile: 'upstream-analysis-v1',
        limits: { kind: 'visits', maxVisits: 500 },
        seed: 'test-seed-123',
        batchSize: 4,
        searchThreads: 2,
        conservativePass: true,
        wideRootNoise: 0.04,
        rootSymmetrySamples: 8,
        allowedRootMoves: [
          { kind: 'play', point: { x: 15, y: 3 } },
          { kind: 'pass' },
        ],
      },
      output: {
        maxCandidates: 5,
        pvLength: 15,
        includePolicy: true,
        ownership: 'root',
        includeOwnershipStdev: false,
        includeCandidateOwnership: false,
        progressIntervalMs: 50,
      },
    };

    const nativeQuery = mapAnalyzeRequestToNative(req, MOCK_CAPS);

    expect(nativeQuery.id).toBe('test-req-42');
    expect(nativeQuery.boardXSize).toBe(19);
    expect(nativeQuery.boardYSize).toBe(19);
    expect(nativeQuery.komi).toBe(7.5);
    expect(nativeQuery.rules).toBe('chinese');
    expect(nativeQuery.whiteHandicapBonus).toBe('N');
    expect(nativeQuery.maxVisits).toBe(500);
    expect(nativeQuery.maxPlayouts).toBe(500);
    expect(nativeQuery.numThreads).toBe(2);
    expect(nativeQuery.rootNumSymmetriesToSample).toBe(8);
    expect(nativeQuery.conservativePass).toBe(true);
    expect(nativeQuery.analysisPVLen).toBe(15);
    expect(nativeQuery.includeOwnership).toBe(true);
    expect(nativeQuery.includePolicy).toBe(true);
    expect(nativeQuery.reportDuringSearchEvery).toBe(0.05);

    // Initial stones & moves
    expect(nativeQuery.initialStones).toEqual([['B', 'D16']]);
    expect(nativeQuery.moves).toEqual([
      ['W', 'Q4'],
      ['B', 'pass'],
    ]);

    // Allowed root moves (Black played last move, so next is White)
    expect(nativeQuery.allowMoves).toBeDefined();
    expect(nativeQuery.allowMoves![0].player).toBe('W');
    expect(nativeQuery.allowMoves![0].moves).toEqual(['Q16', 'pass']);
    expect(nativeQuery.allowMoves![0].untilDepth).toBe(1);

    // Override settings
    expect(nativeQuery.overrideSettings).toEqual({
      wideRootNoise: 0.04,
      searchRandSeed: 'test-seed-123',
    });
  });

  it('maps setup positions correctly without replaying moves', () => {
    const req: AnalyzeRequest = {
      apiVersion: 1,
      requestId: 'setup-req-1',
      position: {
        kind: 'setup',
        dimensions: { width: 13, height: 13 },
        stones: [
          { player: 'black', point: { x: 2, y: 2 } },
          { player: 'white', point: { x: 10, y: 10 } },
        ],
        nextPlayer: 'white',
        rules: {
          kind: 'explicit',
          value: {
            ko: 'positional',
            scoring: 'area',
            tax: 'none',
            multiStoneSuicideLegal: true,
            hasButton: false,
            whiteHandicapBonus: '0',
            friendlyPassOk: true,
          },
        },
        komi: 5.5,
        simpleKoPoint: null,
        acknowledgeMissingHistory: true,
      },
      search: {
        profile: 'upstream-analysis-v1',
        limits: { kind: 'time', maxTimeMs: 2500 },
        seed: null,
        batchSize: 2,
        searchThreads: 1,
        conservativePass: false,
        wideRootNoise: 0,
        rootSymmetrySamples: 1,
        allowedRootMoves: null,
      },
      output: {
        maxCandidates: 3,
        pvLength: 8,
        includePolicy: false,
        ownership: 'none',
        includeOwnershipStdev: false,
        includeCandidateOwnership: false,
        progressIntervalMs: null,
      },
    };

    const nativeQuery = mapAnalyzeRequestToNative(req, MOCK_CAPS);
    expect(nativeQuery.boardXSize).toBe(13);
    expect(nativeQuery.boardYSize).toBe(13);
    expect(nativeQuery.initialPlayer).toBe('w');
    expect(nativeQuery.initialStones).toEqual([
      ['B', 'C11'],
      ['W', 'L3'],
    ]);
    expect(nativeQuery.moves).toEqual([]);
    expect(nativeQuery.maxTime).toBe(2.5);
    expect(nativeQuery.rules).toEqual({
      ko: 'POSITIONAL',
      scoring: 'AREA',
      tax: 'NONE',
      suicide: true,
      hasButton: false,
      friendlyPassOk: true,
      whiteHandicapBonus: '0',
    });
    expect(nativeQuery.includeOwnership).toBe(false);
    expect(nativeQuery.includePolicy).toBe(false);
    expect(nativeQuery.allowMoves).toBeUndefined();
  });
});
