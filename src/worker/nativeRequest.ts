/**
 * Explicit adapter mapping Baduk Public Contract v1 AnalyzeRequest
 * to the canonical JSON query format expected by upstream KataGo C++ analysis engine.
 */

import {
  AnalyzeRequest,
  Capabilities,
  Color,
  Dimensions,
  Move,
  Point,
  Position,
  Rules,
} from '../api/types';
import { resolveRules } from '../api/validation';

export interface NativeAllowMovesEntry {
  player: 'B' | 'W' | 'b' | 'w';
  moves: string[];
  untilDepth: number;
}

export interface NativeAnalysisQuery {
  id: string;
  boardXSize: number;
  boardYSize: number;
  komi: number;
  initialPlayer: 'b' | 'w';
  initialStones: [string, string][];
  moves: [string, string][];
  rules: string | Record<string, unknown>;
  whiteHandicapBonus?: string;
  maxVisits: number;
  maxPlayouts: number;
  maxTime?: number;
  numThreads: number;
  rootNumSymmetriesToSample: number;
  conservativePass: boolean;
  analysisPVLen: number;
  includeOwnership: boolean;
  includeOwnershipStdev: boolean;
  includePolicy: boolean;
  reportDuringSearchEvery: number;
  maxMoves?: number;
  allowMoves?: NativeAllowMovesEntry[];
  overrideSettings?: Record<string, unknown>;
}

export function pointToKataGoCoord(point: Point, boardHeight: number): string {
  if (point.x < 0 || point.y < 0) {
    throw new Error(`Invalid point coordinate: (${point.x}, ${point.y})`);
  }
  // Upstream KataGo skips 'I' (ASCII 73)
  const colCode = 65 + (point.x >= 8 ? point.x + 1 : point.x);
  const colLetter = String.fromCharCode(colCode);
  const rowNum = boardHeight - point.y;
  return `${colLetter}${rowNum}`;
}

export function moveToString(move: Move, boardHeight: number): string {
  if (move.kind === 'pass') {
    return 'pass';
  }
  return pointToKataGoCoord(move.point, boardHeight);
}

export function getNextPlayer(position: Position): Color {
  if (position.kind === 'setup') {
    return position.nextPlayer;
  }
  if (position.moves.length === 0) {
    return position.initialPlayer;
  }
  const lastMove = position.moves[position.moves.length - 1];
  return lastMove.player === 'black' ? 'white' : 'black';
}

export function mapAnalyzeRequestToNative(
  req: AnalyzeRequest,
  caps: Capabilities
): NativeAnalysisQuery {
  const pos = req.position;
  const dims: Dimensions = pos.dimensions;
  const boardXSize = dims.width;
  const boardYSize = dims.height;

  // 1. Resolve Rules
  const resolvedRules: Rules = pos.rules.kind === 'preset'
    ? resolveRules(pos.rules, caps)
    : pos.rules.value;

  let rulesValue: string | Record<string, unknown>;
  let whiteHandicapBonus = resolvedRules.whiteHandicapBonus;

  if (pos.rules.kind === 'preset') {
    rulesValue = pos.rules.name;
    if (pos.rules.name === 'chinese') {
      whiteHandicapBonus = 'N';
    } else {
      whiteHandicapBonus = '0';
    }
  } else {
    rulesValue = {
      ko: resolvedRules.ko.toUpperCase(),
      scoring: resolvedRules.scoring.toUpperCase(),
      tax: resolvedRules.tax.toUpperCase(),
      suicide: resolvedRules.multiStoneSuicideLegal,
      hasButton: resolvedRules.hasButton,
      friendlyPassOk: resolvedRules.friendlyPassOk,
      whiteHandicapBonus: resolvedRules.whiteHandicapBonus,
    };
  }

  // 2. Initial stones and moves
  let initialPlayer: 'b' | 'w';
  let initialStones: [string, string][] = [];
  let moves: [string, string][] = [];

  if (pos.kind === 'game') {
    initialPlayer = pos.initialPlayer === 'black' ? 'b' : 'w';
    initialStones = pos.initialStones.map(s => [
      s.player === 'black' ? 'B' : 'W',
      pointToKataGoCoord(s.point, boardYSize),
    ]);
    moves = pos.moves.map(m => [
      m.player === 'black' ? 'B' : 'W',
      moveToString(m.move, boardYSize),
    ]);
  } else {
    // Setup position
    initialPlayer = pos.nextPlayer === 'black' ? 'b' : 'w';
    initialStones = pos.stones.map(s => [
      s.player === 'black' ? 'B' : 'W',
      pointToKataGoCoord(s.point, boardYSize),
    ]);
    moves = [];
  }

  // 3. Search limits
  const limits = req.search.limits;
  let maxVisits = 10000000;
  let maxPlayouts = 10000000;
  let maxTime: number | undefined = undefined;

  if (limits.kind === 'visits') {
    maxVisits = limits.maxVisits;
    maxPlayouts = limits.maxVisits;
  } else if (limits.kind === 'time') {
    maxTime = limits.maxTimeMs / 1000.0;
  } else if (limits.kind === 'either') {
    maxVisits = limits.maxVisits;
    maxPlayouts = limits.maxVisits;
    maxTime = limits.maxTimeMs / 1000.0;
  }

  // 4. Concurrency & search parameters
  const numThreads = typeof req.search.searchThreads === 'number' ? req.search.searchThreads : 2;
  const rootNumSymmetriesToSample = req.search.rootSymmetrySamples;
  const conservativePass = req.search.conservativePass;

  const overrideSettings: Record<string, unknown> = {};
  if (typeof req.search.wideRootNoise === 'number' && req.search.wideRootNoise > 0) {
    overrideSettings.wideRootNoise = req.search.wideRootNoise;
  }
  if (req.search.seed !== null && req.search.seed !== undefined) {
    overrideSettings.searchRandSeed = req.search.seed;
  }

  // 5. Allowed root moves restriction
  let allowMoves: NativeAllowMovesEntry[] | undefined = undefined;
  if (req.search.allowedRootMoves && req.search.allowedRootMoves.length > 0) {
    const nextPla = getNextPlayer(pos);
    allowMoves = [
      {
        player: nextPla === 'black' ? 'B' : 'W',
        moves: req.search.allowedRootMoves.map(m => moveToString(m, boardYSize)),
        untilDepth: 1,
      },
    ];
  }

  // 6. Output options
  const analysisPVLen = req.output.pvLength;
  const includeOwnership = req.output.ownership !== 'none';
  const includeOwnershipStdev = req.output.includeOwnershipStdev;
  const includePolicy = req.output.includePolicy;
  const reportDuringSearchEvery = req.output.progressIntervalMs && req.output.progressIntervalMs > 0
    ? req.output.progressIntervalMs / 1000.0
    : 0.0;
  const maxMoves = req.output.maxCandidates;

  const query: NativeAnalysisQuery = {
    id: req.requestId,
    boardXSize,
    boardYSize,
    komi: pos.komi,
    initialPlayer,
    initialStones,
    moves,
    rules: rulesValue,
    whiteHandicapBonus,
    maxVisits,
    maxPlayouts,
    numThreads,
    rootNumSymmetriesToSample,
    conservativePass,
    analysisPVLen,
    includeOwnership,
    includeOwnershipStdev,
    includePolicy,
    reportDuringSearchEvery,
    maxMoves,
  };

  if (maxTime !== undefined) {
    query.maxTime = maxTime;
  }
  if (allowMoves !== undefined) {
    query.allowMoves = allowMoves;
  }
  if (Object.keys(overrideSettings).length > 0) {
    query.overrideSettings = overrideSettings;
  }

  return query;
}
