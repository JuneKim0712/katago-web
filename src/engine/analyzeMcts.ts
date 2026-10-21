import * as tf from '@tensorflow/tfjs';
import type { BoardState, FloatArray, GameRules, Move, Player, RegionOfInterest } from '../types/index.js';
import { getAnimationNow } from '../utils/helpers.js';
import { type RandomSource, DefaultRandomSource, Rand } from '../utils/random.js';
import { type ClockSource, DefaultClockSource } from '../utils/clock.js';
import { postprocessKataGoV8 } from './evalV8.js';
import { processNeuralOutputs, EvaluationContext } from './evaluation/batch.js';
import { FeatureCache } from './features/cache.js';
import type { EvalState, NeuralEval, EvalBatchScratch } from './evaluation/types.js';
export type { NeuralEval } from './evaluation/types.js';
import type { KataGoModelV8Tf } from '../model/modelV8.js';
import { expectedWhiteScoreValue, getSqrtBoardArea } from './scoreValue.js';
import { ENGINE_MAX_TIME_MS, ENGINE_MAX_VISITS } from './limits.js';
import { Node, type Edge } from './search/tree.js';
import {
  computeRecentScoreCenter,
  computeBlackUtilityFromEval,
  computeWeightedRootStats,
  type ChildWeightStats,
  type RootSelfStats,
} from './search/utility.js';
import { selectEdge } from './search/selection.js';
import { expandNode, getExpandScratch } from './search/expansion.js';
import {
  moveToGtp,
  buildPv,
  getPvForEdge,
  averageTreeOwnership,
} from './search/report.js';

export { moveToGtp };
import {
  BLACK,
  WHITE,
  EMPTY,
  EMPTY_AREA_MAP,
  BOARD_AREA,
  BOARD_SIZE,
  PASS_MOVE,
  NEIGHBOR_COUNTS,
  NEIGHBOR_LIST,
  NEIGHBOR_STARTS,
  opponentOf,
  playMove,
  playMoveInto,
  undoMove,
  computeLadderFeaturesV7KataGoInto,
  computeLadderedStonesV7KataGoInto,
  computeAreaMapV7KataGoInto,
  computeLibertyMap,
  computeLibertyMapInto,
  updateLibertyMapForSeeds,
  type SimPosition,
  type StoneColor,
  type UndoSnapshot,
} from './fastBoard.js';
import { fillInputsV7Fast, type RecentMove } from './featuresV7Fast.js';
import { POLICY_OPTIMISM, ROOT_POLICY_OPTIMISM } from './searchParams.js';
import { getNNKomi } from './territoryKomi.js';
import { resolveKoState } from '../utils/ko.js';
import { getVisitSpeedFlags } from './visitSpeedConfig.js';

const EMPTY_OWNERSHIP = new Float32Array(0);
const EMPTY_POLICY = new Float32Array(0);

export type OwnershipMode = 'none' | 'root' | 'tree';

type PolicyValueOutput = ReturnType<KataGoModelV8Tf['forwardPolicyValue']>;
type PolicyValueOwnershipOutput = ReturnType<KataGoModelV8Tf['forward']>;

const hasOwnership = (out: PolicyValueOutput | PolicyValueOwnershipOutput): out is PolicyValueOwnershipOutput => {
  return 'ownership' in out;
};


function playerToColor(p: Player): StoneColor {
  return p === 'black' ? BLACK : WHITE;
}

function colorToPlayer(c: StoneColor): Player {
  return c === BLACK ? 'black' : 'white';
}

function boardStateToStones(board: BoardState): Uint8Array<ArrayBuffer> {
  const stones = new Uint8Array(BOARD_AREA);
  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      const v = board[y]?.[x] ?? null;
      if (!v) continue;
      stones[y * BOARD_SIZE + x] = v === 'black' ? BLACK : WHITE;
    }
  }
  return stones;
}

function takeRecentMoves(
  rootMoves: RecentMove[],
  pathMoves: RecentMove[],
  max: number,
  out: RecentMove[] = [],
  pathMovesCount?: number
): RecentMove[] {
  const pCount = pathMovesCount ?? pathMoves.length;
  const total = pCount + rootMoves.length;
  const count = total < max ? total : max;
  out.length = count;

  let outIdx = 0;
  const rootCount = count > pCount ? count - pCount : 0;
  const rootStart = rootMoves.length - rootCount;
  for (let i = rootStart; i < rootMoves.length; i++) {
    const src = rootMoves[i]!;
    let dst = out[outIdx];
    if (!dst) out[outIdx] = { move: src.move, player: src.player };
    else { dst.move = src.move; dst.player = src.player; }
    outIdx++;
  }

  const pathCount = count - rootCount;
  const pathStart = pCount - pathCount;
  for (let i = pathStart; i < pCount; i++) {
    const src = pathMoves[i]!;
    let dst = out[outIdx];
    if (!dst) out[outIdx] = { move: src.move, player: src.player };
    else { dst.move = src.move; dst.player = src.player; }
    outIdx++;
  }

  return out;
}

function normalizeRegionOfInterest(roi?: RegionOfInterest | null): RegionOfInterest | null {
  if (!roi) return null;
  const xMin = Math.max(0, Math.min(BOARD_SIZE - 1, Math.min(roi.xMin, roi.xMax)));
  const xMax = Math.max(0, Math.min(BOARD_SIZE - 1, Math.max(roi.xMin, roi.xMax)));
  const yMin = Math.max(0, Math.min(BOARD_SIZE - 1, Math.min(roi.yMin, roi.yMax)));
  const yMax = Math.max(0, Math.min(BOARD_SIZE - 1, Math.max(roi.yMin, roi.yMax)));
  const isSinglePoint = xMin === xMax && yMin === yMax;
  const isWholeBoard = xMin === 0 && yMin === 0 && xMax === BOARD_SIZE - 1 && yMax === BOARD_SIZE - 1;
  if (isSinglePoint || isWholeBoard) return null;
  return { xMin, xMax, yMin, yMax };
}

function buildAllowedMovesMask(roi?: RegionOfInterest | null): Uint8Array | null {
  const normalized = normalizeRegionOfInterest(roi);
  if (!normalized) return null;
  const allowed = new Uint8Array(BOARD_AREA);
  for (let y = normalized.yMin; y <= normalized.yMax; y++) {
    const rowOff = y * BOARD_SIZE;
    for (let x = normalized.xMin; x <= normalized.xMax; x++) {
      allowed[rowOff + x] = 1;
    }
  }
  return allowed;
}


async function buildRootEval(args: {
  model: KataGoModelV8Tf;
  ownershipMode: OwnershipMode;
  rules: GameRules;
  nnRandomize: boolean;
  rootSymmetrySamples?: number;
  komi: number;
  currentPlayer: Player;
  conservativePass: boolean;
  rootStones: Uint8Array;
  rootKoPoint: number;
  rootPrevStones: Uint8Array;
  rootPrevKoPoint: number;
  rootPrevPrevStones: Uint8Array;
  rootPrevPrevKoPoint: number;
  rootMoves: RecentMove[];
  maxChildren: number;
  regionOfInterest?: RegionOfInterest | null;
  outputScaleMultiplier: number;
  node?: Node;
}): Promise<{
  rootLibertyMap: Uint8Array;
  rootOwnership: Float32Array;
  rootPolicy: Float32Array;
  rootValue: number;
  rootScoreLead: number;
  rootScoreMean: number;
  rootScoreMeanSq: number;
  rootUtility: number;
  recentScoreCenter: number;
}> {
  const includeOwnership = args.ownershipMode !== 'none';
  const rootEval = await evaluateRootEval({
    model: args.model,
    includeOwnership,
    rules: args.rules,
    nnRandomize: args.nnRandomize,
    rootSymmetrySamples: args.rootSymmetrySamples,
    policyOptimism: ROOT_POLICY_OPTIMISM,
    komi: getNNKomi(args.komi, args.rules, args.rootMoves, args.rootMoves.length === 0 ? args.rootStones : undefined),
    state: {
      stones: args.rootStones,
      koPoint: args.rootKoPoint,
      prevStones: args.rootPrevStones,
      prevKoPoint: args.rootPrevKoPoint,
      prevPrevStones: args.rootPrevPrevStones,
      prevPrevKoPoint: args.rootPrevPrevKoPoint,
      currentPlayer: args.currentPlayer,
      recentMoves: takeRecentMoves(args.rootMoves, [], 5),
      conservativePassAndIsRoot: args.conservativePass,
    },
  });

  const rootLibertyMap = new Uint8Array(rootEval.libertyMap);
  const rootOwnership = new Float32Array(BOARD_AREA);
  if (includeOwnership) {
    if (!rootEval.ownership) throw new Error('Missing ownership output');
    const rootOwnershipSign = args.currentPlayer === 'black' ? 1 : -1;
    const rootSym = rootEval.symmetry;
    const rootSymOff = rootSym * BOARD_AREA;
    const symPosMap = rootSym === 0 ? null : getSymPosMap();
    for (let i = 0; i < BOARD_AREA; i++) {
      const symPos = rootSym === 0 ? i : symPosMap![rootSymOff + i]!;
      rootOwnership[i] = rootOwnershipSign * Math.tanh(rootEval.ownership[symPos]! * args.outputScaleMultiplier);
    }
  }

  const rootAllowedMoves = buildAllowedMovesMask(args.regionOfInterest);
  const rootPolicy = new Float32Array(BOARD_AREA + 1);
  const policyNode = args.node ?? new Node(playerToColor(args.currentPlayer));
  expandNode({
    node: policyNode,
    stones: args.rootStones,
    koPoint: args.rootKoPoint,
    policyLogits: rootEval.policy,
    policyLogitsSymmetry: rootEval.symmetry,
    passLogit: rootEval.passLogit,
    maxChildren: args.maxChildren,
    libertyMap: rootEval.libertyMap,
    allowedMoves: rootAllowedMoves ?? undefined,
    policyOut: rootPolicy,
    policyOutputScaling: args.outputScaleMultiplier,
  });

  const recentScoreCenter = computeRecentScoreCenter(-rootEval.blackScoreMean);
  const rootValue = 2 * rootEval.blackWinProb - 1;
  const rootUtility = computeBlackUtilityFromEval({
    blackWinProb: rootEval.blackWinProb,
    blackNoResultProb: rootEval.blackNoResultProb,
    blackScoreMean: rootEval.blackScoreMean,
    blackScoreStdev: rootEval.blackScoreStdev,
    recentScoreCenter,
  });
  const rootScoreMeanSq = rootEval.blackScoreStdev * rootEval.blackScoreStdev + rootEval.blackScoreMean * rootEval.blackScoreMean;

  return {
    rootLibertyMap,
    rootOwnership,
    rootPolicy,
    rootValue,
    rootScoreLead: rootEval.blackScoreLead,
    rootScoreMean: rootEval.blackScoreMean,
    rootScoreMeanSq,
    rootUtility,
    recentScoreCenter,
  };
}

// Mirrors KataGo config "Internal params" defaults (see cpp/configs/*_example.cfg).

function hasLadderCandidates(libertyMap: Uint8Array): boolean {
  for (let i = 0; i < libertyMap.length; i++) {
    const v = libertyMap[i]!;
    if (v === 1 || v === 2) return true;
  }
  return false;
}

function buildLibertySeeds(args: {
  move: number;
  captureStack: number[];
  captureStart: number;
  out: Int16Array;
}): number {
  let count = 0;
  const push = (pos: number) => {
    if (count < args.out.length) args.out[count++] = pos;
  };
  const pushWithNeighbors = (pos: number) => {
    push(pos);
    const nStart = NEIGHBOR_STARTS[pos]!;
    const nCount = NEIGHBOR_COUNTS[pos]!;
    for (let i = 0; i < nCount; i++) push(NEIGHBOR_LIST[nStart + i]!);
  };

  if (args.move !== PASS_MOVE) pushWithNeighbors(args.move);
  for (let i = args.captureStart; i < args.captureStack.length; i++) {
    pushWithNeighbors(args.captureStack[i]!);
  }
  return count;
}


const NUM_SYMMETRIES = 8;
let symPosMapBoardArea = 0;
let SYM_POS_MAP: Int16Array<ArrayBufferLike> = new Int16Array(0);

const buildSymPosMap = (): Int16Array<ArrayBufferLike> => {
  const n = BOARD_SIZE;
  const map = new Int16Array(NUM_SYMMETRIES * BOARD_AREA);
  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const symOff = sym * BOARD_AREA;
    const mirror = sym >= 4;
    const rot = sym & 3;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const sx = mirror ? n - 1 - x : x;
        const sy = y;
        let tx: number;
        let ty: number;
        if (rot === 0) {
          tx = sx;
          ty = sy;
        } else if (rot === 1) {
          tx = sy;
          ty = n - 1 - sx;
        } else if (rot === 2) {
          tx = n - 1 - sx;
          ty = n - 1 - sy;
        } else {
          tx = n - 1 - sy;
          ty = sx;
        }
        map[symOff + y * n + x] = ty * n + tx;
      }
    }
  }
  return map;
};

const getSymPosMap = (): Int16Array<ArrayBufferLike> => {
  const expectedSize = NUM_SYMMETRIES * BOARD_AREA;
  if (symPosMapBoardArea !== BOARD_AREA || SYM_POS_MAP.length !== expectedSize) {
    SYM_POS_MAP = buildSymPosMap();
    symPosMapBoardArea = BOARD_AREA;
  }
  return SYM_POS_MAP;
};

function clampRootSymmetrySamples(samples?: number): number {
  if (typeof samples !== 'number' || !Number.isFinite(samples)) return 1;
  return Math.max(1, Math.min(NUM_SYMMETRIES, Math.floor(samples)));
}

function averageRootEvals(evals: NeuralEval[]): NeuralEval {
  const first = evals[0];
  if (!first) throw new Error('No root evaluations to average');
  if (evals.length === 1) return first;

  const inv = 1.0 / evals.length;
  const symPosMap = getSymPosMap();
  const policyProbSums = new Float64Array(BOARD_AREA);
  const ownershipSums = first.ownership ? new Float64Array(BOARD_AREA) : null;
  let passProbSum = 0;
  let blackWinProb = 0;
  let blackScoreLead = 0;
  let blackScoreMean = 0;
  let blackScoreMeanSq = 0;
  let blackNoResultProb = 0;

  for (const ev of evals) {
    const sym = ev.symmetry;
    const symOff = sym * BOARD_AREA;
    let maxLogit = ev.passLogit;
    for (let p = 0; p < BOARD_AREA; p++) {
      const logit = ev.policy[p]!;
      if (logit > maxLogit) maxLogit = logit;
    }

    let probSum = Math.exp(ev.passLogit - maxLogit);
    for (let p = 0; p < BOARD_AREA; p++) probSum += Math.exp(ev.policy[p]! - maxLogit);
    const probScale = inv / probSum;
    passProbSum += Math.exp(ev.passLogit - maxLogit) * probScale;

    for (let p = 0; p < BOARD_AREA; p++) {
      const symPos = sym === 0 ? p : symPosMap[symOff + p]!;
      policyProbSums[p] += Math.exp(ev.policy[symPos]! - maxLogit) * probScale;
      if (ownershipSums) {
        if (!ev.ownership) throw new Error('Missing ownership output');
        ownershipSums[p] += ev.ownership[symPos]! * inv;
      }
    }

    blackWinProb += ev.blackWinProb * inv;
    blackScoreLead += ev.blackScoreLead * inv;
    blackScoreMean += ev.blackScoreMean * inv;
    blackScoreMeanSq += (ev.blackScoreStdev * ev.blackScoreStdev + ev.blackScoreMean * ev.blackScoreMean) * inv;
    blackNoResultProb += ev.blackNoResultProb * inv;
  }

  const minPolicyProb = 1e-30;
  const policy = new Float32Array(BOARD_AREA);
  for (let p = 0; p < BOARD_AREA; p++) policy[p] = Math.log(Math.max(minPolicyProb, policyProbSums[p]!));

  let ownership: Float32Array | undefined;
  if (ownershipSums) {
    ownership = new Float32Array(BOARD_AREA);
    for (let p = 0; p < BOARD_AREA; p++) ownership[p] = ownershipSums[p]!;
  }

  return {
    policy,
    symmetry: 0,
    passLogit: Math.log(Math.max(minPolicyProb, passProbSum)),
    blackWinProb,
    blackScoreLead,
    blackScoreMean,
    blackScoreStdev: Math.sqrt(Math.max(0, blackScoreMeanSq - blackScoreMean * blackScoreMean)),
    blackNoResultProb,
    libertyMap: new Uint8Array(first.libertyMap),
    areaMap: new Uint8Array(first.areaMap),
    ownership,
  };
}

async function evaluateRootEval(args: {
  model: KataGoModelV8Tf;
  includeOwnership?: boolean;
  rules: GameRules;
  nnRandomize: boolean;
  rootSymmetrySamples?: number;
  policyOptimism: number;
  komi: number;
  state: EvalState;
}): Promise<NeuralEval> {
  const rootSymmetrySamples = clampRootSymmetrySamples(args.rootSymmetrySamples);
  if (rootSymmetrySamples <= 1) {
    return (
      await evaluateBatch({
        model: args.model,
        includeOwnership: args.includeOwnership,
        rules: args.rules,
        nnRandomize: args.nnRandomize,
        policyOptimism: args.policyOptimism,
        komi: args.komi,
        states: [args.state],
      })
    )[0]!;
  }

  const states = new Array<EvalState>(rootSymmetrySamples);
  for (let symmetry = 0; symmetry < rootSymmetrySamples; symmetry++) {
    states[symmetry] = { ...args.state, symmetry };
  }

  return averageRootEvals(
    await evaluateBatch({
      model: args.model,
      includeOwnership: args.includeOwnership,
      rules: args.rules,
      nnRandomize: false,
      policyOptimism: args.policyOptimism,
      komi: args.komi,
      states,
    })
  );
}


// Two ping-pong slots so one can be in-flight to the GPU while the other is
// being filled by the CPU for the next batch (§ 1.1 double-buffered dispatch).
const defaultEvalContext = new EvaluationContext();

function getEvalScratch(args: { batch: number; includeAreaFeature: boolean; slot?: number; evalContext?: EvaluationContext }): EvalBatchScratch {
  return (args.evalContext ?? defaultEvalContext).getScratch(args);
}

function prepareBatchFeatures(args: {
  states: EvalState[];
  rules: GameRules;
  komi: number;
  nnRandomize: boolean;
  scratch: EvalBatchScratch;
  randomSource?: RandomSource;
  featureCache?: FeatureCache;
}): { batch: number; spatialBatch: Float32Array; globalBatch: Float32Array } {
  const { states, rules, komi, nnRandomize, scratch, featureCache } = args;
  const randSource = args.randomSource ?? DefaultRandomSource;
  const includeAreaFeature = rules === 'chinese';
  const batch = states.length;
  const spatialBatch = scratch.spatialBatch.subarray(0, batch * BOARD_AREA * 22);
  const globalBatch = scratch.globalBatch.subarray(0, batch * 19);
  const libertyMapScratch = scratch.libertyMapScratch.subarray(0, batch * BOARD_AREA);
  const areaMapScratch = includeAreaFeature ? scratch.areaMapScratch!.subarray(0, batch * BOARD_AREA) : null;
  const symmetries = scratch.symmetries.subarray(0, batch);

  for (let i = 0; i < batch; i++) {
    const state = states[i]!;
    const libertyMap = libertyMapScratch.subarray(i * BOARD_AREA, (i + 1) * BOARD_AREA);
    if (state.libertyMap) libertyMap.set(state.libertyMap);
    else computeLibertyMapInto(state.stones, libertyMap);

    let areaMap: Uint8Array<ArrayBufferLike> = EMPTY_AREA_MAP;
    if (includeAreaFeature) {
      const dstArea = areaMapScratch!.subarray(i * BOARD_AREA, (i + 1) * BOARD_AREA);
      if (!featureCache || !featureCache.getAreaMap(state.stones, dstArea)) {
        computeAreaMapV7KataGoInto(state.stones, dstArea);
        featureCache?.putAreaMap(state.stones, dstArea);
      }
      areaMap = dstArea;
    }

    const plaColor = playerToColor(state.currentPlayer);
    if (hasLadderCandidates(libertyMap)) {
      if (!featureCache || !featureCache.getLadderFeatures(state.stones, state.koPoint, plaColor, scratch.ladderedStonesScratch, scratch.ladderWorkingMovesScratch)) {
        computeLadderFeaturesV7KataGoInto({
          stones: state.stones,
          koPoint: state.koPoint,
          currentPlayer: plaColor,
          outLadderedStones: scratch.ladderedStonesScratch,
          outLadderWorkingMoves: scratch.ladderWorkingMovesScratch,
        });
        featureCache?.putLadderFeatures(state.stones, state.koPoint, plaColor, scratch.ladderedStonesScratch, scratch.ladderWorkingMovesScratch);
      }
    } else {
      scratch.ladderedStonesScratch.fill(0);
      scratch.ladderWorkingMovesScratch.fill(0);
    }

    const recentMoves = state.recentMoves;
    const lastRecentMove = recentMoves.length > 0 ? recentMoves[recentMoves.length - 1] : null;
    const passWouldEndGame = lastRecentMove?.move === PASS_MOVE;
    const suppressHistory = state.conservativePassAndIsRoot === true && passWouldEndGame;

    const pla = state.currentPlayer;
    const opp = pla === 'black' ? 'white' : 'black';

    let numTurnsOfHistoryIncluded = 0;
    if (!suppressHistory) {
      for (let h = 0; h < 5; h++) {
        const m = recentMoves[recentMoves.length - 1 - h];
        if (!m) break;
        const expectedPlayer = (h & 1) === 0 ? opp : pla;
        if (m.player !== expectedPlayer) break;
        numTurnsOfHistoryIncluded++;
      }
    }

    const prevLadderStones = numTurnsOfHistoryIncluded < 1 ? state.stones : state.prevStones;
    const prevLadderKoPoint = numTurnsOfHistoryIncluded < 1 ? state.koPoint : state.prevKoPoint;
    const prevPrevLadderStones = numTurnsOfHistoryIncluded < 2 ? prevLadderStones : state.prevPrevStones;
    const prevPrevLadderKoPoint = numTurnsOfHistoryIncluded < 2 ? prevLadderKoPoint : state.prevPrevKoPoint;

    const prevLibertyMap = prevLadderStones === state.stones ? libertyMap : state.prevLibertyMap;
    if (prevLibertyMap && !hasLadderCandidates(prevLibertyMap)) {
      scratch.prevLadderedStonesScratch.fill(0);
    } else {
      if (!featureCache || !featureCache.getLadderFeatures(prevLadderStones, prevLadderKoPoint, plaColor, scratch.prevLadderedStonesScratch)) {
        computeLadderedStonesV7KataGoInto({
          stones: prevLadderStones,
          koPoint: prevLadderKoPoint,
          outLadderedStones: scratch.prevLadderedStonesScratch,
        });
        featureCache?.putLadderFeatures(prevLadderStones, prevLadderKoPoint, plaColor, scratch.prevLadderedStonesScratch);
      }
    }
    const prevPrevLibertyMap =
      prevPrevLadderStones === prevLadderStones ? prevLibertyMap : state.prevPrevLibertyMap;
    if (prevPrevLibertyMap && !hasLadderCandidates(prevPrevLibertyMap)) {
      scratch.prevPrevLadderedStonesScratch.fill(0);
    } else {
      if (!featureCache || !featureCache.getLadderFeatures(prevPrevLadderStones, prevPrevLadderKoPoint, plaColor, scratch.prevPrevLadderedStonesScratch)) {
        computeLadderedStonesV7KataGoInto({
          stones: prevPrevLadderStones,
          koPoint: prevPrevLadderKoPoint,
          outLadderedStones: scratch.prevPrevLadderedStonesScratch,
        });
        featureCache?.putLadderFeatures(prevPrevLadderStones, prevPrevLadderKoPoint, plaColor, scratch.prevPrevLadderedStonesScratch);
      }
    }

    const requestedSymmetry = state.symmetry;
    const sym =
      typeof requestedSymmetry === 'number' && Number.isFinite(requestedSymmetry)
        ? Math.max(0, Math.min(NUM_SYMMETRIES - 1, Math.floor(requestedSymmetry)))
        : nnRandomize
          ? ((randSource.next() * NUM_SYMMETRIES) | 0)
          : 0;
    symmetries[i] = sym;
    const spatialOffset = i * BOARD_AREA * 22;
    const outSpatial = spatialBatch.subarray(spatialOffset, spatialOffset + BOARD_AREA * 22);
    const outGlobal = globalBatch.subarray(i * 19, (i + 1) * 19);
    const symMap = sym === 0 ? null : getSymPosMap().subarray(sym * BOARD_AREA, (sym + 1) * BOARD_AREA);

    fillInputsV7Fast({
      stones: state.stones,
      koPoint: state.koPoint,
      currentPlayer: state.currentPlayer,
      recentMoves,
      komi: state.komi ?? komi,
      rules,
      conservativePassAndIsRoot: state.conservativePassAndIsRoot,
      libertyMap,
      areaMap: includeAreaFeature ? areaMap : undefined,
      ladderedStones: scratch.ladderedStonesScratch,
      ladderWorkingMoves: scratch.ladderWorkingMovesScratch,
      prevLadderedStones: scratch.prevLadderedStonesScratch,
      prevPrevLadderedStones: scratch.prevPrevLadderedStonesScratch,
      symmetryMap: symMap,
      outSpatial,
      outGlobal,
    });
  }

  return { batch, spatialBatch, globalBatch };
}

function createInputTensorsSync(
  model: KataGoModelV8Tf,
  spatialBatch: Float32Array,
  globalBatch: Float32Array,
  batch: number
): { spatial: tf.Tensor4D; global: tf.Tensor2D; disposeInputs: boolean } {
  return {
    spatial: tf.tensor4d(spatialBatch, [batch, BOARD_SIZE, BOARD_SIZE, 22]),
    global: tf.tensor2d(globalBatch, [batch, 19]),
    disposeInputs: true,
  };
}

async function evaluateBatch(args: {
  model: KataGoModelV8Tf;
  includeOwnership?: boolean;
  rules: GameRules;
  nnRandomize: boolean;
  policyOptimism: number;
  komi: number;
  states: EvalState[];
  randomSource?: RandomSource;
  evalContext?: EvaluationContext;
}): Promise<NeuralEval[]> {
  const { model, states, randomSource, evalContext } = args;
  const includeOwnership = args.includeOwnership === true;
  const rules = args.rules;
  const nnRandomize = args.nnRandomize;
  const policyOptimism = Math.max(0, Math.min(args.policyOptimism, 1));
  const includeAreaFeature = rules === 'chinese';
  const batch = states.length;
  const scratch = getEvalScratch({ batch, includeAreaFeature, evalContext });
  const { spatialBatch, globalBatch } = prepareBatchFeatures({
    states,
    rules,
    komi: args.komi,
    nnRandomize,
    scratch,
    randomSource,
  });
  const libertyMapScratch = scratch.libertyMapScratch.subarray(0, batch * BOARD_AREA);
  const areaMapScratch = includeAreaFeature ? scratch.areaMapScratch!.subarray(0, batch * BOARD_AREA) : null;
  const symmetries = scratch.symmetries.subarray(0, batch);

  const { spatial: spatialTensor, global: globalTensor, disposeInputs } = createInputTensorsSync(
    model,
    spatialBatch,
    globalBatch,
    batch
  );
  const out = includeOwnership ? model.forward(spatialTensor, globalTensor) : model.forwardPolicyValue(spatialTensor, globalTensor);

  const ownershipPromise = includeOwnership && hasOwnership(out) ? out.ownership.data() : Promise.resolve(null);
  const [policyArr, passArr, valueArr, scoreArr, ownershipArr] = await Promise.all([
    out.policy.data(),
    out.policyPass.data(),
    out.value.data(),
    out.scoreValue.data(),
    ownershipPromise,
  ]);

  if (disposeInputs) {
    spatialTensor.dispose();
    globalTensor.dispose();
  }
  out.policy.dispose();
  out.policyPass.dispose();
  out.value.dispose();
  out.scoreValue.dispose();
  if (hasOwnership(out)) out.ownership.dispose();

  return processNeuralOutputs({
    model,
    states,
    batch,
    includeOwnership,
    policyOptimism,
    symmetries,
    libertyMapScratch,
    areaMapScratch,
    includeAreaFeature,
    scratch,
    policyArr,
    passArr,
    valueArr,
    scoreArr,
    ownershipArr,
  });
}

/**
 * § 1.1  Pipelined (double-buffered) GPU dispatch.
 *
 * Like `evaluateBatch` but returns a Promise *before* the GPU finishes.
 * The caller may perform synchronous CPU work (tree selection / next-batch
 * feature prep) after this function returns and before `await`-ing the
 * returned promise, hiding most of the GPU latency.
 *
 * Inputs and maps belong to this dispatch. TF.js may retain typed-array
 * backing storage, so another batch must never overwrite those inputs.
 */
function dispatchEvalBatchAsync(args: {
  model: KataGoModelV8Tf;
  includeOwnership?: boolean;
  rules: GameRules;
  nnRandomize: boolean;
  policyOptimism: number;
  komi: number;
  states: EvalState[];
  slot?: number;
  randomSource?: RandomSource;
  evalContext?: EvaluationContext;
  featureCache?: FeatureCache;
}): Promise<NeuralEval[]> {
  const { model, states, randomSource, evalContext, featureCache } = args;
  const slot = args.slot ?? 0;
  const includeOwnership = args.includeOwnership === true;
  const rules = args.rules;
  const nnRandomize = args.nnRandomize;
  const policyOptimism = Math.max(0, Math.min(args.policyOptimism, 1));
  const includeAreaFeature = rules === 'chinese';
  const batch = states.length;
  const scratch = getEvalScratch({ batch, includeAreaFeature, slot, evalContext });
  const { spatialBatch, globalBatch } = prepareBatchFeatures({
    states,
    rules,
    komi: args.komi,
    nnRandomize,
    scratch,
    randomSource,
    featureCache,
  });
  const libertyMapScratch = scratch.libertyMapScratch.subarray(0, batch * BOARD_AREA);
  const areaMapScratch = includeAreaFeature ? scratch.areaMapScratch!.subarray(0, batch * BOARD_AREA) : null;
  const symmetries = scratch.symmetries.subarray(0, batch);

  const { spatial: spatialTensor, global: globalTensor, disposeInputs } = createInputTensorsSync(
    model,
    spatialBatch,
    globalBatch,
    batch
  );
  let out: ReturnType<KataGoModelV8Tf['forwardPolicyValue']> | ReturnType<KataGoModelV8Tf['forward']> | undefined;
  try {
    out = includeOwnership
      ? model.forward(spatialTensor, globalTensor)
      : model.forwardPolicyValue(spatialTensor, globalTensor);
  } catch (err) {
    if (disposeInputs) {
      spatialTensor.dispose();
      globalTensor.dispose();
    }
    throw err;
  }

  const readOutputs = async (): Promise<NeuralEval[]> => {
    let values: [
      Float32Array | Int32Array | Uint8Array,
      Float32Array | Int32Array | Uint8Array,
      Float32Array | Int32Array | Uint8Array,
      Float32Array | Int32Array | Uint8Array,
      Float32Array | Int32Array | Uint8Array | null,
    ];
    try {
      values = await Promise.all([
        out!.policy.data(),
        out!.policyPass.data(),
        out!.value.data(),
        out!.scoreValue.data(),
        includeOwnership && hasOwnership(out!) ? out!.ownership.data() : Promise.resolve(null),
      ]);
    } finally {
      if (out) tf.dispose(out);
      if (disposeInputs) {
        spatialTensor.dispose();
        globalTensor.dispose();
      }
    }
    const [policyArr, passArr, valueArr, scoreArr, ownershipArr] = values;

    return processNeuralOutputs({
      model,
      states,
      batch,
      includeOwnership,
      policyOptimism,
      symmetries,
      libertyMapScratch,
      areaMapScratch,
      includeAreaFeature,
      scratch,
      policyArr,
      passArr,
      valueArr,
      scoreArr,
      ownershipArr,
    });
  };

  return readOutputs();
}

export class MctsSearch {
  private readonly nnKomi: number;
  readonly model: KataGoModelV8Tf;
  readonly ownershipMode: OwnershipMode;
  readonly maxChildren: number;
  private currentPlayer: Player;
  readonly komi: number;
  readonly rules: GameRules;
  readonly nnRandomize: boolean;
  readonly conservativePass: boolean;
  readonly wideRootNoise: number;
  readonly rootSymmetrySamples: number;
  private readonly outputScaleMultiplier: number;

  private rootStones: Uint8Array<ArrayBuffer>;
  private rootKoPoint: number;
  private rootPrevStones: Uint8Array<ArrayBuffer>;
  private rootPrevKoPoint: number;
  private rootMoves: RecentMove[];
  private rootLibertyMap: Uint8Array;
  private rootPrevLibertyMap: Uint8Array;

  private rootNode: Node;
  private rootPolicy: Float32Array; // len 362
  private rootOwnership: Float32Array; // len 361

  /** Current visit count at the root — cheap O(1) read for logging. */
  private recentScoreCenter: number;
  private readonly rand: Rand;
  private readonly randomSource: RandomSource;
  private readonly clockSource: ClockSource;
  private readonly evalContext = new EvaluationContext();
  private readonly featureCache = new FeatureCache();
  private isRunning = false;
  private rootSelfValue: number;
  private rootSelfScoreLead: number;
  private rootSelfScoreMean: number;
  private rootSelfScoreMeanSq: number;
  private rootSelfUtility: number;

  private jobStonesScratch = new Uint8Array(0);
  private jobPrevStonesScratch = new Uint8Array(0);
  private jobPrevPrevStonesScratch = new Uint8Array(0);
  private jobLibertyMapScratch = new Uint8Array(0);
  private jobPrevLibertyMapScratch = new Uint8Array(0);
  private jobPrevPrevLibertyMapScratch = new Uint8Array(0);
  private jobRecentMovesScratch: RecentMove[][] = [];
  private jobPathStorage: Node[][] = [];
  private libertyMapStack: Uint8Array[] = [];
  private libertySeedsScratch = new Int16Array(BOARD_AREA * 5);
  private treeOwnershipCache: { visits: number; ownership: Float32Array; ownershipStdev: Float32Array; timestamp: number } | null = null;
  private readonly simScratch: SimPosition = { stones: new Uint8Array(BOARD_AREA), koPoint: -1 };
  private readonly captureStackScratch: number[] = [];
  private readonly undoMovesScratch: number[] = [];
  private readonly undoPlayersScratch: StoneColor[] = [];
  private readonly undoSnapshotsScratch: UndoSnapshot[] = [];
  private readonly pathMovesScratch: RecentMove[] = [];
  private readonly searchPathNodesScratch: Node[] = [];

  private constructor(args: {
    model: KataGoModelV8Tf;
    ownershipMode: OwnershipMode;
    maxChildren: number;
    currentPlayer: Player;
    komi: number;
    rules: GameRules;
    nnRandomize: boolean;
    conservativePass: boolean;
    wideRootNoise: number;
    rootSymmetrySamples: number;
    rootStones: Uint8Array<ArrayBuffer>;
    rootKoPoint: number;
    rootPrevStones: Uint8Array<ArrayBuffer>;
    rootPrevKoPoint: number;
    rootMoves: RecentMove[];
    rootNode: Node;
    rootLibertyMap: Uint8Array;
    rootPrevLibertyMap: Uint8Array;
    rootPolicy: Float32Array;
    rootOwnership: Float32Array;
    recentScoreCenter: number;
    rand: Rand;
    randomSource?: RandomSource;
    clockSource?: ClockSource;
    outputScaleMultiplier: number;
    rootSelfValue: number;
    rootSelfScoreLead: number;
    rootSelfScoreMean: number;
    rootSelfScoreMeanSq: number;
    rootSelfUtility: number;
  }) {
    this.model = args.model;
    this.ownershipMode = args.ownershipMode;
    this.maxChildren = args.maxChildren;
    this.currentPlayer = args.currentPlayer;
    this.komi = args.komi;
    this.nnKomi = getNNKomi(args.komi, args.rules, args.rootMoves, args.rootMoves.length === 0 ? args.rootStones : undefined);
    this.rules = args.rules;
    this.nnRandomize = args.nnRandomize;
    this.conservativePass = args.conservativePass;
    this.wideRootNoise = args.wideRootNoise;
    this.rootSymmetrySamples = args.rootSymmetrySamples;

    this.rootStones = args.rootStones;
    this.rootKoPoint = args.rootKoPoint;
    this.rootPrevStones = args.rootPrevStones;
    this.rootPrevKoPoint = args.rootPrevKoPoint;
    this.rootMoves = args.rootMoves;

    this.rootNode = args.rootNode;
    this.rootLibertyMap = args.rootLibertyMap;
    this.rootPrevLibertyMap = args.rootPrevLibertyMap;
    this.rootPolicy = args.rootPolicy;
    this.rootOwnership = args.rootOwnership;
    this.recentScoreCenter = args.recentScoreCenter;
    this.rand = args.rand;
    this.randomSource = args.randomSource ?? DefaultRandomSource;
    this.clockSource = args.clockSource ?? DefaultClockSource;
    this.outputScaleMultiplier = args.outputScaleMultiplier;
    this.rootSelfValue = args.rootSelfValue;
    this.rootSelfScoreLead = args.rootSelfScoreLead;
    this.rootSelfScoreMean = args.rootSelfScoreMean;
    this.rootSelfScoreMeanSq = args.rootSelfScoreMeanSq;
    this.rootSelfUtility = args.rootSelfUtility;

    // Pre-allocate liberty maps up to max tree depth so traversals never allocate
    while (this.libertyMapStack.length < 64) {
      this.libertyMapStack.push(new Uint8Array(BOARD_AREA));
    }
  }

  static async create(args: {
    model: KataGoModelV8Tf;
    board: BoardState;
    previousBoard?: BoardState;
    previousPreviousBoard?: BoardState;
    previousPreviousPreviousBoard?: BoardState;
    koPoint?: number;
    currentPlayer: Player;
    moveHistory: Move[];
    komi: number;
    rules: GameRules;
    nnRandomize: boolean;
    conservativePass: boolean;
    maxChildren: number;
    ownershipMode: OwnershipMode;
    wideRootNoise: number;
    rootSymmetrySamples?: number;
    regionOfInterest?: RegionOfInterest | null;
    randomSource?: RandomSource;
    clockSource?: ClockSource;
  }): Promise<MctsSearch> {
    const outputScaleMultiplier = args.model.postProcessParams?.outputScaleMultiplier ?? 1.0;
    const rootSymmetrySamples = clampRootSymmetrySamples(args.rootSymmetrySamples);
    const rootStones = boardStateToStones(args.board);
    const { rootKoPoint, rootPrevKoPoint, rootPrevPrevKoPoint } = resolveKoState({
      koPoint: args.koPoint,
      previousBoard: args.previousBoard,
      previousPreviousBoard: args.previousPreviousBoard,
      previousPreviousPreviousBoard: args.previousPreviousPreviousBoard,
      moveHistory: args.moveHistory,
    });

    const rootPrevStones = args.previousBoard ? boardStateToStones(args.previousBoard) : rootStones;
    const rootPrevPrevStones = args.previousPreviousBoard ? boardStateToStones(args.previousPreviousBoard) : rootPrevStones;

    const rootMoves: RecentMove[] = args.moveHistory.map((m) => ({
      move: m.x < 0 || m.y < 0 ? PASS_MOVE : m.y * BOARD_SIZE + m.x,
      player: m.player,
    }));

    const rootNode = new Node(playerToColor(args.currentPlayer));
    const {
      rootLibertyMap,
      rootOwnership,
      rootPolicy,
      rootValue,
      rootScoreLead,
      rootScoreMean,
      rootScoreMeanSq,
      rootUtility,
      recentScoreCenter,
    } = await buildRootEval({
      model: args.model,
      ownershipMode: args.ownershipMode,
      rules: args.rules,
      nnRandomize: args.nnRandomize,
      rootSymmetrySamples,
      komi: args.komi,
      currentPlayer: args.currentPlayer,
      conservativePass: args.conservativePass,
      rootStones,
      rootKoPoint,
      rootPrevStones,
      rootPrevKoPoint,
      rootPrevPrevStones,
      rootPrevPrevKoPoint,
      rootMoves,
      maxChildren: args.maxChildren,
      regionOfInterest: args.regionOfInterest,
      outputScaleMultiplier,
      node: rootNode,
    });
    rootNode.ownership = rootOwnership;
    rootNode.visits = 1;
    rootNode.valueSum = rootValue;
    rootNode.scoreLeadSum = rootScoreLead;
    rootNode.scoreMeanSum = rootScoreMean;
    rootNode.scoreMeanSqSum = rootScoreMeanSq;
    rootNode.utilitySum = rootUtility;
    rootNode.utilitySqSum = rootUtility * rootUtility;
    rootNode.nnUtility = rootUtility;

    const rootPrevLibertyMap =
      rootPrevStones === rootStones ? rootLibertyMap : computeLibertyMapInto(rootPrevStones, new Uint8Array(BOARD_AREA));

    return new MctsSearch({
      model: args.model,
      ownershipMode: args.ownershipMode,
      maxChildren: args.maxChildren,
      currentPlayer: args.currentPlayer,
      komi: args.komi,
      rules: args.rules,
      nnRandomize: args.nnRandomize,
      conservativePass: args.conservativePass,
      wideRootNoise: args.wideRootNoise,
      rootSymmetrySamples,
      rootStones,
      rootKoPoint,
      rootPrevStones,
      rootPrevKoPoint,
      rootMoves,
      rootNode,
      rootLibertyMap,
      rootPrevLibertyMap,
      rootPolicy,
      rootOwnership,
      recentScoreCenter,
      rand: new Rand(args.randomSource ?? DefaultRandomSource),
      randomSource: args.randomSource,
      clockSource: args.clockSource,
      outputScaleMultiplier,
      rootSelfValue: rootValue,
      rootSelfScoreLead: rootScoreLead,
      rootSelfScoreMean: rootScoreMean,
      rootSelfScoreMeanSq: rootScoreMeanSq,
      rootSelfUtility: rootUtility,
    });
  }

  async run(args: {
    visits: number;
    maxTimeMs: number;
    batchSize: number;
    shouldAbort?: () => boolean;
    reportEveryMs?: number;
    onProgress?: () => void;
  }): Promise<boolean> {
    const maxVisits = Math.max(16, Math.min(args.visits, ENGINE_MAX_VISITS));
    const maxTimeMs = Math.max(25, Math.min(args.maxTimeMs, ENGINE_MAX_TIME_MS));
    const batchSize = Math.max(1, Math.min(args.batchSize, 64));
    const shouldAbort = args.shouldAbort;

    if (this.isRunning) {
      throw new Error('Concurrent run() calls on the same SearchSession are not permitted.');
    }
    this.isRunning = true;
    try {
      if (shouldAbort?.()) return true;
      if (this.rootNode.visits >= maxVisits) return shouldAbort?.() ?? false;

    // § 1.1 Double-buffered: 2× capacity so both ping-pong slots can coexist.
    // slotBase 0 uses indices [0, batchSize), slotBase 1 uses [batchSize, 2*batchSize).
    const neededBoardCapacity = 2 * batchSize * BOARD_AREA;
    if (this.jobStonesScratch.length < neededBoardCapacity) this.jobStonesScratch = new Uint8Array(neededBoardCapacity);
    if (this.jobPrevStonesScratch.length < neededBoardCapacity) this.jobPrevStonesScratch = new Uint8Array(neededBoardCapacity);
    if (this.jobPrevPrevStonesScratch.length < neededBoardCapacity) this.jobPrevPrevStonesScratch = new Uint8Array(neededBoardCapacity);
    if (this.jobLibertyMapScratch.length < neededBoardCapacity) this.jobLibertyMapScratch = new Uint8Array(neededBoardCapacity);
    if (this.jobPrevLibertyMapScratch.length < neededBoardCapacity) this.jobPrevLibertyMapScratch = new Uint8Array(neededBoardCapacity);
    if (this.jobPrevPrevLibertyMapScratch.length < neededBoardCapacity)
      this.jobPrevPrevLibertyMapScratch = new Uint8Array(neededBoardCapacity);

    const sim: SimPosition = this.simScratch;
    sim.koPoint = this.rootKoPoint;
    const captureStack = this.captureStackScratch;
    const undoMoves = this.undoMovesScratch;
    const undoPlayers = this.undoPlayersScratch;
    const undoSnapshots = this.undoSnapshotsScratch;
    const pathMoves = this.pathMovesScratch;
    const libertyMapStack = this.libertyMapStack;
    libertyMapStack[0] = this.rootLibertyMap;
    const libertySeedsScratch = this.libertySeedsScratch;

    const clock = this.clockSource;
    const deadline = clock() + maxTimeMs;
    let timeCheckCounter = 0;
    const timeCheckMask = 0x1f;
    const timeExceeded = (): boolean => {
      if ((timeCheckCounter++ & timeCheckMask) !== 0) return false;
      return clock() >= deadline;
    };
    let lastProgressAt = 0;
    let lastProgressVisits = -1;
    const maybeReport = (): void => {
      if (!args.onProgress || !args.reportEveryMs) return;
      const visits = this.rootNode.visits;
      if (visits <= lastProgressVisits) return;
      const now = clock();
      if (now - lastProgressAt < args.reportEveryMs) return;
      lastProgressAt = now;
      lastProgressVisits = visits;
      args.onProgress();
    };
    const batchAttemptMultiplier = getVisitSpeedFlags().widerBatchCollection ? 16 : 8;

    type MctsJob = {
      komi: number;
      leaf: Node;
      path: Node[];
      stones: Uint8Array;
      koPoint: number;
      libertyMap: Uint8Array;
      prevStones: Uint8Array;
      prevKoPoint: number;
      prevLibertyMap?: Uint8Array;
      prevPrevStones: Uint8Array;
      prevPrevKoPoint: number;
      prevPrevLibertyMap?: Uint8Array;
      currentPlayer: Player;
      recentMoves: RecentMove[];
    };

    // Collect up to batchSize leaf nodes. slotBase picks which ping-pong slot of
    // jobStonesScratch / jobLibertyMapScratch etc. to write into so the two
    // concurrent batches don't alias.
    const collectBatch = (slotBase: number): MctsJob[] => {
      const jobList: MctsJob[] = [];
      let attempts = 0;
      while (jobList.length < batchSize && this.rootNode.visits + this.rootNode.inFlight < maxVisits && !timeExceeded()) {
        if (shouldAbort?.()) break;
        attempts++;
        if (attempts > batchSize * batchAttemptMultiplier) break;

        captureStack.length = 0;
        sim.stones.set(this.rootStones);
        sim.koPoint = this.rootKoPoint;
        libertyMapStack[0] = this.rootLibertyMap;
        let depth = 0;
        let pathMovesCount = 0;

        const path = this.searchPathNodesScratch;
        path.length = 1;
        path[0] = this.rootNode;
        let node = this.rootNode;
        let player = this.rootNode.playerToMove;

        while (node.edges && node.edges.length > 0) {
          const e = selectEdge(node, node === this.rootNode, this.wideRootNoise, this.rand);
          const move = e.move;

          playMoveInto(sim, move, player, captureStack, undoSnapshots, depth);
          undoMoves[depth] = move;
          undoPlayers[depth] = player;
          const prevLibertyMap = libertyMapStack[depth] ?? this.rootLibertyMap;
          let nextLibertyMap = libertyMapStack[depth + 1];
          if (!nextLibertyMap) nextLibertyMap = new Uint8Array(BOARD_AREA);
          nextLibertyMap.set(prevLibertyMap);
          const seedCount = buildLibertySeeds({
            move,
            captureStack,
            captureStart: undoSnapshots[depth]!.captureStart,
            out: libertySeedsScratch,
          });
          if (seedCount > 0) {
            updateLibertyMapForSeeds(sim.stones, libertySeedsScratch, seedCount, nextLibertyMap);
          }
          libertyMapStack[depth + 1] = nextLibertyMap;
          depth++;
          const pathPlayer = colorToPlayer(player);
          let pathEntry = pathMoves[pathMovesCount];
          if (!pathEntry) {
            pathEntry = { move, player: pathPlayer };
            pathMoves[pathMovesCount] = pathEntry;
          } else {
            pathEntry.move = move;
            pathEntry.player = pathPlayer;
          }
          pathMovesCount++;

          if (!e.child) e.child = new Node(opponentOf(player));
          node = e.child;
          player = node.playerToMove;
          path.push(node);

          if (!node.edges) break;
        }

        if (node.pendingEval) {
          continue;
        }

        node.pendingEval = true;
        for (const n of path) n.inFlight++;

        const jobIdx = jobList.length;
        // Absolute scratch index: offset by slotBase so two concurrent batches
        // use non-overlapping regions of the scratch buffers.
        const absIdx = slotBase * batchSize + jobIdx;
        const leafStones = this.jobStonesScratch.subarray(absIdx * BOARD_AREA, (absIdx + 1) * BOARD_AREA);
        leafStones.set(sim.stones);
        const leafKoPoint = sim.koPoint;
        let prevStones = leafStones;
        let prevKoPoint = leafKoPoint;
        let prevPrevStones = leafStones;
        let prevPrevKoPoint = leafKoPoint;
        const leafPlayer = colorToPlayer(player);
        const leafDepth = depth;
        const leafLibertyMap = libertyMapStack[leafDepth] ?? this.rootLibertyMap;
        const leafLibertyBuf = this.jobLibertyMapScratch.subarray(absIdx * BOARD_AREA, (absIdx + 1) * BOARD_AREA);
        leafLibertyBuf.set(leafLibertyMap);
        let prevLibertyMap: Uint8Array | undefined;
        let prevPrevLibertyMap: Uint8Array | undefined;
        if (leafDepth >= 1) {
          const prevLiberty = libertyMapStack[leafDepth - 1] ?? this.rootLibertyMap;
          const prevLibertyBuf = this.jobPrevLibertyMapScratch.subarray(absIdx * BOARD_AREA, (absIdx + 1) * BOARD_AREA);
          prevLibertyBuf.set(prevLiberty);
          prevLibertyMap = prevLibertyBuf;
          if (leafDepth >= 2) {
            const prevPrevLiberty = libertyMapStack[leafDepth - 2];
            if (prevPrevLiberty) {
              const prevPrevLibertyBuf = this.jobPrevPrevLibertyMapScratch.subarray(
                absIdx * BOARD_AREA,
                (absIdx + 1) * BOARD_AREA
              );
              prevPrevLibertyBuf.set(prevPrevLiberty);
              prevPrevLibertyMap = prevPrevLibertyBuf;
            }
          } else {
            const prevPrevLibertyBuf = this.jobPrevPrevLibertyMapScratch.subarray(
              absIdx * BOARD_AREA,
              (absIdx + 1) * BOARD_AREA
            );
            prevPrevLibertyBuf.set(this.rootPrevLibertyMap);
            prevPrevLibertyMap = prevPrevLibertyBuf;
          }
        }

        if (depth >= 1) {
          const lastIdx = depth - 1;
          undoMove(sim, undoMoves[lastIdx]!, undoPlayers[lastIdx]!, undoSnapshots[lastIdx]!, captureStack);

          if (lastIdx === 0) {
            // Leaf is a child of the root: prev state is the root, and prev-prev is the pre-root position.
            prevStones = this.rootStones;
            prevKoPoint = this.rootKoPoint;
            prevPrevStones = this.rootPrevStones;
            prevPrevKoPoint = this.rootPrevKoPoint;
          } else {
            const prevBuf = this.jobPrevStonesScratch.subarray(absIdx * BOARD_AREA, (absIdx + 1) * BOARD_AREA);
            prevBuf.set(sim.stones);
            prevStones = prevBuf;
            prevKoPoint = sim.koPoint;

            const secondIdx = depth - 2;
            undoMove(sim, undoMoves[secondIdx]!, undoPlayers[secondIdx]!, undoSnapshots[secondIdx]!, captureStack);

            if (secondIdx === 0) {
              // Leaf is depth 2: prev-prev is the root.
              prevPrevStones = this.rootStones;
              prevPrevKoPoint = this.rootKoPoint;
            } else {
              const prevPrevBuf = this.jobPrevPrevStonesScratch.subarray(absIdx * BOARD_AREA, (absIdx + 1) * BOARD_AREA);
              prevPrevBuf.set(sim.stones);
              prevPrevStones = prevPrevBuf;
              prevPrevKoPoint = sim.koPoint;
            }
          }
        }

        const recentMovesScratch = this.jobRecentMovesScratch[absIdx] ?? (this.jobRecentMovesScratch[absIdx] = []);
        const jobPath = this.jobPathStorage[absIdx] ?? (this.jobPathStorage[absIdx] = []);
        jobPath.length = path.length;
        for (let p = 0; p < path.length; p++) jobPath[p] = path[p]!;

        jobList.push({
          komi: getNNKomi(this.nnKomi, this.rules, pathMoves, undefined, pathMovesCount),
          leaf: node,
          path: jobPath,
          stones: leafStones,
          koPoint: leafKoPoint,
          libertyMap: leafLibertyBuf,
          prevStones,
          prevKoPoint,
          prevLibertyMap,
          prevPrevStones,
          prevPrevKoPoint,
          prevPrevLibertyMap,
          currentPlayer: leafPlayer,
          recentMoves: takeRecentMoves(this.rootMoves, pathMoves, 5, recentMovesScratch, pathMovesCount),
        });
      }
      return jobList;
    };

    // Backprop a completed batch of GPU evals into the tree.
    const backpropBatch = (jobs: MctsJob[], evals: NeuralEval[], inclOwnership: boolean) => {
      for (let i = 0; i < jobs.length; i++) {
        const job = jobs[i]!;
        const ev = evals[i]!;

        if (inclOwnership) {
          if (!ev.ownership) throw new Error('Missing ownership output');
          const ownershipSign = job.currentPlayer === 'black' ? 1 : -1;
          const own = new Float32Array(BOARD_AREA);
          const sym = ev.symmetry;
          const symOff = sym * BOARD_AREA;
          const symPosMap = sym === 0 ? null : getSymPosMap();
          for (let p = 0; p < BOARD_AREA; p++) {
            const symPos = sym === 0 ? p : symPosMap![symOff + p]!;
            own[p] = ownershipSign * Math.tanh(ev.ownership[symPos]! * this.outputScaleMultiplier);
          }
          job.leaf.ownership = own;
        }

        expandNode({
          node: job.leaf,
          stones: job.stones,
          koPoint: job.koPoint,
          policyLogits: ev.policy,
          policyLogitsSymmetry: ev.symmetry,
          passLogit: ev.passLogit,
          maxChildren: this.maxChildren,
          libertyMap: ev.libertyMap,
          policyOutputScaling: this.outputScaleMultiplier,
        });

        const leafValue = 2 * ev.blackWinProb - 1;
        const leafUtility = computeBlackUtilityFromEval({
          blackWinProb: ev.blackWinProb,
          blackNoResultProb: ev.blackNoResultProb,
          blackScoreMean: ev.blackScoreMean,
          blackScoreStdev: ev.blackScoreStdev,
          recentScoreCenter: this.recentScoreCenter,
        });
        job.leaf.nnUtility = leafUtility;
        const leafScoreMeanSq = ev.blackScoreStdev * ev.blackScoreStdev + ev.blackScoreMean * ev.blackScoreMean;
        const leafUtilitySq = leafUtility * leafUtility;
        for (const n of job.path) {
          n.visits += 1;
          n.valueSum += leafValue;
          n.scoreLeadSum += ev.blackScoreLead;
          n.scoreMeanSum += ev.blackScoreMean;
          n.scoreMeanSqSum += leafScoreMeanSq;
          n.utilitySum += leafUtility;
          n.utilitySqSum += leafUtilitySq;
          n.inFlight -= 1;
        }
        job.leaf.pendingEval = false;
      }
    };

    const includeOwnership = this.ownershipMode === 'tree';
    const discardJobs = (jobs: MctsJob[]) => {
      for (const job of jobs) {
        for (const node of job.path) node.inFlight--;
        job.leaf.pendingEval = false;
      }
    };

    // ── § 1.1 Pipelined main loop ────────────────────────────────────────────
    // 1. Collect batch A (slot 0) and dispatch to GPU immediately.
    // 2. While GPU runs A, collect batch B (slot 1) and dispatch B to GPU.
    //    (GPU queues B behind A; both execute serially on the hardware but
    //     we overlap the CPU feature-prep time with GPU execution.)
    // 3. Await A, backprop A, then loop: A's slot is now free for batch C.

    let currentJobs = collectBatch(0);
    if (currentJobs.length === 0) return shouldAbort?.() ?? false;

    let currentSlot = 0;
    let currentGpuPromise = dispatchEvalBatchAsync({
      model: this.model,
      includeOwnership,
      rules: this.rules,
      nnRandomize: this.nnRandomize,
      policyOptimism: POLICY_OPTIMISM,
      komi: this.komi,
      states: currentJobs,
      slot: currentSlot,
      randomSource: this.randomSource,
      evalContext: this.evalContext,
      featureCache: this.featureCache,
    });

    while (true) {
      if (shouldAbort?.()) {
        // Drain submitted work before the caller can reuse the tree or dispose
        // its model/backend. Cancelled evaluations are not counted as visits.
        try { await currentGpuPromise; } finally { discardJobs(currentJobs); }
        return true;
      }

      // CPU: collect next batch and dispatch it to the GPU while the current
      // batch is still executing.  Both batches are in flight simultaneously
      // but execute sequentially on the GPU hardware command queue.
      const nextSlot = 1 - currentSlot;
      const nextJobs = collectBatch(nextSlot);
      let nextGpuPromise: Promise<NeuralEval[]> | null = null;
      if (nextJobs.length > 0) {
        nextGpuPromise = dispatchEvalBatchAsync({
          model: this.model,
          includeOwnership,
          rules: this.rules,
          nnRandomize: this.nnRandomize,
          policyOptimism: POLICY_OPTIMISM,
          komi: this.komi,
          states: nextJobs,
          slot: nextSlot,
          randomSource: this.randomSource,
          evalContext: this.evalContext,
          featureCache: this.featureCache,
        });
        // Attach a handler immediately while awaiting the older batch below.
        void nextGpuPromise.catch(() => {});
      }

      // Await the current batch (GPU may already be done by now).
      let evals: NeuralEval[];
      try { evals = await currentGpuPromise; }
      catch (error) {
        discardJobs(currentJobs);
        if (nextGpuPromise) {
          try { await nextGpuPromise; } catch { /* Preserve the first error. */ }
          discardJobs(nextJobs);
        }
        throw error;
      }
      timeCheckCounter = 0;

      // Backprop while GPU executes the next batch.
      backpropBatch(currentJobs, evals, includeOwnership);
      maybeReport();

      if (shouldAbort?.()) {
        if (nextGpuPromise) {
          try { await nextGpuPromise; } finally { discardJobs(nextJobs); }
        }
        return true;
      }

      if (this.rootNode.visits >= maxVisits || timeExceeded()) {
        if (nextGpuPromise) {
          try { backpropBatch(nextJobs, await nextGpuPromise, includeOwnership); }
          catch (error) { discardJobs(nextJobs); throw error; }
          maybeReport();
        }
        break;
      }

      if (!nextGpuPromise) {
        // collectBatch for the pipelined slot often returns 0 while the current
        // batch is still in-flight (leaves marked pendingEval). After backprop,
        // retry on the current slot instead of ending the search after one batch.
        const retryJobs = collectBatch(currentSlot);
        if (retryJobs.length === 0) break;
        currentJobs = retryJobs;
        currentGpuPromise = dispatchEvalBatchAsync({
          model: this.model,
          includeOwnership,
          rules: this.rules,
          nnRandomize: this.nnRandomize,
          policyOptimism: POLICY_OPTIMISM,
          komi: this.komi,
          states: currentJobs,
          slot: currentSlot,
          randomSource: this.randomSource,
          evalContext: this.evalContext,
          featureCache: this.featureCache,
        });
        continue;
      }

      currentJobs = nextJobs;
      currentGpuPromise = nextGpuPromise;
      currentSlot = nextSlot;
    }

    return shouldAbort?.() ?? false;
    } finally {
      this.isRunning = false;
    }
  }

  getAnalysisLight(args: {
    topK: number;
    analysisPvLen: number;
    includeMovesOwnership?: boolean;
  }): ReturnType<MctsSearch['getAnalysis']> {
    return this.getAnalysis({ ...args, light: true, cloneBuffers: false });
  }

  getAnalysis(args: {
    topK: number;
    analysisPvLen: number;
    includeMovesOwnership?: boolean;
    cloneBuffers?: boolean;
    light?: boolean;
    ownershipRefreshIntervalMs?: number;
  }): {
    rootWinRate: number;
    rootScoreLead: number;
    rootScoreSelfplay: number;
    rootScoreStdev: number;
    rootVisits: number;
    ownership: FloatArray;
    ownershipStdev: FloatArray;
    policy: FloatArray;
    moves: Array<{
      x: number;
      y: number;
      winRate: number;
      winRateLost: number;
      scoreLead: number;
      scoreSelfplay: number;
      scoreStdev: number;
      visits: number;
      pointsLost: number;
      relativePointsLost: number;
      order: number;
      prior: number;
      pv: string[];
      ownership?: FloatArray;
    }>;
  } {
    const topK = Math.max(1, Math.min(args.topK, 50));
    const includeMovesOwnership = args.includeMovesOwnership === true;
    const light = args.light === true;
    const cloneBuffers = !light && args.cloneBuffers !== false;
    const analysisPvLen = Math.max(0, Math.min(args.analysisPvLen, 60));
    const pvDepth = 1 + analysisPvLen;

    const edges = this.rootNode.edges ?? [];
    const EMPTY_PV: string[] = [];
    const childStats: ChildWeightStats[] = [];
    const topMoves: Array<{
      edge: Edge;
      move: number;
      visits: number;
      winRate: number;
      scoreLead: number;
      scoreSelfplay: number;
      scoreStdev: number;
      prior: number;
      pv: string[];
      orderIndex: number;
    }> = [];
    let orderIndex = 0;
    let minIdx = -1;

    const isBetter = (a: (typeof topMoves)[number], b: (typeof topMoves)[number]) =>
      a.visits > b.visits || (a.visits === b.visits && a.orderIndex < b.orderIndex);
    const isWorse = (a: (typeof topMoves)[number], b: (typeof topMoves)[number]) =>
      a.visits < b.visits || (a.visits === b.visits && a.orderIndex > b.orderIndex);

    const updateMinIdx = () => {
      minIdx = 0;
      for (let i = 1; i < topMoves.length; i++) {
        if (isWorse(topMoves[i]!, topMoves[minIdx]!)) minIdx = i;
      }
    };

    for (const e of edges) {
      const child = e.child;
      if (!child || child.visits <= 0) continue;
      const q = child.valueSum / child.visits;
      const winRate = (q + 1) * 0.5;
      const scoreLead = child.scoreLeadSum / child.visits;
      const scoreSelfplay = child.scoreMeanSum / child.visits;
      const scoreMeanSq = child.scoreMeanSqSum / child.visits;
      const scoreStdev = Math.sqrt(Math.max(0, scoreMeanSq - scoreSelfplay * scoreSelfplay));
      const utility = child.utilitySum / child.visits;
      childStats.push({
        weightAdjusted: child.visits,
        selfUtility: utility,
        policy: e.prior,
        value: q,
        scoreLead,
        scoreMean: scoreSelfplay,
        scoreMeanSq,
      });
      const row = {
        edge: e,
        move: e.move,
        visits: child.visits,
        winRate,
        scoreLead,
        scoreSelfplay,
        scoreStdev,
        prior: e.prior,
        pv: EMPTY_PV,
        orderIndex: orderIndex++,
      };
      if (topMoves.length < topK) {
        topMoves.push(row);
        if (topMoves.length === topK) updateMinIdx();
      } else if (minIdx >= 0 && isBetter(row, topMoves[minIdx]!)) {
        topMoves[minIdx] = row;
        updateMinIdx();
      }
    }

    // Before MCTS visits land on children, show policy-prior candidates from the root NN eval.
    if (topMoves.length < topK) {
      const policyWinRate = (this.rootSelfValue + 1) * 0.5;
      const policyScoreStdev = Math.sqrt(
        Math.max(0, this.rootSelfScoreMeanSq - this.rootSelfScoreMean * this.rootSelfScoreMean)
      );
      const policyCandidates: (typeof topMoves)[number][] = [];
      for (const e of edges) {
        const child = e.child;
        // Unexpanded edges have child=null; include them via policy prior until MCTS visits land.
        if (child && child.visits > 0) continue;
        if (e.prior <= 0) continue;
        if (topMoves.some((m) => m.move === e.move)) continue;
        policyCandidates.push({
          edge: e,
          move: e.move,
          visits: 0,
          winRate: policyWinRate,
          scoreLead: this.rootSelfScoreLead,
          scoreSelfplay: this.rootSelfScoreMean,
          scoreStdev: policyScoreStdev,
          prior: e.prior,
          pv: EMPTY_PV,
          orderIndex: orderIndex++,
        });
      }
      policyCandidates.sort((a, b) => b.prior - a.prior);
      for (const row of policyCandidates) {
        if (topMoves.length >= topK) break;
        topMoves.push(row);
      }
    }

    topMoves.sort((a, b) => {
      const visitDiff = b.visits - a.visits;
      if (visitDiff !== 0) return visitDiff;
      const priorDiff = b.prior - a.prior;
      if (priorDiff !== 0) return priorDiff;
      return a.orderIndex - b.orderIndex;
    });
    for (const row of topMoves) row.pv = getPvForEdge(row.edge, pvDepth);

    const rootStats = computeWeightedRootStats({
      children: childStats,
      rootSelf: {
        value: this.rootSelfValue,
        scoreLead: this.rootSelfScoreLead,
        scoreMean: this.rootSelfScoreMean,
        scoreMeanSq: this.rootSelfScoreMeanSq,
        utility: this.rootSelfUtility,
        weight: 1,
      },
    });
    const rootWinRate = rootStats.rootWinRate;
    const rootScoreLead = rootStats.rootScoreLead;
    const rootScoreSelfplay = rootStats.rootScoreSelfplay;
    const rootScoreStdev = rootStats.rootScoreStdev;

    const best = topMoves[0] ?? null;
    const bestScoreLead = best ? best.scoreLead : rootScoreLead;
    const sign = this.currentPlayer === 'black' ? 1 : -1;

    const moves = topMoves.map((m, i) => {
      const pointsLost = sign * (rootScoreLead - m.scoreLead);
      const relativePointsLost = sign * (bestScoreLead - m.scoreLead);
      const winRateLost = sign * (rootWinRate - m.winRate);

      const x = m.move === PASS_MOVE ? -1 : m.move % BOARD_SIZE;
      const y = m.move === PASS_MOVE ? -1 : (m.move / BOARD_SIZE) | 0;

      return {
        x,
        y,
        winRate: m.winRate,
        winRateLost,
        scoreLead: m.scoreLead,
        scoreSelfplay: m.scoreSelfplay,
        scoreStdev: m.scoreStdev,
        visits: m.visits,
        pointsLost,
        relativePointsLost,
        order: i,
        prior: m.prior,
        pv: m.pv,
        ownership:
          includeMovesOwnership && m.edge.child?.ownership
            ? cloneBuffers
              ? new Float32Array(m.edge.child.ownership)
              : m.edge.child.ownership
            : undefined,
      };
    });

    let ownership: Float32Array;
    let ownershipStdev: Float32Array;
    if (light) {
      ownership = EMPTY_OWNERSHIP;
      ownershipStdev = EMPTY_POLICY;
    } else if (this.ownershipMode === 'tree') {
      const visits = this.rootNode.visits;
      let cached = this.treeOwnershipCache;
      const refreshIntervalMs = args.ownershipRefreshIntervalMs ?? 0;
      const now = this.clockSource();
      if (!cached) {
        cached = { visits, timestamp: now, ...averageTreeOwnership(this.rootNode) };
        this.treeOwnershipCache = cached;
      } else if (cached.visits !== visits && (refreshIntervalMs <= 0 || now - cached.timestamp >= refreshIntervalMs)) {
        cached = { visits, timestamp: now, ...averageTreeOwnership(this.rootNode) };
        this.treeOwnershipCache = cached;
      }
      ownership = cloneBuffers ? new Float32Array(cached.ownership) : cached.ownership;
      ownershipStdev = cloneBuffers ? new Float32Array(cached.ownershipStdev) : cached.ownershipStdev;
    } else {
      ownership = cloneBuffers ? new Float32Array(this.rootOwnership) : this.rootOwnership;
      ownershipStdev = new Float32Array(BOARD_AREA);
    }
    const policyOut = light ? EMPTY_POLICY : cloneBuffers ? new Float32Array(this.rootPolicy) : this.rootPolicy;

    return {
      rootWinRate,
      rootScoreLead,
      rootScoreSelfplay,
      rootScoreStdev,
      rootVisits: this.rootNode.visits,
      ownership,
      ownershipStdev,
      policy: policyOut,
      moves,
    };
  }
}

export async function analyzeMcts(args: {
  model: KataGoModelV8Tf;
  board: BoardState;
  previousBoard?: BoardState;
  previousPreviousBoard?: BoardState;
  previousPreviousPreviousBoard?: BoardState;
  koPoint?: number;
  currentPlayer: Player;
  moveHistory: Move[];
  komi: number;
  topK?: number;
  analysisPvLen?: number;
  wideRootNoise?: number;
  rules?: GameRules;
  nnRandomize?: boolean;
  visits?: number;
  maxTimeMs?: number;
  batchSize?: number;
  maxChildren?: number;
  rootSymmetrySamples?: number;
  regionOfInterest?: RegionOfInterest | null;
  randomSource?: RandomSource;
  clockSource?: ClockSource;
}): Promise<{
  rootWinRate: number;
  rootScoreLead: number;
  rootScoreSelfplay: number;
  rootScoreStdev: number;
  rootVisits: number;
  ownership: FloatArray; // len 361, +1 black owns, -1 white owns (tree-averaged)
  ownershipStdev: FloatArray; // len 361 (tree stdev)
  policy: FloatArray; // len 362, illegal = -1, pass at index 361
  moves: Array<{
    x: number;
    y: number;
    winRate: number;
    winRateLost: number;
    scoreLead: number;
    scoreSelfplay: number;
    scoreStdev: number;
    visits: number;
    pointsLost: number;
    relativePointsLost: number;
    order: number;
    prior: number;
    pv: string[];
  }>;
}> {
  const outputScaleMultiplier = args.model.postProcessParams?.outputScaleMultiplier ?? 1.0;
  const maxVisits = Math.max(16, Math.min(args.visits ?? 256, ENGINE_MAX_VISITS));
  const maxTimeMs = Math.max(25, Math.min(args.maxTimeMs ?? 800, ENGINE_MAX_TIME_MS));
  const batchSize = Math.max(1, Math.min(args.batchSize ?? (tf.getBackend() === 'webgpu' ? 16 : 4), 64));
  const maxChildren = Math.max(4, Math.min(args.maxChildren ?? 64, 361));
  const topK = Math.max(1, Math.min(args.topK ?? 10, 50));
  const analysisPvLen = Math.max(0, Math.min(args.analysisPvLen ?? 15, 60));
  const wideRootNoise = Math.max(0, Math.min(args.wideRootNoise ?? 0.04, 5));
  const rules: GameRules = args.rules ?? 'japanese';
  const nnRandomize = args.nnRandomize !== false;
  const rootSymmetrySamples = clampRootSymmetrySamples(args.rootSymmetrySamples ?? (tf.getBackend() === 'webgpu' && nnRandomize ? NUM_SYMMETRIES : 1));
  const pvDepth = 1 + analysisPvLen;
  const randomSource = args.randomSource ?? DefaultRandomSource;
  const clockSource = args.clockSource ?? DefaultClockSource;
  const rand = new Rand(randomSource);

  const rootStones = boardStateToStones(args.board);
  const { rootKoPoint, rootPrevKoPoint, rootPrevPrevKoPoint } = resolveKoState({
    koPoint: args.koPoint,
    previousBoard: args.previousBoard,
    previousPreviousBoard: args.previousPreviousBoard,
    previousPreviousPreviousBoard: args.previousPreviousPreviousBoard,
    moveHistory: args.moveHistory,
  });

  const rootPrevStones = args.previousBoard ? boardStateToStones(args.previousBoard) : rootStones;
  const rootPrevPrevStones = args.previousPreviousBoard ? boardStateToStones(args.previousPreviousBoard) : rootPrevStones;

  const rootMoves: RecentMove[] = args.moveHistory.map((m) => ({
    move: m.x < 0 || m.y < 0 ? PASS_MOVE : m.y * BOARD_SIZE + m.x,
    player: m.player,
  }));

  const rootPos: SimPosition = { stones: rootStones.slice(), koPoint: rootKoPoint };
  const rootNode = new Node(playerToColor(args.currentPlayer));

  const rootEval = await evaluateRootEval({
    model: args.model,
    includeOwnership: true,
    rules,
    nnRandomize,
    rootSymmetrySamples,
    policyOptimism: ROOT_POLICY_OPTIMISM,
    komi: getNNKomi(args.komi, rules, rootMoves, rootMoves.length === 0 ? rootStones : undefined),
    state: {
      stones: rootPos.stones,
      koPoint: rootPos.koPoint,
      prevStones: rootPrevStones,
      prevKoPoint: rootPrevKoPoint,
      prevPrevStones: rootPrevPrevStones,
      prevPrevKoPoint: rootPrevPrevKoPoint,
      currentPlayer: args.currentPlayer,
      recentMoves: takeRecentMoves(rootMoves, [], 5),
    },
  });
  if (!rootEval.ownership) throw new Error('Missing ownership output');

  const rootOwnershipSign = args.currentPlayer === 'black' ? 1 : -1;
  const rootOwnership = new Float32Array(BOARD_AREA);
  const rootSym = rootEval.symmetry;
  const rootSymOff = rootSym * BOARD_AREA;
  const symPosMap = rootSym === 0 ? null : getSymPosMap();
  for (let i = 0; i < BOARD_AREA; i++) {
    const symPos = rootSym === 0 ? i : symPosMap![rootSymOff + i]!;
    rootOwnership[i] = rootOwnershipSign * Math.tanh(rootEval.ownership[symPos]! * outputScaleMultiplier);
  }

  const rootAllowedMoves = buildAllowedMovesMask(args.regionOfInterest);
  const rootPolicy = new Float32Array(BOARD_AREA + 1);
  expandNode({
    node: rootNode,
    stones: rootPos.stones,
    koPoint: rootPos.koPoint,
    policyLogits: rootEval.policy,
    policyLogitsSymmetry: rootSym,
    passLogit: rootEval.passLogit,
    maxChildren,
    libertyMap: rootEval.libertyMap,
    allowedMoves: rootAllowedMoves ?? undefined,
    policyOut: rootPolicy,
    policyOutputScaling: outputScaleMultiplier,
  });
  rootNode.ownership = rootOwnership;

  const recentScoreCenter = computeRecentScoreCenter(-rootEval.blackScoreMean);

  const rootValue = 2 * rootEval.blackWinProb - 1;
  const rootUtility = computeBlackUtilityFromEval({
    blackWinProb: rootEval.blackWinProb,
    blackNoResultProb: rootEval.blackNoResultProb,
    blackScoreMean: rootEval.blackScoreMean,
    blackScoreStdev: rootEval.blackScoreStdev,
    recentScoreCenter,
  });
  rootNode.visits = 1;
  rootNode.valueSum = rootValue;
  rootNode.scoreLeadSum = rootEval.blackScoreLead;
  rootNode.scoreMeanSum = rootEval.blackScoreMean;
  const rootScoreMeanSq = rootEval.blackScoreStdev * rootEval.blackScoreStdev + rootEval.blackScoreMean * rootEval.blackScoreMean;
  rootNode.scoreMeanSqSum = rootScoreMeanSq;
  rootNode.utilitySum = rootUtility;
  rootNode.utilitySqSum = rootUtility * rootUtility;
  rootNode.nnUtility = rootUtility;

  const sim: SimPosition = { stones: rootStones.slice(), koPoint: rootKoPoint };
  const captureStack: number[] = [];
  const undoMoves: number[] = [];
  const undoPlayers: StoneColor[] = [];
  const undoSnapshots: UndoSnapshot[] = [];
  const pathMoves: RecentMove[] = [];

  const deadline = clockSource() + maxTimeMs;
  let timeCheckCounter = 0;
  const timeCheckMask = 0x1f;
  const timeExceeded = (): boolean => {
    if ((timeCheckCounter++ & timeCheckMask) !== 0) return false;
    return clockSource() >= deadline;
  };

  while (rootNode.visits < maxVisits && !timeExceeded()) {
    const jobs: Array<{
      komi: number;
      leaf: Node;
      path: Node[];
      stones: Uint8Array;
      koPoint: number;
      prevStones: Uint8Array;
      prevKoPoint: number;
      prevPrevStones: Uint8Array;
      prevPrevKoPoint: number;
      currentPlayer: Player;
      recentMoves: RecentMove[];
    }> = [];

    let attempts = 0;
    while (jobs.length < batchSize && rootNode.visits + jobs.length < maxVisits && !timeExceeded()) {
      attempts++;
      if (attempts > batchSize * 8) break;

      undoMoves.length = 0;
      undoPlayers.length = 0;
      undoSnapshots.length = 0;
      pathMoves.length = 0;
      sim.stones.set(rootStones);
      sim.koPoint = rootKoPoint;

      const path: Node[] = [rootNode];
      let node = rootNode;
      let player = rootNode.playerToMove;

      while (node.edges && node.edges.length > 0) {
        const e = selectEdge(node, node === rootNode, wideRootNoise, rand);
        const move = e.move;

        const snapshot = playMove(sim, move, player, captureStack);
        undoMoves.push(move);
        undoPlayers.push(player);
        undoSnapshots.push(snapshot);
        const pathIdx = pathMoves.length;
        const pathPlayer = colorToPlayer(player);
        let pathEntry = pathMoves[pathIdx];
        if (!pathEntry) {
          pathEntry = { move, player: pathPlayer };
          pathMoves[pathIdx] = pathEntry;
        } else {
          pathEntry.move = move;
          pathEntry.player = pathPlayer;
        }
        pathMoves.length = pathIdx + 1;

        if (!e.child) e.child = new Node(opponentOf(player));
        node = e.child;
        player = node.playerToMove;
        path.push(node);

        if (!node.edges) break;
      }

      if (node.pendingEval) {
        for (let i = undoMoves.length - 1; i >= 0; i--) {
          undoMove(sim, undoMoves[i]!, undoPlayers[i]!, undoSnapshots[i]!, captureStack);
        }
        continue;
      }

      node.pendingEval = true;
      for (const n of path) n.inFlight++;

      const leafStones = sim.stones.slice();
      const leafKoPoint = sim.koPoint;
      let prevStones = leafStones;
      let prevKoPoint = leafKoPoint;
      let prevPrevStones = leafStones;
      let prevPrevKoPoint = leafKoPoint;
      const leafPlayer = colorToPlayer(player);

      if (undoMoves.length >= 1) {
        const lastIdx = undoMoves.length - 1;
        undoMove(sim, undoMoves[lastIdx]!, undoPlayers[lastIdx]!, undoSnapshots[lastIdx]!, captureStack);

        if (lastIdx === 0) {
          prevStones = rootStones;
          prevKoPoint = rootKoPoint;
          prevPrevStones = rootPrevStones;
          prevPrevKoPoint = rootPrevKoPoint;
        } else {
          prevStones = sim.stones.slice();
          prevKoPoint = sim.koPoint;

          const secondIdx = undoMoves.length - 2;
          undoMove(sim, undoMoves[secondIdx]!, undoPlayers[secondIdx]!, undoSnapshots[secondIdx]!, captureStack);

          if (secondIdx === 0) {
            prevPrevStones = rootStones;
            prevPrevKoPoint = rootKoPoint;
          } else {
            prevPrevStones = sim.stones.slice();
            prevPrevKoPoint = sim.koPoint;
          }

          for (let i = secondIdx - 1; i >= 0; i--) {
            undoMove(sim, undoMoves[i]!, undoPlayers[i]!, undoSnapshots[i]!, captureStack);
          }
        }
      }

      jobs.push({
        komi: getNNKomi(getNNKomi(args.komi, rules, rootMoves, rootMoves.length === 0 ? rootStones : undefined), rules, pathMoves),
        leaf: node,
        path,
        stones: leafStones,
        koPoint: leafKoPoint,
        prevStones,
        prevKoPoint,
        prevPrevStones,
        prevPrevKoPoint,
        currentPlayer: leafPlayer,
        recentMoves: takeRecentMoves(rootMoves, pathMoves, 5),
      });
    }

    if (jobs.length === 0) break;

    const evals = await evaluateBatch({
      model: args.model,
      includeOwnership: true,
      rules,
      nnRandomize,
      policyOptimism: POLICY_OPTIMISM,
      komi: args.komi,
      states: jobs,
      randomSource,
    });
    timeCheckCounter = 0;

    for (let i = 0; i < jobs.length; i++) {
      const job = jobs[i]!;
      const ev = evals[i]!;
      if (!ev.ownership) throw new Error('Missing ownership output');

      const ownershipSign = job.currentPlayer === 'black' ? 1 : -1;
      const own = new Float32Array(BOARD_AREA);
      const sym = ev.symmetry;
      const symOff = sym * BOARD_AREA;
      const symPosMap = sym === 0 ? null : getSymPosMap();
      for (let p = 0; p < BOARD_AREA; p++) {
        const symPos = sym === 0 ? p : symPosMap![symOff + p]!;
        own[p] = ownershipSign * Math.tanh(ev.ownership[symPos]! * outputScaleMultiplier);
      }
      job.leaf.ownership = own;

      expandNode({
        node: job.leaf,
        stones: job.stones,
        koPoint: job.koPoint,
        policyLogits: ev.policy,
        policyLogitsSymmetry: ev.symmetry,
        passLogit: ev.passLogit,
        maxChildren,
        libertyMap: ev.libertyMap,
        policyOutputScaling: outputScaleMultiplier,
      });

      const leafValue = 2 * ev.blackWinProb - 1;
      const leafUtility = computeBlackUtilityFromEval({
        blackWinProb: ev.blackWinProb,
        blackNoResultProb: ev.blackNoResultProb,
        blackScoreMean: ev.blackScoreMean,
        blackScoreStdev: ev.blackScoreStdev,
        recentScoreCenter,
      });
      job.leaf.nnUtility = leafUtility;
      for (const n of job.path) {
        n.visits += 1;
        n.valueSum += leafValue;
        n.scoreLeadSum += ev.blackScoreLead;
        n.scoreMeanSum += ev.blackScoreMean;
        n.scoreMeanSqSum += ev.blackScoreStdev * ev.blackScoreStdev + ev.blackScoreMean * ev.blackScoreMean;
        n.utilitySum += leafUtility;
        n.utilitySqSum += leafUtility * leafUtility;
        n.inFlight -= 1;
      }
      job.leaf.pendingEval = false;
    }
  }

  const edges = rootNode.edges ?? [];
  const childStats: ChildWeightStats[] = [];
  const moveRows: Array<{
    edge: Edge;
    move: number;
    visits: number;
    winRate: number;
    scoreLead: number;
    scoreSelfplay: number;
    scoreStdev: number;
    prior: number;
    pv: string[];
  }> = [];

  for (const e of edges) {
    const child = e.child;
    if (!child || child.visits <= 0) continue;
    const q = child.valueSum / child.visits;
    const winRate = (q + 1) * 0.5;
    const scoreLead = child.scoreLeadSum / child.visits;
    const scoreSelfplay = child.scoreMeanSum / child.visits;
    const scoreMeanSq = child.scoreMeanSqSum / child.visits;
    const scoreStdev = Math.sqrt(Math.max(0, scoreMeanSq - scoreSelfplay * scoreSelfplay));
    const utility = child.utilitySum / child.visits;
    childStats.push({
      weightAdjusted: child.visits,
      selfUtility: utility,
      policy: e.prior,
      value: q,
      scoreLead,
      scoreMean: scoreSelfplay,
      scoreMeanSq,
    });
    moveRows.push({
      edge: e,
      move: e.move,
      visits: child.visits,
      winRate,
      scoreLead,
      scoreSelfplay,
      scoreStdev,
      prior: e.prior,
      pv: getPvForEdge(e, pvDepth),
    });
  }

  moveRows.sort((a, b) => b.visits - a.visits);

  const rootStats = computeWeightedRootStats({
    children: childStats,
    rootSelf: {
      value: rootValue,
      scoreLead: rootEval.blackScoreLead,
      scoreMean: rootEval.blackScoreMean,
      scoreMeanSq: rootScoreMeanSq,
      utility: rootUtility,
      weight: 1,
    },
  });
  const rootWinRate = rootStats.rootWinRate;
  const rootScoreLead = rootStats.rootScoreLead;
  const rootScoreSelfplay = rootStats.rootScoreSelfplay;
  const rootScoreStdev = rootStats.rootScoreStdev;

  const topMoves = moveRows.slice(0, Math.min(topK, moveRows.length));
  const best = topMoves[0] ?? null;
  const bestScoreLead = best ? best.scoreLead : rootScoreLead;
  const sign = args.currentPlayer === 'black' ? 1 : -1;

  const moves = topMoves.map((m) => {
    const pointsLost = sign * (rootScoreLead - m.scoreLead);
    const relativePointsLost = sign * (bestScoreLead - m.scoreLead);
    const winRateLost = sign * (rootWinRate - m.winRate);

    const x = m.move === PASS_MOVE ? -1 : m.move % BOARD_SIZE;
    const y = m.move === PASS_MOVE ? -1 : (m.move / BOARD_SIZE) | 0;

    return {
      x,
      y,
      winRate: m.winRate,
      winRateLost,
      scoreLead: m.scoreLead,
      scoreSelfplay: m.scoreSelfplay,
      scoreStdev: m.scoreStdev,
      visits: m.visits,
      pointsLost,
      relativePointsLost,
      order: 0,
      prior: m.prior,
      pv: m.pv,
    };
  });

  moves.sort((a, b) => b.visits - a.visits);
  moves.forEach((m, i) => (m.order = i));

  const { ownership, ownershipStdev } = averageTreeOwnership(rootNode);
  return {
    rootWinRate,
    rootScoreLead,
    rootScoreSelfplay,
    rootScoreStdev,
    rootVisits: rootNode.visits,
    ownership,
    ownershipStdev,
    policy: rootPolicy,
    moves,
  };
}
