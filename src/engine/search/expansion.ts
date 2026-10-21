import {
  EMPTY,
  PASS_MOVE,
  BOARD_AREA,
  NEIGHBOR_STARTS,
  NEIGHBOR_COUNTS,
  NEIGHBOR_LIST,
  opponentOf,
  computeLibertyMap,
  type StoneColor,
} from '../fastBoard.js';
import { getSymPosMap } from '../features/encoder.js';
import type { Node, Edge } from './tree.js';

export type ExpandScratch = {
  moves: Int16Array;
  logits: Float32Array;
  priors: Float64Array;
  topMoves: Int16Array;
  topPriors: Float64Array;
  order: number[];
};

let expandScratch: ExpandScratch | null = null;
let expandScratchBoardArea = 0;

export const getExpandScratch = (area = BOARD_AREA): ExpandScratch => {
  if (!expandScratch || expandScratchBoardArea !== area) {
    expandScratch = {
      moves: new Int16Array(area),
      logits: new Float32Array(area),
      priors: new Float64Array(area),
      topMoves: new Int16Array(area),
      topPriors: new Float64Array(area),
      order: [],
    };
    expandScratchBoardArea = area;
  }
  return expandScratch;
};

export function expandNode(args: {
  node: Node;
  stones: Uint8Array;
  koPoint: number;
  policyLogits: ArrayLike<number>; // len 361 (in symmetry space if policyLogitsSymmetry != 0)
  policyLogitsSymmetry?: number; // 0..7, where 0 is identity
  passLogit: number;
  maxChildren: number;
  libertyMap?: Uint8Array;
  allowedMoves?: Uint8Array;
  policyOut?: Float32Array; // len 362, illegal = -1, pass at index 361
  policyOutputScaling?: number;
}): void {
  const { node, stones, koPoint, policyLogits, passLogit, maxChildren } = args;
  const policyScale = args.policyOutputScaling ?? 1.0;
  const pla = node.playerToMove;
  const opp = opponentOf(pla);
  const sym = args.policyLogitsSymmetry ?? 0;
  const symOff = sym * BOARD_AREA;
  const symPosMap = sym === 0 ? null : getSymPosMap();

  const libs = args.libertyMap ?? computeLibertyMap(stones);

  const scratch = getExpandScratch();
  const movesScratch = scratch.moves;
  const logitsScratch = scratch.logits;
  const priorsScratch = scratch.priors;
  let moveCount = 0;
  const passLogitScaled = passLogit * policyScale;
  let maxLogit = passLogitScaled;
  const allowedMoves = args.allowedMoves;
  for (let p = 0; p < BOARD_AREA; p++) {
    if (allowedMoves && allowedMoves[p] === 0) continue;
    if (stones[p] !== EMPTY) continue;
    if (p === koPoint) continue;

    let hasEmptyNeighbor = false;
    let captures = false;
    let connectsToSafeGroup = false;

    const nStart = NEIGHBOR_STARTS[p]!;
    const nCount = NEIGHBOR_COUNTS[p]!;

    for (let i = 0; i < nCount; i++) {
      const n = NEIGHBOR_LIST[nStart + i]!;
      const c = stones[n] as StoneColor;
      if (c === EMPTY) {
        hasEmptyNeighbor = true;
        break;
      }
      if (c === opp) {
        if (libs[n] === 1) {
          captures = true;
          break;
        }
        continue;
      }
      if (c === pla && libs[n] > 1) {
        connectsToSafeGroup = true;
        break;
      }
    }

    if (!hasEmptyNeighbor && !captures && !connectsToSafeGroup) continue;
    const symPos = sym === 0 ? p : symPosMap![symOff + p]!;
    const logit = policyLogits[symPos]! * policyScale;
    movesScratch[moveCount] = p;
    logitsScratch[moveCount] = logit;
    if (logit > maxLogit) maxLogit = logit;
    moveCount++;
  }

  let sum = 0;
  for (let i = 0; i < moveCount; i++) {
    const v = Math.exp(logitsScratch[i]! - maxLogit);
    priorsScratch[i] = v;
    sum += v;
  }
  const passPriorRaw = Math.exp(passLogitScaled - maxLogit);
  sum += passPriorRaw;
  const invSum = 1.0 / sum;
  for (let i = 0; i < moveCount; i++) priorsScratch[i] *= invSum;
  const passPrior = passPriorRaw * invSum;

  if (args.policyOut) {
    const out = args.policyOut;
    out.fill(-1);
    for (let i = 0; i < moveCount; i++) out[movesScratch[i]!] = priorsScratch[i]! as number;
    out[PASS_MOVE] = passPrior as number;
  }

  const topMoves = scratch.topMoves;
  const topPriors = scratch.topPriors;
  const maxKids = Math.max(0, maxChildren);
  let topCount = 0;
  let minIdx = 0;
  for (let i = 0; i < moveCount; i++) {
    const prior = priorsScratch[i]!;
    if (topCount < maxKids) {
      topMoves[topCount] = movesScratch[i]!;
      topPriors[topCount] = prior;
      topCount++;
      if (topCount === maxKids) {
        minIdx = 0;
        for (let j = 1; j < topCount; j++) {
          if (topPriors[j]! < topPriors[minIdx]!) minIdx = j;
        }
      }
    } else if (maxKids > 0 && prior > topPriors[minIdx]!) {
      topMoves[minIdx] = movesScratch[i]!;
      topPriors[minIdx] = prior;
      minIdx = 0;
      for (let j = 1; j < topCount; j++) {
        if (topPriors[j]! < topPriors[minIdx]!) minIdx = j;
      }
    }
  }

  const order = scratch.order;
  order.length = topCount;
  for (let i = 0; i < topCount; i++) order[i] = i;
  order.sort((a, b) => {
    const diff = topPriors[b]! - topPriors[a]!;
    if (diff !== 0) return diff;
    return topMoves[a]! - topMoves[b]!;
  });

  const edges: Edge[] = new Array(topCount + 1);
  let edgeIdx = 0;
  for (let i = 0; i < order.length; i++) {
    const idx = order[i]!;
    edges[edgeIdx++] = { move: topMoves[idx]!, prior: topPriors[idx]!, child: null };
  }
  edges[edgeIdx++] = { move: PASS_MOVE, prior: passPrior, child: null };

  node.edges = edges;
}
