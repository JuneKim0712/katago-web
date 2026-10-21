import { PASS_MOVE, BOARD_SIZE, BOARD_AREA } from '../fastBoard.js';
import type { Node, Edge } from './tree.js';

export function moveToGtp(move: number, boardSize = BOARD_SIZE): string {
  if (move === PASS_MOVE || move === boardSize * boardSize) return 'pass';
  const x = move % boardSize;
  const y = (move / boardSize) | 0;
  const col = x >= 8 ? x + 1 : x; // Skip 'I'
  const letter = String.fromCharCode(65 + col);
  return `${letter}${boardSize - y}`;
}

export function buildPv(edge: Edge, maxDepth: number, boardSize = BOARD_SIZE): string[] {
  const pvMoves: number[] = [edge.move];
  let node = edge.child;
  let depth = 1;

  while (node && node.edges && node.edges.length > 0 && depth < maxDepth) {
    let best: Edge | null = null;
    let bestVisits = 0;
    for (const e of node.edges) {
      const v = e.child ? e.child.visits : 0;
      if (v > bestVisits) {
        bestVisits = v;
        best = e;
      }
    }
    if (!best || bestVisits <= 0) break;
    pvMoves.push(best.move);
    node = best.child;
    depth++;
  }

  return pvMoves.map((m) => moveToGtp(m, boardSize));
}

export function getPvForEdge(edge: Edge, maxDepth: number, boardSize = BOARD_SIZE): string[] {
  const visits = edge.child?.visits ?? 0;
  const cache = edge.pvCache;
  if (cache && cache.visits === visits && cache.depth === maxDepth) return cache.pv;
  const pv = buildPv(edge, maxDepth, boardSize);
  edge.pvCache = { visits, depth: maxDepth, pv };
  return pv;
}

export function averageTreeOwnership(
  node: Node,
  boardArea = BOARD_AREA
): { ownership: Float32Array; ownershipStdev: Float32Array } {
  const out = new Float32Array(boardArea);
  const outSq = new Float32Array(boardArea);

  const visits = node.visits;
  const minProp = 0.5 / Math.pow(Math.max(1, visits), 0.75);
  const pruneProp = minProp * 0.01;

  const accumulate = (map: Float32Array, prop: number) => {
    for (let i = 0; i < boardArea; i++) {
      const v = map[i]!;
      out[i] += prop * v;
      outSq[i] += prop * v * v;
    }
  };

  const traverse = (n: Node, desiredProp: number): boolean => {
    if (!n.ownership) return false;

    if (desiredProp < minProp) {
      accumulate(n.ownership, desiredProp);
      return true;
    }

    const edges = n.edges;
    if (!edges || edges.length === 0) {
      accumulate(n.ownership, desiredProp);
      return true;
    }

    // Two-pass edge scan: pass 1 computes weight sums with zero allocations
    let childrenWeightSum = 0;
    let relativeChildrenWeightSum = 0;
    for (const e of edges) {
      const child = e.child;
      if (!child || child.visits <= 0) continue;
      const w = child.visits;
      childrenWeightSum += w;
      relativeChildrenWeightSum += w * w;
    }

    const parentNNWeight = 1.0;
    const denom = childrenWeightSum + parentNNWeight;
    const desiredPropFromChildren = denom > 0 ? (desiredProp * childrenWeightSum) / denom : 0;
    let selfProp = denom > 0 ? (desiredProp * parentNNWeight) / denom : desiredProp;

    if (desiredPropFromChildren <= 0 || relativeChildrenWeightSum <= 0) {
      selfProp += desiredPropFromChildren;
    } else {
      // Pass 2: recurse over the exact same edges in the same order
      for (const e of edges) {
        const child = e.child;
        if (!child || child.visits <= 0) continue;
        const w = child.visits;
        const childProp = (w * w * desiredPropFromChildren) / relativeChildrenWeightSum;
        if (childProp < pruneProp) {
          selfProp += childProp;
          continue;
        }
        const ok = traverse(child, childProp);
        if (!ok) selfProp += childProp;
      }
    }

    accumulate(n.ownership, selfProp);
    return true;
  };

  traverse(node, 1.0);

  const stdev = new Float32Array(boardArea);
  for (let i = 0; i < boardArea; i++) {
    const mean = out[i]!;
    const variance = outSq[i]! - mean * mean;
    stdev[i] = Math.sqrt(Math.max(0, variance));
  }

  return { ownership: out, ownershipStdev: stdev };
}
