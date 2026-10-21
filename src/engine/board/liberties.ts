import {
  EMPTY,
  type StoneColor,
  type BoardContext,
  getDefaultBoardContext,
} from './context.js';

export let lastFindLibertiesGroupLen = 0;

export function collectGroupAndLiberties(
  stones: Uint8Array,
  start: number,
  color: StoneColor,
  maxLibertiesToCount: number,
  context: BoardContext = getDefaultBoardContext()
): { groupLen: number; liberties: number } {
  const stamp = context.nextBfsStamp();
  const scratch = context.scratch;
  const geom = context.geometry;
  const visited = scratch.visited;
  const libVisited = scratch.libVisited;
  const groupBuf = scratch.groupBuf;
  const stackBuf = scratch.stackBuf;
  const nStarts = geom.neighborStarts;
  const nCounts = geom.neighborCounts;
  const neighbors = geom.neighbors;

  let groupLen = 0;
  let stackLen = 0;
  let liberties = 0;

  visited[start] = stamp;
  stackBuf[stackLen++] = start;

  while (stackLen > 0) {
    const p = stackBuf[--stackLen]!;
    groupBuf[groupLen++] = p;

    const nStart = nStarts[p]!;
    const nCount = nCounts[p]!;
    for (let i = 0; i < nCount; i++) {
      const n = neighbors[nStart + i]!;
      const c = stones[n] as StoneColor;
      if (c === EMPTY) {
        if (liberties < maxLibertiesToCount && libVisited[n] !== stamp) {
          libVisited[n] = stamp;
          liberties++;
        }
      } else if (c === color) {
        if (visited[n] !== stamp) {
          visited[n] = stamp;
          stackBuf[stackLen++] = n;
        }
      }
    }
  }

  return { groupLen, liberties };
}

export function findLibertiesIntoBuf(
  stones: Uint8Array,
  start: number,
  color: StoneColor,
  buf: Int16Array,
  bufIdx: number,
  max: number,
  context: BoardContext = getDefaultBoardContext()
): number {
  const stamp = context.nextBfsStamp();
  const scratch = context.scratch;
  const geom = context.geometry;
  const visited = scratch.visited;
  const libVisited = scratch.libVisited;
  const groupBuf = scratch.groupBuf;
  const stackBuf = scratch.stackBuf;
  const nStarts = geom.neighborStarts;
  const nCounts = geom.neighborCounts;
  const neighbors = geom.neighbors;

  let stackLen = 0;
  let groupLen = 0;
  let liberties = 0;

  visited[start] = stamp;
  stackBuf[stackLen++] = start;

  while (stackLen > 0) {
    const p = stackBuf[--stackLen]!;
    groupBuf[groupLen++] = p;

    const nStart = nStarts[p]!;
    const nCount = nCounts[p]!;
    for (let i = 0; i < nCount; i++) {
      const n = neighbors[nStart + i]!;
      const c = stones[n] as StoneColor;
      if (c === EMPTY) {
        if (liberties < max && libVisited[n] !== stamp) {
          libVisited[n] = stamp;
          buf[bufIdx + liberties] = n;
          liberties++;
        }
      } else if (c === color) {
        if (visited[n] !== stamp) {
          visited[n] = stamp;
          stackBuf[stackLen++] = n;
        }
      }
    }
  }

  lastFindLibertiesGroupLen = groupLen;
  return liberties;
}

export function computeLibertyMapInto(
  stones: Uint8Array,
  out: Uint8Array,
  context: BoardContext = getDefaultBoardContext()
): Uint8Array {
  const area = context.geometry.area;
  if (out.length !== area) {
    throw new Error(`computeLibertyMapInto: expected out length ${area}, got ${out.length}`);
  }
  out.fill(0);
  const stamp = context.nextGroupSeenStamp();
  const groupSeen = context.scratch.groupSeen;
  const groupBuf = context.scratch.groupBuf;

  for (let p = 0; p < area; p++) {
    const c = stones[p] as StoneColor;
    if (c === EMPTY) continue;
    if (groupSeen[p] === stamp) continue;

    const { groupLen, liberties } = collectGroupAndLiberties(stones, p, c, 4, context);
    const capLibs = liberties >= 4 ? 4 : liberties;
    for (let i = 0; i < groupLen; i++) {
      const gp = groupBuf[i]!;
      out[gp] = capLibs;
      groupSeen[gp] = stamp;
    }
  }

  return out;
}

export function updateLibertyMapForSeeds(
  stones: Uint8Array,
  seeds: Int16Array,
  seedCount: number,
  out: Uint8Array,
  context: BoardContext = getDefaultBoardContext()
): void {
  const area = context.geometry.area;
  if (out.length !== area) {
    throw new Error(`updateLibertyMapForSeeds: expected out length ${area}, got ${out.length}`);
  }
  if (seedCount <= 0) return;
  const stamp = context.nextGroupSeenStamp();
  const groupSeen = context.scratch.groupSeen;
  const groupBuf = context.scratch.groupBuf;

  for (let i = 0; i < seedCount; i++) {
    const p = seeds[i]!;
    const c = stones[p] as StoneColor;
    if (c === EMPTY) continue;
    if (groupSeen[p] === stamp) continue;

    const { groupLen, liberties } = collectGroupAndLiberties(stones, p, c, 4, context);
    const capLibs = liberties >= 4 ? 4 : liberties;
    for (let j = 0; j < groupLen; j++) {
      const gp = groupBuf[j]!;
      out[gp] = capLibs;
      groupSeen[gp] = stamp;
    }
  }
}

export function computeLibertyMap(
  stones: Uint8Array,
  context: BoardContext = getDefaultBoardContext()
): Uint8Array {
  return computeLibertyMapInto(stones, new Uint8Array(context.geometry.area), context);
}
