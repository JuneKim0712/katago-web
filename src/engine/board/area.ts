import {
  EMPTY,
  BLACK,
  WHITE,
  opponentOf,
  type StoneColor,
  type BoardContext,
  getDefaultBoardContext,
} from './context.js';

function buildGroups(stones: Uint8Array, context: BoardContext): number {
  const geom = context.geometry;
  const scratch = context.scratch;
  const area = geom.area;
  const groupIndexByPos = scratch.groupIndexByPos;
  const groupColorByGroup = scratch.groupColorByGroup;
  const groupStartByGroup = scratch.groupStartByGroup;
  const groupLenByGroup = scratch.groupLenByGroup;
  const groupStonesFlat = scratch.groupStonesFlat;
  const stackBuf = scratch.stackBuf;
  const nStarts = geom.neighborStarts;
  const nCounts = geom.neighborCounts;
  const neighbors = geom.neighbors;

  groupIndexByPos.fill(-1);
  let numGroups = 0;
  let flat = 0;

  for (let p = 0; p < area; p++) {
    const c = stones[p] as StoneColor;
    if (c === EMPTY) continue;
    if (groupIndexByPos[p] !== -1) continue;

    const groupIdx = numGroups++;
    groupColorByGroup[groupIdx] = c;
    groupStartByGroup[groupIdx] = flat;

    let stackLen = 0;
    stackBuf[stackLen++] = p;
    groupIndexByPos[p] = groupIdx;

    let groupLen = 0;
    while (stackLen > 0) {
      const cur = stackBuf[--stackLen]!;
      groupStonesFlat[flat++] = cur;
      groupLen++;

      const nStart = nStarts[cur]!;
      const nCount = nCounts[cur]!;
      for (let i = 0; i < nCount; i++) {
        const n = neighbors[nStart + i]!;
        if ((stones[n] as StoneColor) !== c) continue;
        if (groupIndexByPos[n] !== -1) continue;
        groupIndexByPos[n] = groupIdx;
        stackBuf[stackLen++] = n;
      }
    }

    groupLenByGroup[groupIdx] = groupLen;
  }

  return numGroups;
}

function isAdjacentToColor(stones: Uint8Array, pos: number, color: StoneColor, context: BoardContext): boolean {
  const geom = context.geometry;
  const nStart = geom.neighborStarts[pos]!;
  const nCount = geom.neighborCounts[pos]!;
  const neighbors = geom.neighbors;
  for (let i = 0; i < nCount; i++) {
    const n = neighbors[nStart + i]!;
    if ((stones[n] as StoneColor) === color) return true;
  }
  return false;
}

function isAdjacentToPlaGroup(
  stones: Uint8Array,
  pos: number,
  plaColor: StoneColor,
  plaGroup: number,
  context: BoardContext
): boolean {
  const geom = context.geometry;
  const scratch = context.scratch;
  const nStart = geom.neighborStarts[pos]!;
  const nCount = geom.neighborCounts[pos]!;
  const neighbors = geom.neighbors;
  const groupIndexByPos = scratch.groupIndexByPos;
  for (let i = 0; i < nCount; i++) {
    const n = neighbors[nStart + i]!;
    if ((stones[n] as StoneColor) !== plaColor) continue;
    if (groupIndexByPos[n] === plaGroup) return true;
  }
  return false;
}

function calculateAreaForPla(
  args: {
    stones: Uint8Array;
    numGroups: number;
    plaColor: StoneColor;
    safeBigTerritories: boolean;
    unsafeBigTerritories: boolean;
    isMultiStoneSuicideLegal: boolean;
    result: Uint8Array;
  },
  context: BoardContext
): void {
  const { stones, numGroups, plaColor, safeBigTerritories, unsafeBigTerritories, isMultiStoneSuicideLegal, result } = args;
  const oppColor = opponentOf(plaColor);
  const geom = context.geometry;
  const scratch = context.scratch;
  const area = geom.area;
  const nStarts = geom.neighborStarts;
  const nCounts = geom.neighborCounts;
  const neighbors = geom.neighbors;

  const regionIdxByPos = scratch.regionIdxByPos;
  const bordersNonPassAlive = scratch.bordersNonPassAliveByHeadPos;
  const regionHeads = scratch.regionHeads;
  const vitalStart = scratch.vitalStart;
  const vitalLen = scratch.vitalLen;
  const numInternalSpaces = scratch.numInternalSpacesMax2;
  const containsOpp = scratch.containsOpp;
  const vitalList = scratch.vitalList;
  const nextEmptyOrOpp = scratch.nextEmptyOrOpp;
  const regionQueue = scratch.regionQueue;
  const plaGroups = scratch.plaGroups;
  const plaGroupKilled = scratch.plaGroupKilled;
  const vitalCountByGroup = scratch.vitalCountByGroup;
  const groupIndexByPos = scratch.groupIndexByPos;
  const groupColorByGroup = scratch.groupColorByGroup;
  const groupStartByGroup = scratch.groupStartByGroup;
  const groupLenByGroup = scratch.groupLenByGroup;
  const groupStonesFlat = scratch.groupStonesFlat;

  regionIdxByPos.fill(-1);
  bordersNonPassAlive.fill(0);

  let numRegions = 0;
  let vitalTotal = 0;
  let atLeastOnePla = false;

  const buildRegion = (initialPos: number, regionIdx: number): number => {
    let tailTarget = initialPos;

    let qh = 0;
    let qt = 1;
    regionQueue[0] = initialPos;
    regionIdxByPos[initialPos] = regionIdx;

    let hasVital = vitalLen[regionIdx]! > 0;

    while (qh !== qt) {
      const pos = regionQueue[qh++]!;

      if (hasVital && (isMultiStoneSuicideLegal || (stones[pos] as StoneColor) === EMPTY)) {
        const vStart = vitalStart[regionIdx]!;
        const oldLen = vitalLen[regionIdx]!;
        let newLen = 0;
        for (let i = 0; i < oldLen; i++) {
          const g = vitalList[vStart + i]!;
          if (isAdjacentToPlaGroup(stones, pos, plaColor, g, context)) {
            vitalList[vStart + newLen] = g;
            newLen++;
          }
        }
        vitalLen[regionIdx] = newLen;
        hasVital = newLen > 0;
      }

      if (numInternalSpaces[regionIdx]! < 2 && !isAdjacentToColor(stones, pos, plaColor, context)) {
        numInternalSpaces[regionIdx] = (numInternalSpaces[regionIdx]! + 1) as number;
      }

      if ((stones[pos] as StoneColor) === oppColor) containsOpp[regionIdx] = 1;

      nextEmptyOrOpp[pos] = tailTarget;
      tailTarget = pos;

      const nStart = nStarts[pos]!;
      const nCount = nCounts[pos]!;
      for (let i = 0; i < nCount; i++) {
        const n = neighbors[nStart + i]!;
        const c = stones[n] as StoneColor;
        if (c !== EMPTY && c !== oppColor) continue;
        if (regionIdxByPos[n] !== -1) continue;
        regionIdxByPos[n] = regionIdx;
        regionQueue[qt++] = n;
      }
    }

    return tailTarget;
  };

  for (let p = 0; p < area; p++) {
    if (regionIdxByPos[p] !== -1) continue;
    const c = stones[p] as StoneColor;
    if (c !== EMPTY) {
      if (c === plaColor) atLeastOnePla = true;
      continue;
    }

    const regionIdx = numRegions++;
    regionHeads[regionIdx] = p;
    vitalStart[regionIdx] = vitalTotal;
    vitalLen[regionIdx] = 0;
    numInternalSpaces[regionIdx] = 0;
    containsOpp[regionIdx] = 0;

    let initialVLen = 0;
    {
      const nStart = nStarts[p]!;
      const nCount = nCounts[p]!;
      for (let i = 0; i < nCount; i++) {
        const adj = neighbors[nStart + i]!;
        if ((stones[adj] as StoneColor) !== plaColor) continue;
        const g = groupIndexByPos[adj]!;
        let alreadyPresent = false;
        for (let j = 0; j < initialVLen; j++) {
          if (vitalList[vitalTotal + j] === g) {
            alreadyPresent = true;
            break;
          }
        }
        if (!alreadyPresent) {
          vitalList[vitalTotal + initialVLen] = g;
          initialVLen++;
          if (initialVLen >= 4) break;
        }
      }
    }
    vitalLen[regionIdx] = initialVLen;

    const tail = buildRegion(p, regionIdx);
    nextEmptyOrOpp[p] = tail;

    vitalTotal += vitalLen[regionIdx]!;
  }

  let numPlaGroups = 0;
  for (let g = 0; g < numGroups; g++) {
    if ((groupColorByGroup[g] as StoneColor) === plaColor) {
      plaGroups[numPlaGroups++] = g;
      plaGroupKilled[g] = 0;
      vitalCountByGroup[g] = 0;
    }
  }

  for (let i = 0; i < numRegions; i++) {
    const vStart = vitalStart[i]!;
    const vLen = vitalLen[i]!;
    for (let j = 0; j < vLen; j++) {
      const g = vitalList[vStart + j]!;
      vitalCountByGroup[g] = (vitalCountByGroup[g]! + 1) as number;
    }
  }

  while (true) {
    let killedAnything = false;

    for (let i = 0; i < numPlaGroups; i++) {
      const g = plaGroups[i]!;
      if (plaGroupKilled[g]) continue;
      if (vitalCountByGroup[g]! >= 2) continue;

      plaGroupKilled[g] = 1;
      killedAnything = true;

      const start = groupStartByGroup[g]!;
      const len = groupLenByGroup[g]!;
      for (let t = 0; t < len; t++) {
        const cur = groupStonesFlat[start + t]!;
        const nStart = nStarts[cur]!;
        const nCount = nCounts[cur]!;
        for (let k = 0; k < nCount; k++) {
          const adj = neighbors[nStart + k]!;
          const regionIdx = regionIdxByPos[adj]!;
          if (regionIdx < 0) continue;

          const headPos = regionHeads[regionIdx]!;
          if (bordersNonPassAlive[headPos]) continue;
          const ac = stones[adj] as StoneColor;
          if (ac !== EMPTY && ac !== oppColor) continue;

          bordersNonPassAlive[headPos] = 1;

          const vs = vitalStart[regionIdx]!;
          const vl = vitalLen[regionIdx]!;
          for (let u = 0; u < vl; u++) {
            const gg = vitalList[vs + u]!;
            vitalCountByGroup[gg] = (vitalCountByGroup[gg]! - 1) as number;
          }
        }
      }
    }

    if (!killedAnything) break;
  }

  for (let i = 0; i < numPlaGroups; i++) {
    const g = plaGroups[i]!;
    if (plaGroupKilled[g]) continue;
    const start = groupStartByGroup[g]!;
    const len = groupLenByGroup[g]!;
    for (let t = 0; t < len; t++) {
      result[groupStonesFlat[start + t]!] = plaColor;
    }
  }

  for (let i = 0; i < numRegions; i++) {
    const headPos = regionHeads[i]!;

    let shouldMark = numInternalSpaces[i]! <= 1 && !bordersNonPassAlive[headPos] && atLeastOnePla;
    shouldMark = shouldMark || (safeBigTerritories && !containsOpp[i] && !bordersNonPassAlive[headPos] && atLeastOnePla);

    if (shouldMark) {
      let cur = headPos;
      do {
        result[cur] = plaColor;
        cur = nextEmptyOrOpp[cur]!;
      } while (cur !== headPos);
    } else {
      const shouldMarkIfEmpty = unsafeBigTerritories && !containsOpp[i] && atLeastOnePla;
      if (shouldMarkIfEmpty) {
        let cur = headPos;
        do {
          if ((result[cur] as StoneColor) === EMPTY) result[cur] = plaColor;
          cur = nextEmptyOrOpp[cur]!;
        } while (cur !== headPos);
      }
    }
  }
}

export function computeAreaMapV7KataGo(
  stones: Uint8Array,
  isMultiStoneSuicideLegal = false,
  context: BoardContext = getDefaultBoardContext()
): Uint8Array {
  return computeAreaMapV7KataGoInto(stones, new Uint8Array(context.geometry.area), isMultiStoneSuicideLegal, context);
}

export function computeAreaMapV7KataGoInto(
  stones: Uint8Array,
  out: Uint8Array,
  isMultiStoneSuicideLegal = false,
  context: BoardContext = getDefaultBoardContext()
): Uint8Array {
  const area = context.geometry.area;
  if (out.length !== area) {
    throw new Error(`computeAreaMapV7KataGoInto: expected out length ${area}, got ${out.length}`);
  }
  out.fill(EMPTY);
  const numGroups = buildGroups(stones, context);

  calculateAreaForPla(
    {
      stones,
      numGroups,
      plaColor: BLACK,
      safeBigTerritories: true,
      unsafeBigTerritories: true,
      isMultiStoneSuicideLegal,
      result: out,
    },
    context
  );
  calculateAreaForPla(
    {
      stones,
      numGroups,
      plaColor: WHITE,
      safeBigTerritories: true,
      unsafeBigTerritories: true,
      isMultiStoneSuicideLegal,
      result: out,
    },
    context
  );

  for (let p = 0; p < area; p++) {
    if ((out[p] as StoneColor) === EMPTY) out[p] = stones[p]!;
  }

  return out;
}
