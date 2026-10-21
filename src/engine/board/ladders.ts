import {
  EMPTY,
  BLACK,
  WHITE,
  opponentOf,
  type StoneColor,
  type BoardContext,
  getDefaultBoardContext,
  type LadderSearchScratch,
} from './context.js';
import { collectGroupAndLiberties, findLibertiesIntoBuf, lastFindLibertiesGroupLen } from './liberties.js';
import type { SimPosition } from './moves.js';

export const LADDER_SEARCH_NODE_BUDGET = 25_000;

function isAdjacent(a: number, b: number, boardSize: number): boolean {
  if (a === b) return false;
  const d = a - b;
  if (d === 1 || d === -1) return ((a / boardSize) | 0) === ((b / boardSize) | 0);
  return d === boardSize || d === -boardSize;
}

function getNumImmediateLiberties(stones: Uint8Array, pos: number, context: BoardContext): number {
  let num = 0;
  const geom = context.geometry;
  const nStart = geom.neighborStarts[pos]!;
  const nCount = geom.neighborCounts[pos]!;
  const neighbors = geom.neighbors;
  for (let i = 0; i < nCount; i++) {
    const n = neighbors[nStart + i]!;
    if ((stones[n] as StoneColor) === EMPTY) num++;
  }
  return num;
}

function getNumLibertiesCapped(stones: Uint8Array, loc: number, cap: number, context: BoardContext): number {
  const c = stones[loc] as StoneColor;
  if (c === EMPTY) return 0;
  return collectGroupAndLiberties(stones, loc, c, cap, context).liberties;
}

function wouldBeKoCapture(stones: Uint8Array, loc: number, pla: StoneColor, context: BoardContext): boolean {
  if ((stones[loc] as StoneColor) !== EMPTY) return false;
  const opp = opponentOf(pla);
  let oppCapturableLoc = -1;
  const geom = context.geometry;
  const nStart = geom.neighborStarts[loc]!;
  const nCount = geom.neighborCounts[loc]!;
  const neighbors = geom.neighbors;

  for (let i = 0; i < nCount; i++) {
    const adj = neighbors[nStart + i]!;
    const c = stones[adj] as StoneColor;
    if (c !== opp) return false;
    if (getNumLibertiesCapped(stones, adj, 2, context) === 1) {
      if (oppCapturableLoc !== -1) return false;
      oppCapturableLoc = adj;
    }
  }

  if (oppCapturableLoc === -1) return false;

  const oppGroup = collectGroupAndLiberties(stones, oppCapturableLoc, opp, 2, context);
  if (oppGroup.liberties !== 1) return false;
  return oppGroup.groupLen === 1;
}

function hasLibertyGainingCaptures(stones: Uint8Array, loc: number, context: BoardContext): boolean {
  const pla = stones[loc] as StoneColor;
  if (pla === EMPTY) return false;
  const opp = opponentOf(pla);
  const geom = context.geometry;
  const scratch = context.scratch;
  const groupBuf = scratch.groupBuf;
  const ladderGroupCopy = scratch.ladderGroupCopy;
  const nStarts = geom.neighborStarts;
  const nCounts = geom.neighborCounts;
  const neighbors = geom.neighbors;

  const g = collectGroupAndLiberties(stones, loc, pla, 2, context);
  for (let i = 0; i < g.groupLen; i++) ladderGroupCopy[i] = groupBuf[i]!;

  for (let i = 0; i < g.groupLen; i++) {
    const p = ladderGroupCopy[i]!;
    const nStart = nStarts[p]!;
    const nCount = nCounts[p]!;
    for (let j = 0; j < nCount; j++) {
      const adj = neighbors[nStart + j]!;
      if ((stones[adj] as StoneColor) !== opp) continue;
      if (getNumLibertiesCapped(stones, adj, 2, context) === 1) return true;
    }
  }
  return false;
}

function countHeuristicConnectionLibertiesX2(stones: Uint8Array, loc: number, pla: StoneColor, context: BoardContext): number {
  let numLibsX2 = 0;
  const geom = context.geometry;
  const nStart = geom.neighborStarts[loc]!;
  const nCount = geom.neighborCounts[loc]!;
  const neighbors = geom.neighbors;
  for (let i = 0; i < nCount; i++) {
    const adj = neighbors[nStart + i]!;
    if ((stones[adj] as StoneColor) !== pla) continue;
    const libs = getNumLibertiesCapped(stones, adj, 20, context);
    if (libs > 1) numLibsX2 += libs * 2 - 3;
  }
  return numLibsX2;
}

function getBoundNumLibertiesAfterPlay(
  stones: Uint8Array,
  loc: number,
  pla: StoneColor,
  context: BoardContext
): { lowerBound: number; upperBound: number } {
  const opp = opponentOf(pla);
  let numImmediateLibs = 0;
  let numCaps = 0;
  let potentialLibsFromCaps = 0;
  let numConnectionLibs = 0;
  let maxConnectionLibs = 0;

  const geom = context.geometry;
  const nStart = geom.neighborStarts[loc]!;
  const nCount = geom.neighborCounts[loc]!;
  const neighbors = geom.neighbors;

  for (let i = 0; i < nCount; i++) {
    const adj = neighbors[nStart + i]!;
    const c = stones[adj] as StoneColor;
    if (c === EMPTY) {
      numImmediateLibs++;
    } else if (c === opp) {
      const capInfo = collectGroupAndLiberties(stones, adj, opp, 2, context);
      if (capInfo.liberties === 1) {
        numCaps++;
        potentialLibsFromCaps += capInfo.groupLen;
      }
    } else if (c === pla) {
      const libs = getNumLibertiesCapped(stones, adj, 20, context);
      const connLibs = libs - 1;
      numConnectionLibs += connLibs;
      if (connLibs > maxConnectionLibs) maxConnectionLibs = connLibs;
    }
  }

  const lowerBound = numCaps + (maxConnectionLibs > numImmediateLibs ? maxConnectionLibs : numImmediateLibs);
  const upperBound = numImmediateLibs + potentialLibsFromCaps + numConnectionLibs;
  return { lowerBound, upperBound };
}

function getNumLibertiesAfterPlay(stones: Uint8Array, loc: number, pla: StoneColor, max: number, context: BoardContext): number {
  if ((stones[loc] as StoneColor) !== EMPTY) return 0;
  const opp = opponentOf(pla);
  const geom = context.geometry;
  const scratch = context.scratch;
  const ladderScratch = scratch.ladderScratch;
  const groupBuf = scratch.groupBuf;
  const ladderCaptured = scratch.ladderCaptured;
  const ladderConnectGroupSeen = scratch.ladderConnectGroupSeen;

  const capStamp = context.nextLadderCapturedStamp();

  // Collect initial liberties.
  const libs = ladderScratch.tempLibs16;
  let numLibs = 0;

  const addLib = (p: number): boolean => {
    for (let i = 0; i < numLibs; i++) if (libs[i] === p) return false;
    libs[numLibs++] = p;
    return true;
  };

  const nStarts = geom.neighborStarts;
  const nCounts = geom.neighborCounts;
  const neighbors = geom.neighbors;

  const nStart = nStarts[loc]!;
  const nCount = nCounts[loc]!;
  for (let i = 0; i < nCount; i++) {
    const adj = neighbors[nStart + i]!;
    const c = stones[adj] as StoneColor;
    if (c === EMPTY) {
      addLib(adj);
      if (numLibs >= max) return max;
    } else if (c === opp) {
      const oppInfo = collectGroupAndLiberties(stones, adj, opp, 2, context);
      if (oppInfo.liberties === 1) {
        addLib(adj);
        if (numLibs >= max) return max;
        for (let j = 0; j < oppInfo.groupLen; j++) ladderCaptured[groupBuf[j]!] = capStamp;
      }
    }
  }

  const seenStamp = context.nextLadderConnectGroupSeenStamp();

  const wouldBeEmpty = (p: number): boolean => {
    const c = stones[p] as StoneColor;
    if (c === EMPTY) return true;
    return c === opp && ladderCaptured[p] === capStamp;
  };

  for (let i = 0; i < nCount; i++) {
    const adj = neighbors[nStart + i]!;
    if ((stones[adj] as StoneColor) !== pla) continue;
    if (ladderConnectGroupSeen[adj] === seenStamp) continue;

    const g = collectGroupAndLiberties(stones, adj, pla, max, context);
    for (let j = 0; j < g.groupLen; j++) ladderConnectGroupSeen[groupBuf[j]!] = seenStamp;

    for (let j = 0; j < g.groupLen; j++) {
      const cur = groupBuf[j]!;
      const cs = nStarts[cur]!;
      const cc = nCounts[cur]!;
      for (let k = 0; k < cc; k++) {
        const possibleLib = neighbors[cs + k]!;
        if (possibleLib === loc) continue;
        if (!wouldBeEmpty(possibleLib)) continue;
        addLib(possibleLib);
        if (numLibs >= max) return max;
      }
    }
  }

  return numLibs;
}

function findLibertyGainingCaptures(
  stones: Uint8Array,
  loc: number,
  buf: Int16Array,
  bufStart: number,
  context: BoardContext
): number {
  const pla = stones[loc] as StoneColor;
  if (pla === EMPTY) return 0;
  const opp = opponentOf(pla);
  const geom = context.geometry;
  const scratch = context.scratch;
  const groupBuf = scratch.groupBuf;
  const ladderGroupCopy = scratch.ladderGroupCopy;
  const ladderOppGroupSeen = scratch.ladderOppGroupSeen;

  const seenStamp = context.nextLadderOppGroupSeenStamp();

  const g = collectGroupAndLiberties(stones, loc, pla, 2, context);
  for (let i = 0; i < g.groupLen; i++) ladderGroupCopy[i] = groupBuf[i]!;

  let numFound = 0;
  for (let i = 0; i < g.groupLen; i++) {
    const p = ladderGroupCopy[i]!;
    const nStart = geom.neighborStarts[p]!;
    const nCount = geom.neighborCounts[p]!;
    for (let j = 0; j < nCount; j++) {
      const adj = geom.neighbors[nStart + j]!;
      if ((stones[adj] as StoneColor) !== opp) continue;
      if (ladderOppGroupSeen[adj] === seenStamp) continue;

      const libCount = findLibertiesIntoBuf(stones, adj, opp, buf, bufStart + numFound, 2, context);
      const groupLen = lastFindLibertiesGroupLen;
      for (let k = 0; k < groupLen; k++) ladderOppGroupSeen[groupBuf[k]!] = seenStamp;

      if (libCount === 1) {
        numFound += 1;
        if (bufStart + numFound >= buf.length) return numFound;
      }
    }
  }
  return numFound;
}

function undoMoveRaw(
  pos: SimPosition,
  move: number,
  player: StoneColor,
  koPointBefore: number,
  captureStart: number,
  captureStack: number[],
  passMove: number
): void {
  const captureEnd = captureStack.length;
  const opp = opponentOf(player);

  if (move !== passMove) pos.stones[move] = EMPTY;

  for (let i = captureStart; i < captureEnd; i++) {
    const p = captureStack[i]!;
    pos.stones[p] = opp;
  }
  captureStack.length = captureStart;
  pos.koPoint = koPointBefore;
}

function tryPlayMoveNoThrow(
  pos: SimPosition,
  move: number,
  player: StoneColor,
  captureStack: number[],
  koPointBeforeOut: Int16Array,
  captureStartOut: Int32Array,
  recordIdx: number,
  context: BoardContext
): boolean {
  const geom = context.geometry;
  const scratch = context.scratch;
  const area = geom.area;
  const passMove = geom.passMove;

  const koPointBefore = pos.koPoint;
  const captureStart = captureStack.length;
  koPointBeforeOut[recordIdx] = koPointBefore;
  captureStartOut[recordIdx] = captureStart;

  if (move === passMove) {
    pos.koPoint = -1;
    return true;
  }

  if (move < 0 || move >= area) return false;
  if ((pos.stones[move] as StoneColor) !== EMPTY) return false;
  if (pos.koPoint === move) return false;

  const opp = opponentOf(player);
  pos.stones[move] = player;

  let totalCaptured = 0;
  let capturedSinglePos = -1;

  const pStamp = context.nextProcessedStamp();
  const processedGroup = scratch.processedGroup;
  const groupBuf = scratch.groupBuf;
  const nStarts = geom.neighborStarts;
  const nCounts = geom.neighborCounts;
  const neighbors = geom.neighbors;

  const mStart = nStarts[move]!;
  const mCount = nCounts[move]!;
  for (let i = 0; i < mCount; i++) {
    const n = neighbors[mStart + i]!;
    if ((pos.stones[n] as StoneColor) !== opp) continue;
    if (processedGroup[n] === pStamp) continue;

    const { groupLen, liberties } = collectGroupAndLiberties(pos.stones, n, opp, 1, context);
    for (let j = 0; j < groupLen; j++) processedGroup[groupBuf[j]!] = pStamp;
    if (liberties !== 0) continue;

    for (let j = 0; j < groupLen; j++) {
      const gp = groupBuf[j]!;
      pos.stones[gp] = EMPTY;
      captureStack.push(gp);
    }

    totalCaptured += groupLen;
    if (totalCaptured === 1 && groupLen === 1) capturedSinglePos = groupBuf[0]!;
    if (totalCaptured > 1) capturedSinglePos = -1;
  }

  const selfGroup = collectGroupAndLiberties(pos.stones, move, player, 2, context);
  if (selfGroup.liberties === 0) {
    undoMoveRaw(pos, move, player, koPointBefore, captureStart, captureStack, passMove);
    return false;
  }

  if (totalCaptured === 1 && capturedSinglePos >= 0 && selfGroup.groupLen === 1 && selfGroup.liberties === 1) {
    pos.koPoint = capturedSinglePos;
  } else {
    pos.koPoint = -1;
  }

  return true;
}

export function searchIsLadderCaptured(
  pos: SimPosition,
  loc: number,
  defenderFirst: boolean,
  scratch: LadderSearchScratch,
  context: BoardContext = getDefaultBoardContext()
): boolean {
  const geom = context.geometry;
  const area = geom.area;
  const size = geom.size;
  const passMove = geom.passMove;

  if (loc < 0 || loc >= area) return false;
  const c = pos.stones[loc] as StoneColor;
  if (c !== BLACK && c !== WHITE) return false;

  const libs0 = getNumLibertiesCapped(pos.stones, loc, 3, context);
  if (libs0 > 2 || (defenderFirst && libs0 > 1)) return false;

  const pla = c;
  const opp = opponentOf(pla);

  const koSaved = pos.koPoint;
  if (defenderFirst) pos.koPoint = -1;

  const {
    bufMoves,
    moveListStarts,
    moveListLens,
    moveListCur,
    recordMoves,
    recordPlayers,
    recordKoPointBefore,
    recordCaptureStart,
    captureStack,
  } = scratch;

  const stackSize = ((area * 3) / 2 + 2) | 0;
  let stackIdx = 0;
  let searchNodeCount = 0;

  moveListCur[0] = -1;
  moveListStarts[0] = 0;
  moveListLens[0] = 0;

  let returnValue = false;
  let returnedFromDeeper = false;

  while (true) {
    if (stackIdx <= -1) {
      pos.koPoint = koSaved;
      return returnValue;
    }

    if (stackIdx >= stackSize - 1) {
      returnValue = true;
      returnedFromDeeper = true;
      stackIdx--;
      continue;
    }

    if (searchNodeCount >= LADDER_SEARCH_NODE_BUDGET) {
      for (let i = stackIdx - 1; i >= 0; i--) {
        undoMoveRaw(
          pos,
          recordMoves[i]!,
          recordPlayers[i]! as StoneColor,
          recordKoPointBefore[i]!,
          recordCaptureStart[i]!,
          captureStack,
          passMove
        );
      }
      pos.koPoint = koSaved;
      return false;
    }

    const isDefender = (defenderFirst && (stackIdx % 2) === 0) || (!defenderFirst && (stackIdx % 2) === 1);

    if (moveListCur[stackIdx] === -1) {
      const libs = getNumLibertiesCapped(pos.stones, loc, 3, context);

      if (!isDefender && libs <= 1) {
        returnValue = true;
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }
      if (!isDefender && libs >= 3) {
        returnValue = false;
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }
      if (isDefender && libs >= 2) {
        returnValue = false;
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }
      if (isDefender && pos.koPoint !== -1) {
        returnValue = false;
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }

      const start = moveListStarts[stackIdx]!;
      let moveListLen = 0;

      if (isDefender) {
        moveListLen = findLibertyGainingCaptures(pos.stones, loc, bufMoves, start, context);
        moveListLen += findLibertiesIntoBuf(pos.stones, loc, pla, bufMoves, start + moveListLen, 1, context);

        if (moveListLen <= 0) {
          returnValue = true;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }

        const lastMove = bufMoves[start + moveListLen - 1]!;
        const bounds = getBoundNumLibertiesAfterPlay(pos.stones, lastMove, pla, context);
        if (bounds.lowerBound >= 3) {
          returnValue = false;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
        if (moveListLen === 1 && bounds.upperBound <= 1) {
          returnValue = true;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
      } else {
        moveListLen += findLibertiesIntoBuf(pos.stones, loc, pla, bufMoves, start, 2, context);
        if (moveListLen !== 2) {
          returnValue = false;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }

        let libsA = getNumImmediateLiberties(pos.stones, bufMoves[start]!, context);
        let libsB = getNumImmediateLiberties(pos.stones, bufMoves[start + 1]!, context);

        if (
          libsA === 0 &&
          libsB === 0 &&
          wouldBeKoCapture(pos.stones, bufMoves[start]!, opp, context) &&
          wouldBeKoCapture(pos.stones, bufMoves[start + 1]!, opp, context)
        ) {
          if (
            getNumLibertiesAfterPlay(pos.stones, bufMoves[start]!, pla, 3, context) <= 2 &&
            getNumLibertiesAfterPlay(pos.stones, bufMoves[start + 1]!, pla, 3, context) <= 2
          ) {
            if (!hasLibertyGainingCaptures(pos.stones, loc, context)) {
              returnValue = true;
              returnedFromDeeper = true;
              stackIdx--;
              continue;
            }
          }
        }

        if (!isAdjacent(bufMoves[start]!, bufMoves[start + 1]!, size)) {
          if (libsA >= 3 && libsB >= 3) {
            returnValue = false;
            returnedFromDeeper = true;
            stackIdx--;
            continue;
          } else if (libsA >= 3) {
            moveListLen = 1;
          } else if (libsB >= 3) {
            bufMoves[start] = bufMoves[start + 1]!;
            moveListLen = 1;
          }
        }

        if (moveListLen > 1) {
          libsA = libsA * 2 + countHeuristicConnectionLibertiesX2(pos.stones, bufMoves[start]!, pla, context);
          libsB = libsB * 2 + countHeuristicConnectionLibertiesX2(pos.stones, bufMoves[start + 1]!, pla, context);
          if (libsB > libsA) {
            const tmp = bufMoves[start]!;
            bufMoves[start] = bufMoves[start + 1]!;
            bufMoves[start + 1] = tmp;
          }
        }
      }

      moveListLens[stackIdx] = moveListLen;
      moveListCur[stackIdx] = 0;
    } else {
      if (returnedFromDeeper) {
        undoMoveRaw(
          pos,
          recordMoves[stackIdx]!,
          recordPlayers[stackIdx]! as StoneColor,
          recordKoPointBefore[stackIdx]!,
          recordCaptureStart[stackIdx]!,
          captureStack,
          passMove
        );
      }

      if (isDefender && !returnValue) {
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }
      if (!isDefender && returnValue) {
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }

      moveListCur[stackIdx] = (moveListCur[stackIdx]! + 1) | 0;
    }

    if (moveListCur[stackIdx]! >= moveListLens[stackIdx]!) {
      returnValue = isDefender;
      returnedFromDeeper = true;
      stackIdx--;
      continue;
    }

    const move = bufMoves[moveListStarts[stackIdx]! + moveListCur[stackIdx]!]!;
    const p = isDefender ? pla : opp;

    recordMoves[stackIdx] = move;
    recordPlayers[stackIdx] = p;

    if (!tryPlayMoveNoThrow(pos, move, p, captureStack, recordKoPointBefore, recordCaptureStart, stackIdx, context)) {
      returnValue = isDefender;
      returnedFromDeeper = false;
      continue;
    }

    searchNodeCount++;

    stackIdx++;
    moveListCur[stackIdx] = -1;
    moveListStarts[stackIdx] = moveListStarts[stackIdx - 1]! + moveListLens[stackIdx - 1]!;
    moveListLens[stackIdx] = 0;
  }
}

export function searchIsLadderCapturedAttackerFirst2Libs(
  pos: SimPosition,
  loc: number,
  scratch: LadderSearchScratch,
  outWorkingMoves: number[],
  context: BoardContext = getDefaultBoardContext()
): boolean {
  const geom = context.geometry;
  const area = geom.area;
  const passMove = geom.passMove;

  if (loc < 0 || loc >= area) return false;
  const c = pos.stones[loc] as StoneColor;
  if (c !== BLACK && c !== WHITE) return false;

  if (getNumLibertiesCapped(pos.stones, loc, 3, context) !== 2) return false;

  const pla = c;
  const opp = opponentOf(pla);

  const tmpLibs = scratch.attackerLibs2;
  const libCount = findLibertiesIntoBuf(pos.stones, loc, pla, tmpLibs, 0, 2, context);
  if (libCount !== 2) return false;
  const move0 = tmpLibs[0]!;
  const move1 = tmpLibs[1]!;

  let move0Works = false;
  let move1Works = false;

  outWorkingMoves.length = 0;
  scratch.captureStack.length = 0;

  if (tryPlayMoveNoThrow(pos, move0, opp, scratch.captureStack, scratch.tmpKoPointBefore, scratch.tmpCaptureStart, 0, context)) {
    move0Works = searchIsLadderCaptured(pos, loc, true, scratch, context);
    undoMoveRaw(pos, move0, opp, scratch.tmpKoPointBefore[0]!, scratch.tmpCaptureStart[0]!, scratch.captureStack, passMove);
  }

  if (tryPlayMoveNoThrow(pos, move1, opp, scratch.captureStack, scratch.tmpKoPointBefore, scratch.tmpCaptureStart, 0, context)) {
    move1Works = searchIsLadderCaptured(pos, loc, true, scratch, context);
    undoMoveRaw(pos, move1, opp, scratch.tmpKoPointBefore[0]!, scratch.tmpCaptureStart[0]!, scratch.captureStack, passMove);
  }

  if (move0Works || move1Works) {
    if (move0Works) outWorkingMoves.push(move0);
    if (move1Works) outWorkingMoves.push(move1);
    return true;
  }
  return false;
}

export type KataGoLadderFeaturesV7 = {
  ladderedStones: Uint8Array; // plane 14
  ladderWorkingMoves: Uint8Array; // plane 17
};

export function computeLadderFeaturesV7KataGoInto(
  args: {
    stones: Uint8Array;
    koPoint: number;
    currentPlayer: StoneColor;
    outLadderedStones: Uint8Array;
    outLadderWorkingMoves: Uint8Array;
  },
  context: BoardContext = getDefaultBoardContext()
): void {
  const { stones, koPoint, currentPlayer, outLadderedStones, outLadderWorkingMoves } = args;
  const geom = context.geometry;
  const scratch = context.scratch;
  const area = geom.area;

  outLadderedStones.fill(0);
  outLadderWorkingMoves.fill(0);

  const opp = opponentOf(currentPlayer);

  const seenStamp = context.nextLadderGroupSeenStamp();
  const ladderGroupSeen = scratch.ladderGroupSeen;
  const ladderFeaturesScratch = scratch.ladderFeaturesScratch;
  const copyPos = ladderFeaturesScratch.copyPos;
  const groupStones = ladderFeaturesScratch.groupStones;
  const workingMoves = ladderFeaturesScratch.workingMoves;
  const groupBuf = scratch.groupBuf;
  const ladderScratch = scratch.ladderScratch;

  for (let p = 0; p < area; p++) {
    const c = stones[p] as StoneColor;
    if (c === EMPTY) continue;
    if (ladderGroupSeen[p] === seenStamp) continue;

    const g = collectGroupAndLiberties(stones, p, c, 3, context);
    for (let i = 0; i < g.groupLen; i++) ladderGroupSeen[groupBuf[i]!] = seenStamp;

    if (g.liberties !== 1 && g.liberties !== 2) continue;

    groupStones.set(groupBuf.subarray(0, g.groupLen));

    copyPos.stones.set(stones);
    copyPos.koPoint = koPoint;
    ladderScratch.captureStack.length = 0;

    let laddered = false;
    workingMoves.length = 0;
    if (g.liberties === 1) {
      laddered = searchIsLadderCaptured(copyPos, p, true, ladderScratch, context);
    } else {
      laddered = searchIsLadderCapturedAttackerFirst2Libs(copyPos, p, ladderScratch, workingMoves, context);
    }

    if (!laddered) continue;

    for (let i = 0; i < g.groupLen; i++) outLadderedStones[groupStones[i]!] = 1;
    if (g.liberties === 2 && c === opp && workingMoves.length > 0) {
      for (let i = 0; i < workingMoves.length; i++) outLadderWorkingMoves[workingMoves[i]!] = 1;
    }
  }
}

export function computeLadderFeaturesV7KataGo(
  args: { stones: Uint8Array; koPoint: number; currentPlayer: StoneColor },
  context: BoardContext = getDefaultBoardContext()
): KataGoLadderFeaturesV7 {
  const area = context.geometry.area;
  const ladderedStones = new Uint8Array(area);
  const ladderWorkingMoves = new Uint8Array(area);
  computeLadderFeaturesV7KataGoInto(
    { ...args, outLadderedStones: ladderedStones, outLadderWorkingMoves: ladderWorkingMoves },
    context
  );

  return { ladderedStones, ladderWorkingMoves };
}

export function computeLadderedStonesV7KataGoInto(
  args: { stones: Uint8Array; koPoint: number; outLadderedStones: Uint8Array },
  context: BoardContext = getDefaultBoardContext()
): void {
  const { stones, koPoint, outLadderedStones } = args;
  const geom = context.geometry;
  const scratch = context.scratch;
  const area = geom.area;

  outLadderedStones.fill(0);

  const seenStamp = context.nextLadderGroupSeenStamp();
  const ladderGroupSeen = scratch.ladderGroupSeen;
  const ladderFeaturesScratch = scratch.ladderFeaturesScratch;
  const copyPos = ladderFeaturesScratch.copyPos;
  const groupStones = ladderFeaturesScratch.groupStones;
  const workingMoves = ladderFeaturesScratch.workingMoves;
  const groupBuf = scratch.groupBuf;
  const ladderScratch = scratch.ladderScratch;

  for (let p = 0; p < area; p++) {
    const c = stones[p] as StoneColor;
    if (c === EMPTY) continue;
    if (ladderGroupSeen[p] === seenStamp) continue;

    const g = collectGroupAndLiberties(stones, p, c, 3, context);
    for (let i = 0; i < g.groupLen; i++) ladderGroupSeen[groupBuf[i]!] = seenStamp;

    if (g.liberties !== 1 && g.liberties !== 2) continue;

    groupStones.set(groupBuf.subarray(0, g.groupLen));

    copyPos.stones.set(stones);
    copyPos.koPoint = koPoint;
    ladderScratch.captureStack.length = 0;

    workingMoves.length = 0;
    let ladderedGroup = false;
    if (g.liberties === 1) ladderedGroup = searchIsLadderCaptured(copyPos, p, true, ladderScratch, context);
    else ladderedGroup = searchIsLadderCapturedAttackerFirst2Libs(copyPos, p, ladderScratch, workingMoves, context);

    if (!ladderedGroup) continue;
    for (let i = 0; i < g.groupLen; i++) outLadderedStones[groupStones[i]!] = 1;
  }
}

export function computeLadderedStonesV7KataGo(
  args: { stones: Uint8Array; koPoint: number },
  context: BoardContext = getDefaultBoardContext()
): Uint8Array {
  const laddered = new Uint8Array(context.geometry.area);
  computeLadderedStonesV7KataGoInto({ ...args, outLadderedStones: laddered }, context);
  return laddered;
}
