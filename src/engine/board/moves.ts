import {
  EMPTY,
  opponentOf,
  type StoneColor,
  type BoardContext,
  getDefaultBoardContext,
} from './context.js';
import { collectGroupAndLiberties } from './liberties.js';

export type UndoSnapshot = {
  koPointBefore: number;
  captureStart: number;
};

export type SimPosition = {
  readonly stones: Uint8Array;
  koPoint: number;
};

export function playMoveInto(
  pos: SimPosition,
  move: number,
  player: StoneColor,
  captureStack: number[],
  outSnapshots: UndoSnapshot[],
  depth: number,
  context: BoardContext = getDefaultBoardContext()
): void {
  const geom = context.geometry;
  const scratch = context.scratch;
  const passMove = geom.passMove;
  const area = geom.area;
  const koPointBefore = pos.koPoint;
  const captureStart = captureStack.length;

  if (move === passMove) {
    pos.koPoint = -1;
    let snap = outSnapshots[depth];
    if (!snap) outSnapshots[depth] = { koPointBefore, captureStart };
    else {
      snap.koPointBefore = koPointBefore;
      snap.captureStart = captureStart;
    }
    return;
  }

  if (move < 0 || move >= area) throw new Error(`Invalid move index ${move}`);
  if (pos.stones[move] !== EMPTY) throw new Error('Move on occupied point');
  if (pos.koPoint === move) throw new Error('Move violates simple ko');

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
    throw new Error('Illegal suicide move');
  }

  if (totalCaptured === 1 && capturedSinglePos >= 0 && selfGroup.groupLen === 1 && selfGroup.liberties === 1) {
    pos.koPoint = capturedSinglePos;
  } else {
    pos.koPoint = -1;
  }

  let snap = outSnapshots[depth];
  if (!snap) outSnapshots[depth] = { koPointBefore, captureStart };
  else {
    snap.koPointBefore = koPointBefore;
    snap.captureStart = captureStart;
  }
}

export function playMove(
  pos: SimPosition,
  move: number,
  player: StoneColor,
  captureStack: number[],
  context: BoardContext = getDefaultBoardContext()
): UndoSnapshot {
  const geom = context.geometry;
  const scratch = context.scratch;
  const passMove = geom.passMove;
  const area = geom.area;
  const koPointBefore = pos.koPoint;
  const captureStart = captureStack.length;

  if (move === passMove) {
    pos.koPoint = -1;
    return { koPointBefore, captureStart };
  }

  if (move < 0 || move >= area) throw new Error(`Invalid move index ${move}`);
  if (pos.stones[move] !== EMPTY) throw new Error('Move on occupied point');
  if (pos.koPoint === move) throw new Error('Move violates simple ko');

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
    throw new Error('Illegal suicide move');
  }

  if (totalCaptured === 1 && capturedSinglePos >= 0 && selfGroup.groupLen === 1 && selfGroup.liberties === 1) {
    pos.koPoint = capturedSinglePos;
  } else {
    pos.koPoint = -1;
  }

  return { koPointBefore, captureStart };
}

export function undoMove(
  pos: SimPosition,
  move: number,
  player: StoneColor,
  snapshot: UndoSnapshot,
  captureStack: number[],
  context: BoardContext = getDefaultBoardContext()
): void {
  const passMove = context.geometry.passMove;
  const captureEnd = captureStack.length;
  const opp = opponentOf(player);

  if (move !== passMove) {
    pos.stones[move] = EMPTY;
  }

  for (let i = snapshot.captureStart; i < captureEnd; i++) {
    const p = captureStack[i]!;
    pos.stones[p] = opp;
  }
  captureStack.length = snapshot.captureStart;
  pos.koPoint = snapshot.koPointBefore;
}
