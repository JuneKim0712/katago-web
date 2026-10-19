import type { BoardState, Move, Player } from '../types/index.js';
import {
  BLACK,
  WHITE,
  playMove,
  type SimPosition,
  type StoneColor,
} from '../engine/fastBoard.js';

function playerToColor(p: Player): StoneColor {
  return p === 'black' ? BLACK : WHITE;
}

export function boardGridToStones(board: BoardState, boardSize: number): Uint8Array {
  const stones = new Uint8Array(boardSize * boardSize);
  for (let y = 0; y < boardSize; y++) {
    const row = board[y];
    for (let x = 0; x < boardSize; x++) {
      const v = row?.[x] ?? null;
      if (!v) continue;
      stones[y * boardSize + x] = v === 'black' ? BLACK : WHITE;
    }
  }
  return stones;
}

export function computeKoPointAfterMoveOnStones(
  stonesBefore: Uint8Array,
  koPointBefore: number,
  move: Move | null,
  boardSize: number,
  dest?: SimPosition,
  captureStack?: number[]
): number {
  if (!move) return -1;
  if (move.x < 0 || move.y < 0) return -1;

  const stack = captureStack ?? [];
  stack.length = 0;

  const pos: SimPosition = dest ?? { stones: stonesBefore.slice(), koPoint: koPointBefore };
  if (dest) {
    dest.stones.set(stonesBefore);
    dest.koPoint = koPointBefore;
  }

  try {
    playMove(pos, move.y * boardSize + move.x, playerToColor(move.player), stack);
    return pos.koPoint;
  } catch {
    return -1;
  }
}

export function computeKoPointAfterMoveOnBoard(
  boardBefore: BoardState,
  koPointBefore: number,
  move: Move | null,
  boardSize?: number
): number {
  const size = boardSize ?? boardBefore.length;
  const stones = boardGridToStones(boardBefore, size);
  return computeKoPointAfterMoveOnStones(stones, koPointBefore, move, size);
}

export function resolveRootKoPoint(args: {
  explicitKoPoint?: number;
  previousBoard?: BoardState;
  previousPreviousBoard?: BoardState;
  moveHistory: Move[];
}): number {
  if (typeof args.explicitKoPoint === 'number' && args.explicitKoPoint >= 0) {
    return args.explicitKoPoint;
  }

  const boardSize = args.previousBoard?.length ?? 19;
  if (!args.previousBoard || args.moveHistory.length === 0) return -1;

  const lastMove = args.moveHistory[args.moveHistory.length - 1]!;
  const prevMove = args.moveHistory.length >= 2 ? args.moveHistory[args.moveHistory.length - 2]! : null;
  const stones = boardGridToStones(args.previousBoard, boardSize);
  const koBeforeLast =
    args.previousPreviousBoard && prevMove
      ? computeKoPointAfterMoveOnBoard(args.previousPreviousBoard, -1, prevMove, boardSize)
      : -1;

  return computeKoPointAfterMoveOnStones(stones, koBeforeLast, lastMove, boardSize);
}

export function resolvePrevKoPoint(args: {
  previousPreviousBoard?: BoardState;
  previousPreviousPreviousBoard?: BoardState;
  moveHistory: Move[];
}): number {
  if (args.moveHistory.length < 2) return -1;
  return resolveRootKoPoint({
    previousBoard: args.previousPreviousBoard,
    previousPreviousBoard: args.previousPreviousPreviousBoard,
    moveHistory: args.moveHistory.slice(0, -1),
  });
}

export function resolvePrevPrevKoPoint(args: {
  previousPreviousPreviousBoard?: BoardState;
  moveHistory: Move[];
}): number {
  if (args.moveHistory.length < 3 || !args.previousPreviousPreviousBoard) return -1;
  return resolveRootKoPoint({
    previousBoard: args.previousPreviousPreviousBoard,
    moveHistory: args.moveHistory.slice(0, -2),
  });
}

export function resolveKoState(args: {
  koPoint?: number;
  previousBoard?: BoardState;
  previousPreviousBoard?: BoardState;
  previousPreviousPreviousBoard?: BoardState;
  moveHistory: Move[];
}): { rootKoPoint: number; rootPrevKoPoint: number; rootPrevPrevKoPoint: number } {
  return {
    rootKoPoint: resolveRootKoPoint({
      explicitKoPoint: args.koPoint,
      previousBoard: args.previousBoard,
      previousPreviousBoard: args.previousPreviousBoard,
      moveHistory: args.moveHistory,
    }),
    rootPrevKoPoint: resolvePrevKoPoint({
      previousPreviousBoard: args.previousPreviousBoard,
      previousPreviousPreviousBoard: args.previousPreviousPreviousBoard,
      moveHistory: args.moveHistory,
    }),
    rootPrevPrevKoPoint: resolvePrevPrevKoPoint({
      previousPreviousPreviousBoard: args.previousPreviousPreviousBoard,
      moveHistory: args.moveHistory,
    }),
  };
}
