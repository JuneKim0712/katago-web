/**
 * Canonical position and move representation matching upstream KataGo conventions.
 */

import type { Rules } from './rules.js';

export type Player = 'black' | 'white';

export const EMPTY = 0;
export const BLACK = 1;
export const WHITE = 2;
export type Color = typeof EMPTY | typeof BLACK | typeof WHITE;

export const PASS_LOC = -1;
export const DEFAULT_BOARD_SIZE = 19;

export interface Move {
  readonly loc: number; // 0 .. width*height-1, or PASS_LOC (-1)
  readonly x: number; // 0 .. width-1, or -1 for pass
  readonly y: number; // 0 .. height-1, or -1 for pass
  readonly player: Player;
  readonly isPass: boolean;
}

export interface StonePlacement {
  readonly x: number;
  readonly y: number;
  readonly color: Player;
}

export interface CanonicalPosition {
  readonly width: number;
  readonly height: number;
  readonly initialStones: readonly StonePlacement[];
  readonly initialPlayer: Player;
  readonly moveHistory: readonly Move[];
  readonly currentPlayer: Player;
  readonly komi: number;
  readonly rules: Rules;
  readonly whiteHandicapBonus: number;
}

export function opponent(p: Player): Player {
  return p === 'black' ? 'white' : 'black';
}

export function playerToColor(p: Player): Color {
  return p === 'black' ? BLACK : WHITE;
}

export function colorToPlayer(c: Color): Player | null {
  if (c === BLACK) return 'black';
  if (c === WHITE) return 'white';
  return null;
}

export function locToX(loc: number, width = DEFAULT_BOARD_SIZE): number {
  if (loc < 0) return -1;
  return loc % width;
}

export function locToY(loc: number, width = DEFAULT_BOARD_SIZE): number {
  if (loc < 0) return -1;
  return Math.floor(loc / width);
}

export function xyToLoc(x: number, y: number, width = DEFAULT_BOARD_SIZE): number {
  if (x < 0 || y < 0) return PASS_LOC;
  return y * width + x;
}

export function createMove(player: Player, x: number, y: number, width = DEFAULT_BOARD_SIZE): Move {
  if (x < 0 || y < 0) {
    return createPassMove(player);
  }
  const loc = xyToLoc(x, y, width);
  return Object.freeze({
    loc,
    x,
    y,
    player,
    isPass: false,
  });
}

export function createPassMove(player: Player): Move {
  return Object.freeze({
    loc: PASS_LOC,
    x: -1,
    y: -1,
    player,
    isPass: true,
  });
}
