import type { GameRules, Player } from '../types/index.js';
import { getOpponent } from '../utils/helpers.js';
import {
  BLACK,
  WHITE,
  EMPTY,
  PASS_MOVE,
  BOARD_SIZE,
  computeLibertyMap,
  computeAreaMapV7KataGo,
  type StoneColor,
} from './fastBoard.js';

const INPUT_SPATIAL_CHANNELS_V7 = 22;
const INPUT_GLOBAL_CHANNELS_V7 = 19;

export type KataGoInputsV7 = {
  spatial: Float32Array; // [19,19,22] NHWC
  global: Float32Array; // [19]
};

export type RecentMove = {
  move: number; // 0..360 or PASS_MOVE
  player: Player;
};

export const idxNHWC = (x: number, y: number, c: number) =>
  (y * BOARD_SIZE + x) * INPUT_SPATIAL_CHANNELS_V7 + c;

function playerToColor(p: Player): StoneColor {
  return p === 'black' ? BLACK : WHITE;
}

export function fillInputsV7Fast(args: {
  stones: Uint8Array; // 0 empty, 1 black, 2 white
  koPoint: number; // 0..360 or -1
  currentPlayer: Player;
  recentMoves: RecentMove[]; // chronological order, last item is most recent
  komi: number;
  rules?: GameRules;
  conservativePassAndIsRoot?: boolean;
  libertyMap?: Uint8Array; // per-point liberties capped to 3, for stones only
  areaMap?: Uint8Array; // KataGo-style area map for planes 18/19
  ladderedStones?: Uint8Array; // V7 plane 14, 1 where stones are ladder-capturable
  prevLadderedStones?: Uint8Array; // V7 plane 15
  prevPrevLadderedStones?: Uint8Array; // V7 plane 16
  ladderWorkingMoves?: Uint8Array; // V7 plane 17, 1 where moves are ladder-capturing
  outSpatial: Float32Array; // len 19*19*22
  outGlobal: Float32Array; // len 19
}): void {
  const { stones, koPoint, currentPlayer, recentMoves, komi } = args;
  const rules: GameRules = args.rules ?? 'japanese';
  const pla = currentPlayer;
  const opp = getOpponent(pla);
  const plaColor = playerToColor(pla);
  const oppColor = playerToColor(opp);

  const spatial = args.outSpatial;
  const global = args.outGlobal;
  spatial.fill(0);
  global.fill(0);

  const totalPoints = BOARD_SIZE * BOARD_SIZE;

  // Ko point (plane 6)
  if (koPoint >= 0 && koPoint < totalPoints) {
    spatial[koPoint * INPUT_SPATIAL_CHANNELS_V7 + 6] = 1.0;
  }

  const libs = args.libertyMap ?? computeLibertyMap(stones);
  const l0 = args.ladderedStones;
  const l1 = args.prevLadderedStones;
  const l2 = args.prevPrevLadderedStones;
  const lm = args.ladderWorkingMoves;
  const hasLadders = Boolean(l0 || l1 || l2 || lm);
  const area = rules === 'chinese' ? (args.areaMap ?? computeAreaMapV7KataGo(stones)) : undefined;

  // Unified single-pass loop over all board points for maximum memory locality & zero redundant math
  for (let pos = 0; pos < totalPoints; pos++) {
    const base = pos * INPUT_SPATIAL_CHANNELS_V7;
    // Plane 0: constant 1.0
    spatial[base] = 1.0;

    const v = stones[pos] as StoneColor;
    if (v === plaColor) {
      spatial[base + 1] = 1.0;
    } else if (v === oppColor) {
      spatial[base + 2] = 1.0;
    }

    if (v !== EMPTY) {
      const l = libs[pos]!;
      if (l === 1) spatial[base + 3] = 1.0;
      else if (l === 2) spatial[base + 4] = 1.0;
      else if (l === 3) spatial[base + 5] = 1.0;
    }

    if (hasLadders) {
      if (l0 && l0[pos]) spatial[base + 14] = 1.0;
      if (l1 && l1[pos]) spatial[base + 15] = 1.0;
      if (l2 && l2[pos]) spatial[base + 16] = 1.0;
      if (lm && lm[pos]) spatial[base + 17] = 1.0;
    }

    if (area) {
      const av = area[pos] as StoneColor;
      if (av === plaColor) spatial[base + 18] = 1.0;
      else if (av === oppColor) spatial[base + 19] = 1.0;
    }
  }

  // KataGo conservativePassAndIsRoot: if a pass now would end the game, suppress history features and also
  // suppress the passWouldEndPhase global so the net doesn't treat the game as ending.
  const lastMove = recentMoves.length > 0 ? recentMoves[recentMoves.length - 1] : null;
  const passWouldEndGame = lastMove?.move === PASS_MOVE;
  const suppressHistory = args.conservativePassAndIsRoot === true && passWouldEndGame;

  const historyPlanes = [9, 10, 11, 12, 13] as const;
  const passGlobals = [0, 1, 2, 3, 4] as const;
  const expectedPlayers: Player[] = [opp, pla, opp, pla, opp];
  if (!suppressHistory) {
    for (let i = 0; i < 5; i++) {
      const m = recentMoves[recentMoves.length - 1 - i];
      if (!m) break;
      if (m.player !== expectedPlayers[i]) break;
      if (m.move === PASS_MOVE) {
        global[passGlobals[i]] = 1.0;
      } else if (m.move >= 0 && m.move < totalPoints) {
        spatial[m.move * INPUT_SPATIAL_CHANNELS_V7 + historyPlanes[i]] = 1.0;
      }
    }
  }

  const selfKomi = pla === 'white' ? komi : -komi;
  global[5] = selfKomi / 20.0;

  if (rules === 'japanese' || rules === 'korean') {
    // KataGo "Japanese": territory scoring + seki tax.
    global[9] = 1.0; // scoring: territory
    global[10] = 1.0; // tax: seki
  }

  global[14] = !suppressHistory && passWouldEndGame ? 1.0 : 0.0;

  if (rules === 'chinese') {
    const boardAreaIsEven = (BOARD_SIZE * BOARD_SIZE) % 2 === 0;
    const drawableKomisAreEven = boardAreaIsEven;

    let komiFloor: number;
    if (drawableKomisAreEven) komiFloor = Math.floor(selfKomi / 2.0) * 2.0;
    else komiFloor = Math.floor((selfKomi - 1.0) / 2.0) * 2.0 + 1.0;

    let delta = selfKomi - komiFloor;
    if (delta < 0.0) delta = 0.0;
    if (delta > 2.0) delta = 2.0;

    let wave: number;
    if (delta < 0.5) wave = delta;
    else if (delta < 1.5) wave = 1.0 - delta;
    else wave = delta - 2.0;
    global[18] = wave;
  }
}

export function extractInputsV7Fast(args: {
  stones: Uint8Array; // 0 empty, 1 black, 2 white
  koPoint: number; // 0..360 or -1
  currentPlayer: Player;
  recentMoves: RecentMove[]; // chronological order, last item is most recent
  komi: number;
  rules?: GameRules;
  conservativePassAndIsRoot?: boolean;
  libertyMap?: Uint8Array; // per-point liberties capped to 3, for stones only
  areaMap?: Uint8Array; // KataGo-style area map for planes 18/19
  ladderedStones?: Uint8Array; // V7 plane 14, 1 where stones are ladder-capturable
  prevLadderedStones?: Uint8Array; // V7 plane 15
  prevPrevLadderedStones?: Uint8Array; // V7 plane 16
  ladderWorkingMoves?: Uint8Array; // V7 plane 17, 1 where moves are ladder-capturing
}): KataGoInputsV7 {
  const spatial = new Float32Array(BOARD_SIZE * BOARD_SIZE * INPUT_SPATIAL_CHANNELS_V7);
  const global = new Float32Array(INPUT_GLOBAL_CHANNELS_V7);
  fillInputsV7Fast({ ...args, outSpatial: spatial, outGlobal: global });
  return { spatial, global };
}
