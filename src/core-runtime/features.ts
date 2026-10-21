/**
 * Upstream-faithful neural network feature generator matching KataGo v1.18.1 (cpp/neuralnet/nninputs.cpp).
 * Generates 22 spatial channels and 19 global channels from canonical BoardHistory.
 * Supports both NHWC and NCHW layouts.
 */

import {
  type Player,
  DEFAULT_BOARD_SIZE,
  PASS_LOC,
  opponent,
  xyToLoc,
  playerToColor,
  BLACK,
  WHITE,
  EMPTY,
} from '../core-contract/position.js';
import { BoardHistory } from './boardHistory.js';

export const NUM_SPATIAL_CHANNELS = 22;
export const NUM_GLOBAL_CHANNELS = 19;

export type TensorLayout = 'NHWC' | 'NCHW';

export interface FeatureExtractionOptions {
  layout?: TensorLayout;
  conservativePassAndIsRoot?: boolean;
  symmetry?: number; // 0..7
}

export interface EncodedFeatures {
  readonly spatial: Float32Array; // [19, 19, 22] or [22, 19, 19]
  readonly global: Float32Array; // [19]
  readonly layout: TensorLayout;
  readonly width: number;
  readonly height: number;
}

/**
 * Computes 8-way board symmetries for spatial coordinates.
 */
export function applySymmetry(x: number, y: number, sym: number, width = DEFAULT_BOARD_SIZE, height = DEFAULT_BOARD_SIZE): [number, number] {
  let sx = x;
  let sy = y;
  // Symmetries: 0..7
  // 0: identity
  // 1: flip X
  // 2: flip Y
  // 3: flip both
  // 4: transpose
  // 5: rot 90
  // 6: rot 180
  // 7: rot 270
  if (sym & 1) sx = width - 1 - sx;
  if (sym & 2) sy = height - 1 - sy;
  if (sym & 4) {
    const tmp = sx;
    sx = sy;
    sy = tmp;
  }
  return [sx, sy];
}

/**
 * Extracts KataGo v1.18.1 input features directly from a BoardHistory.
 */
export function extractFeatures(
  history: BoardHistory,
  komi: number,
  options: FeatureExtractionOptions = {}
): EncodedFeatures {
  const layout = options.layout ?? 'NHWC';
  const symmetry = options.symmetry ?? 0;
  const board = history.currentBoard;
  const width = board.width;
  const height = board.height;
  const area = board.area;

  const spatial = new Float32Array(area * NUM_SPATIAL_CHANNELS);
  const global = new Float32Array(NUM_GLOBAL_CHANNELS);

  const pla = history.currentPlayer;
  const opp = opponent(pla);
  const plaColor = playerToColor(pla);
  const oppColor = playerToColor(opp);

  // Compute liberties for stones on the board
  const libertyMap = new Uint8Array(area);
  board.computeLibertyMap(libertyMap);

  // Helper to set spatial feature value based on chosen layout
  function setSpatial(x: number, y: number, channel: number, val: number): void {
    const [symX, symY] = symmetry > 0 ? applySymmetry(x, y, symmetry, width, height) : [x, y];
    if (layout === 'NHWC') {
      const idx = (symY * width + symX) * NUM_SPATIAL_CHANNELS + channel;
      spatial[idx] = val;
    } else {
      // NCHW: [channel, y, x]
      const idx = (channel * height + symY) * width + symX;
      spatial[idx] = val;
    }
  }

  // Plane 0: Constant 1.0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      setSpatial(x, y, 0, 1.0);
    }
  }

  // Planes 1-5: Current player stones, opponent stones, and liberties
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const loc = xyToLoc(x, y, width);
      const color = board.getColor(loc);
      if (color === plaColor) {
        setSpatial(x, y, 1, 1.0);
      } else if (color === oppColor) {
        setSpatial(x, y, 2, 1.0);
      }

      if (color !== EMPTY) {
        const libs = libertyMap[loc];
        if (libs === 1) setSpatial(x, y, 3, 1.0);
        else if (libs === 2) setSpatial(x, y, 4, 1.0);
        else if (libs === 3) setSpatial(x, y, 5, 1.0);
      }
    }
  }

  // Plane 6: Ko location
  if (history.koLoc !== PASS_LOC && history.koLoc >= 0 && history.koLoc < area) {
    const koX = history.koLoc % width;
    const koY = Math.floor(history.koLoc / width);
    setSpatial(koX, koY, 6, 1.0);
  }

  // Planes 9-13: Untruncated move history (last 5 moves)
  const recentMoves = history.moveHistory;
  const lastMove = recentMoves.length > 0 ? recentMoves[recentMoves.length - 1] : null;
  const passWouldEndGame = lastMove?.isPass === true;
  const suppressHistory = options.conservativePassAndIsRoot === true && passWouldEndGame;

  const historyPlanes = [9, 10, 11, 12, 13] as const;
  const passGlobals = [0, 1, 2, 3, 4] as const;
  const expectedPlayers: Player[] = [opp, pla, opp, pla, opp];

  if (!suppressHistory) {
    for (let i = 0; i < 5; i++) {
      const m = recentMoves[recentMoves.length - 1 - i];
      if (!m) break;
      if (m.player !== expectedPlayers[i]) break;
      if (m.isPass) {
        global[passGlobals[i]] = 1.0;
      } else if (m.loc >= 0 && m.loc < area) {
        setSpatial(m.x, m.y, historyPlanes[i], 1.0);
      }
    }
  }

  // Global features
  const selfKomi = pla === 'white' ? komi : -komi;
  global[5] = selfKomi / 20.0;
  global[6] = komi / 20.0;

  // Rules indicators
  if (history.rules.koRule === 'KO_SITUATIONAL') global[7] = 1.0;
  if (history.rules.multiStoneSuicideLegal) global[8] = 1.0;
  if (history.rules.scoringRule === 'SCORING_TERRITORY') global[9] = 1.0;
  if (history.rules.taxRule === 'TAX_SEKI') global[10] = 1.0;
  else if (history.rules.taxRule === 'TAX_ALL') global[10] = -1.0;
  if (history.rules.hasButton) global[11] = 1.0;

  // White handicap bonus
  if (history.rules.whiteHandicapBonusRule === 'WHB_N') global[12] = 1.0;
  else if (history.rules.whiteHandicapBonusRule === 'WHB_N_MINUS_ONE') global[12] = 0.5;

  global[14] = !suppressHistory && passWouldEndGame ? 1.0 : 0.0;

  // Chinese rules odd/even parity wave on komi
  if (history.rules.scoringRule === 'SCORING_AREA') {
    const boardAreaIsEven = area % 2 === 0;
    const drawableKomisAreEven = boardAreaIsEven;

    let komiFloor: number;
    if (drawableKomisAreEven) {
      komiFloor = Math.floor(selfKomi / 2.0) * 2.0;
    } else {
      komiFloor = Math.floor((selfKomi - 1.0) / 2.0) * 2.0 + 1.0;
    }

    let delta = selfKomi - komiFloor;
    if (delta < 0.0) delta = 0.0;
    if (delta > 2.0) delta = 2.0;

    let wave: number;
    if (delta < 0.5) wave = delta;
    else if (delta < 1.5) wave = 1.0 - delta;
    else wave = delta - 2.0;

    global[18] = wave;
  }

  return {
    spatial,
    global,
    layout,
    width,
    height,
  };
}
