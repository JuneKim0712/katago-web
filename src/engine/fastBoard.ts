import {
  EMPTY,
  BLACK,
  WHITE,
  type StoneColor,
  opponentOf,
  BoardGeometry,
  BoardContext,
  type BoardScratch,
  type LadderSearchScratch,
  type LadderFeaturesScratchV7,
  getDefaultBoardContext,
  setDefaultBoardSize,
  getBoardGeometry,
} from './board/context.js';

import {
  type UndoSnapshot,
  type SimPosition,
  playMoveInto,
  playMove,
  undoMove,
} from './board/moves.js';

import {
  collectGroupAndLiberties,
  findLibertiesIntoBuf,
  computeLibertyMapInto,
  updateLibertyMapForSeeds,
  computeLibertyMap,
} from './board/liberties.js';

import {
  computeAreaMapV7KataGo,
  computeAreaMapV7KataGoInto,
} from './board/area.js';

import {
  type KataGoLadderFeaturesV7,
  computeLadderFeaturesV7KataGoInto,
  computeLadderFeaturesV7KataGo,
  computeLadderedStonesV7KataGoInto,
  computeLadderedStonesV7KataGo,
  searchIsLadderCaptured,
  searchIsLadderCapturedAttackerFirst2Libs,
} from './board/ladders.js';

// Re-export types
export type {
  StoneColor,
  UndoSnapshot,
  SimPosition,
  KataGoLadderFeaturesV7,
  BoardScratch,
  LadderSearchScratch,
  LadderFeaturesScratchV7,
};

// Re-export classes & context helpers
export {
  BoardGeometry,
  BoardContext,
  getDefaultBoardContext,
  getBoardGeometry,
  opponentOf,
  playMoveInto,
  playMove,
  undoMove,
  collectGroupAndLiberties,
  findLibertiesIntoBuf,
  computeLibertyMapInto,
  updateLibertyMapForSeeds,
  computeLibertyMap,
  computeAreaMapV7KataGo,
  computeAreaMapV7KataGoInto,
  computeLadderFeaturesV7KataGoInto,
  computeLadderFeaturesV7KataGo,
  computeLadderedStonesV7KataGoInto,
  computeLadderedStonesV7KataGo,
  searchIsLadderCaptured,
  searchIsLadderCapturedAttackerFirst2Libs,
};

// Constants & Compatibility getters/setters for default context
export { EMPTY, BLACK, WHITE };

export let BOARD_SIZE = getDefaultBoardContext().geometry.size;
export let BOARD_AREA = getDefaultBoardContext().geometry.area;
export let PASS_MOVE = getDefaultBoardContext().geometry.passMove;

export let NEIGHBOR_STARTS = getDefaultBoardContext().geometry.neighborStarts;
export let NEIGHBOR_COUNTS = getDefaultBoardContext().geometry.neighborCounts;
export let NEIGHBOR_LIST = getDefaultBoardContext().geometry.neighbors;

export const EMPTY_AREA_MAP = new Uint8Array(BOARD_AREA);

export function setBoardSize(size: number): void {
  const next = Math.max(2, Math.floor(size));
  if (next === BOARD_SIZE) return;
  setDefaultBoardSize(next);
  const def = getDefaultBoardContext();
  BOARD_SIZE = def.geometry.size;
  BOARD_AREA = def.geometry.area;
  PASS_MOVE = def.geometry.passMove;
  NEIGHBOR_STARTS = def.geometry.neighborStarts;
  NEIGHBOR_COUNTS = def.geometry.neighborCounts;
  NEIGHBOR_LIST = def.geometry.neighbors;
}
