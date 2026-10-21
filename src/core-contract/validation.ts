/**
 * Validation and normalization between external/legacy requests and CanonicalPosition.
 * Enforces move legality, board-history consistency, and superko detection.
 */

import {
  type CanonicalPosition,
  type Move,
  type Player,
  type StonePlacement,
  DEFAULT_BOARD_SIZE,
  PASS_LOC,
  createMove,
  createPassMove,
  opponent,
  xyToLoc,
} from './position.js';
import { type Rules, parseRules, DEFAULT_RULES } from './rules.js';
import { BoardHistory } from '../core-runtime/boardHistory.js';
import { Board } from '../core-runtime/board.js';

export interface LegacyAnalyzeRequestLike {
  board?: (Player | null)[][];
  currentPlayer?: Player;
  moveHistory?: Array<{ x: number; y: number; player?: Player }>;
  komi?: number;
  rules?: string | Partial<Rules>;
  initialStones?: StonePlacement[];
  width?: number;
  height?: number;
}

export class PositionValidationError extends Error {
  constructor(message: string) {
    super(`Position validation failed: ${message}`);
    this.name = 'PositionValidationError';
  }
}

/**
 * Builds a BoardHistory instance directly from a CanonicalPosition.
 */
export function createBoardHistoryFromPosition(pos: CanonicalPosition): BoardHistory {
  const history = new BoardHistory(
    pos.rules,
    pos.initialPlayer,
    pos.width,
    pos.height,
    pos.initialStones
  );

  for (const move of pos.moveHistory) {
    const success = history.makeMove(move.loc, move.player);
    if (!success) {
      throw new PositionValidationError(
        `Move history replay failed at move ${move.player} loc=${move.loc} (${move.x}, ${move.y})`
      );
    }
  }

  return history;
}

/**
 * Validates and normalizes any incoming position / legacy request into CanonicalPosition.
 */
export function validateAndNormalizePosition(input: unknown): CanonicalPosition {
  if (!input || typeof input !== 'object') {
    throw new PositionValidationError('Input must be a valid position object');
  }

  const req = input as LegacyAnalyzeRequestLike & Partial<CanonicalPosition>;

  // Determine board dimensions
  let width = req.width ?? DEFAULT_BOARD_SIZE;
  let height = req.height ?? DEFAULT_BOARD_SIZE;

  if (Array.isArray(req.board) && req.board.length > 0) {
    height = req.board.length;
    if (Array.isArray(req.board[0])) {
      width = req.board[0].length;
    }
  }

  if (width < 2 || width > DEFAULT_BOARD_SIZE || height < 2 || height > DEFAULT_BOARD_SIZE) {
    throw new PositionValidationError(
      `Invalid board dimensions ${width}x${height}. Supported ranges: 2..${DEFAULT_BOARD_SIZE}`
    );
  }

  // Parse rules
  const rules = parseRules(req.rules);

  // Komi
  const komi = typeof req.komi === 'number' && Number.isFinite(req.komi) ? req.komi : 7.5;
  const whiteHandicapBonus = req.whiteHandicapBonus ?? 0;

  // Initial stones
  const initialStones: StonePlacement[] = [];
  if (Array.isArray(req.initialStones)) {
    for (const s of req.initialStones) {
      if (s.x >= 0 && s.x < width && s.y >= 0 && s.y < height && (s.color === 'black' || s.color === 'white')) {
        initialStones.push({ x: s.x, y: s.y, color: s.color });
      }
    }
  }

  // Normalize move history
  const normalizedMoves: Move[] = [];
  let runningPlayer: Player = 'black';

  if (Array.isArray(req.moveHistory)) {
    for (const rawMove of req.moveHistory) {
      const pla: Player = rawMove.player === 'white' ? 'white' : 'black';
      const x = typeof rawMove.x === 'number' ? rawMove.x : -1;
      const y = typeof rawMove.y === 'number' ? rawMove.y : -1;

      if (x < 0 || y < 0) {
        normalizedMoves.push(createPassMove(pla));
      } else {
        if (x >= width || y >= height) {
          throw new PositionValidationError(`Move coordinates out of bounds: (${x}, ${y}) for ${width}x${height}`);
        }
        normalizedMoves.push(createMove(pla, x, y, width));
      }
      runningPlayer = opponent(pla);
    }
  }

  const initialPlayer: Player =
    normalizedMoves.length > 0 ? normalizedMoves[0].player : (req.currentPlayer ?? 'black');
  const currentPlayer: Player = req.currentPlayer ?? runningPlayer;

  // Board snapshot consistency check
  if (Array.isArray(req.board)) {
    const board = req.board;

    // Check if replaying from initialStones + moves produces the board
    const testHistory = new BoardHistory(rules, initialPlayer, width, height, initialStones);
    let replayConsistent = true;

    for (const m of normalizedMoves) {
      const ok = testHistory.makeMove(m.loc, m.player);
      if (!ok) {
        replayConsistent = false;
        break;
      }
    }

    if (replayConsistent) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const expected = board[y]?.[x] ?? null;
          const loc = xyToLoc(x, y, width);
          const actualColor = testHistory.currentBoard.getColor(loc);
          const actual = actualColor === 1 ? 'black' : actualColor === 2 ? 'white' : null;
          if (expected !== actual) {
            replayConsistent = false;
            break;
          }
        }
        if (!replayConsistent) break;
      }
    }

    if (!replayConsistent) {
      // If moveHistory was not provided or was partial, infer initial stones from board snapshot
      if (normalizedMoves.length === 0 && initialStones.length === 0) {
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const c = board[y]?.[x];
            if (c === 'black' || c === 'white') {
              initialStones.push({ x, y, color: c });
            }
          }
        }
      } else if (initialStones.length > 0 && normalizedMoves.length > 0) {
        throw new PositionValidationError(
          'Contradictory position: replaying initialStones + moveHistory does not match provided board snapshot'
        );
      }
    }
  }

  const canonical: CanonicalPosition = Object.freeze({
    width,
    height,
    initialStones: Object.freeze(initialStones),
    initialPlayer,
    moveHistory: Object.freeze(normalizedMoves),
    currentPlayer,
    komi,
    rules: Object.freeze(rules),
    whiteHandicapBonus,
  });

  return canonical;
}
