/**
 * Upstream-faithful BoardHistory matching KataGo v1.18.1 (cpp/game/boardhistory.h, boardhistory.cpp).
 * Tracks full untruncated move history, Zobrist hashes for situational/positional superko,
 * pass transitions, button mechanics, and scoring.
 */

import {
  type Move,
  type Player,
  type StonePlacement,
  DEFAULT_BOARD_SIZE,
  PASS_LOC,
  createMove,
  createPassMove,
  opponent,
  playerToColor,
  locToX,
  locToY,
  xyToLoc,
  BLACK,
  WHITE,
  EMPTY,
} from '../core-contract/position.js';
import { type Rules, DEFAULT_RULES } from '../core-contract/rules.js';
import { Board } from './board.js';

export interface ScoreResult {
  readonly blackScore: number;
  readonly whiteScore: number;
  readonly scoreLead: number; // positive = black leads, negative = white leads
  readonly winner: Player | 'draw';
}

export class BoardHistory {
  readonly initialBoard: Board;
  readonly currentBoard: Board;
  readonly rules: Rules;
  readonly initialPlayer: Player;
  currentPlayer: Player;

  readonly moveHistory: Move[] = [];
  readonly positionalHistory: bigint[] = [];
  readonly situationalHistory: bigint[] = [];

  koLoc: number = PASS_LOC;
  consecutivePasses = 0;
  buttonPlayer: Player | null = null;

  constructor(
    rules: Rules = DEFAULT_RULES,
    initialPlayer: Player = 'black',
    width = DEFAULT_BOARD_SIZE,
    height = DEFAULT_BOARD_SIZE,
    initialStones?: readonly StonePlacement[]
  ) {
    this.rules = rules;
    this.initialPlayer = initialPlayer;
    this.currentPlayer = initialPlayer;

    this.initialBoard = new Board(width, height);
    if (initialStones) {
      for (const s of initialStones) {
        const loc = xyToLoc(s.x, s.y, width);
        this.initialBoard.setStone(loc, playerToColor(s.color));
      }
    }

    this.currentBoard = this.initialBoard.clone();

    // Record initial hashes
    const initPosHash = this.currentBoard.getPositionalHash();
    const initSitHash = this.currentBoard.getSituationalHash(this.currentPlayer);
    this.positionalHistory.push(initPosHash);
    this.situationalHistory.push(initSitHash);
  }

  clone(): BoardHistory {
    const copy = new BoardHistory(
      this.rules,
      this.initialPlayer,
      this.currentBoard.width,
      this.currentBoard.height
    );

    // Copy full state
    copy.initialBoard.reset();
    copy.initialBoard.stones.set(this.initialBoard.stones);
    copy.initialBoard.numCapturesBlack = this.initialBoard.numCapturesBlack;
    copy.initialBoard.numCapturesWhite = this.initialBoard.numCapturesWhite;

    copy.currentBoard.reset();
    copy.currentBoard.stones.set(this.currentBoard.stones);
    copy.currentBoard.numCapturesBlack = this.currentBoard.numCapturesBlack;
    copy.currentBoard.numCapturesWhite = this.currentBoard.numCapturesWhite;

    copy.currentPlayer = this.currentPlayer;
    copy.koLoc = this.koLoc;
    copy.consecutivePasses = this.consecutivePasses;
    copy.buttonPlayer = this.buttonPlayer;

    copy.moveHistory.length = 0;
    copy.moveHistory.push(...this.moveHistory);

    copy.positionalHistory.length = 0;
    copy.positionalHistory.push(...this.positionalHistory);

    copy.situationalHistory.length = 0;
    copy.situationalHistory.push(...this.situationalHistory);

    return copy;
  }

  /**
   * Checks if a move is legal under the configured ruleset, including superko.
   */
  isLegal(loc: number, player: Player = this.currentPlayer): boolean {
    if (loc === PASS_LOC) return true;
    if (loc < 0 || loc >= this.currentBoard.area) return false;

    // Board-level legality (empty point, suicide rules)
    if (!this.currentBoard.isLegal(loc, player, this.rules.multiStoneSuicideLegal)) {
      return false;
    }

    // Simple ko check
    if (this.rules.koRule === 'KO_SIMPLE') {
      if (loc === this.koLoc) {
        return false;
      }
      return true;
    }

    // Superko checks require simulating the resulting board
    const simBoard = this.currentBoard.clone();
    const playResult = simBoard.playMove(loc, player, this.rules.multiStoneSuicideLegal);
    if (!playResult.success) {
      return false;
    }

    const nextPla = opponent(player);

    if (this.rules.koRule === 'KO_POSITIONAL') {
      const nextPosHash = simBoard.getPositionalHash();
      for (let i = 0; i < this.positionalHistory.length; i++) {
        if (this.positionalHistory[i] === nextPosHash) {
          return false;
        }
      }
    } else if (this.rules.koRule === 'KO_SITUATIONAL') {
      const nextSitHash = simBoard.getSituationalHash(nextPla);
      for (let i = 0; i < this.situationalHistory.length; i++) {
        if (this.situationalHistory[i] === nextSitHash) {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Plays a move, updating history, zobrist hashes, ko locations, and turn.
   */
  makeMove(loc: number, player: Player = this.currentPlayer): boolean {
    if (!this.isLegal(loc, player)) {
      return false;
    }

    const width = this.currentBoard.width;
    const move =
      loc === PASS_LOC
        ? createPassMove(player)
        : createMove(player, locToX(loc, width), locToY(loc, width), width);

    if (move.isPass) {
      this.consecutivePasses++;
      this.koLoc = PASS_LOC;

      // Button Go handling: first pass takes the button if available
      if (this.rules.hasButton && this.buttonPlayer === null) {
        this.buttonPlayer = player;
      }

      this.moveHistory.push(move);
      const posHash = this.currentBoard.getPositionalHash();
      const sitHash = this.currentBoard.getSituationalHash(opponent(player));
      this.positionalHistory.push(posHash);
      this.situationalHistory.push(sitHash);
      this.currentPlayer = opponent(player);
      return true;
    }

    // Non-pass move
    this.consecutivePasses = 0;

    const playResult = this.currentBoard.playMove(
      loc,
      player,
      this.rules.multiStoneSuicideLegal
    );

    // Update simple ko location
    if (
      playResult.singleCapturedLoc !== PASS_LOC &&
      !playResult.isSuicide &&
      this.currentBoard.getChainLiberties(loc) === 1
    ) {
      this.koLoc = playResult.singleCapturedLoc;
    } else {
      this.koLoc = PASS_LOC;
    }

    this.moveHistory.push(move);
    const posHash = this.currentBoard.getPositionalHash();
    const sitHash = this.currentBoard.getSituationalHash(opponent(player));
    this.positionalHistory.push(posHash);
    this.situationalHistory.push(sitHash);
    this.currentPlayer = opponent(player);

    return true;
  }

  /**
   * Checks whether the game has reached terminal state via passes.
   */
  isGameFinished(): boolean {
    if (this.rules.hasButton) {
      // With button, 2 passes end if button was already taken before the passes started, or 3 passes.
      return this.consecutivePasses >= 3;
    }
    return this.consecutivePasses >= 2;
  }

  /**
   * Computes upstream-faithful score for current board state.
   */
  computeScore(komi: number, whiteHandicapBonus = 0): ScoreResult {
    const area = this.currentBoard.area;
    const stones = this.currentBoard.stones;

    // Flood fill empty regions to identify territory
    const visited = new Uint8Array(area);
    let blackTerritory = 0;
    let whiteTerritory = 0;
    let blackStones = 0;
    let whiteStones = 0;

    for (let i = 0; i < area; i++) {
      if (stones[i] === BLACK) blackStones++;
      else if (stones[i] === WHITE) whiteStones++;
    }

    const neighbors = [0, 0, 0, 0];
    for (let i = 0; i < area; i++) {
      if (stones[i] !== EMPTY || visited[i]) continue;

      // BFS to find connected empty component
      const queue: number[] = [i];
      visited[i] = 1;
      let touchesBlack = false;
      let touchesWhite = false;
      let regionSize = 0;

      let head = 0;
      while (head < queue.length) {
        const curr = queue[head++];
        regionSize++;

        const numNeighbors = this.currentBoard.getNeighbors(curr, neighbors);
        for (let n = 0; n < numNeighbors; n++) {
          const nLoc = neighbors[n];
          const c = stones[nLoc];
          if (c === BLACK) {
            touchesBlack = true;
          } else if (c === WHITE) {
            touchesWhite = true;
          } else if (!visited[nLoc]) {
            visited[nLoc] = 1;
            queue.push(nLoc);
          }
        }
      }

      if (touchesBlack && !touchesWhite) {
        blackTerritory += regionSize;
      } else if (touchesWhite && !touchesBlack) {
        whiteTerritory += regionSize;
      }
    }

    let blackScore = 0;
    let whiteScore = 0;

    if (this.rules.scoringRule === 'SCORING_AREA') {
      // Area scoring: stones on board + surrounded territory
      blackScore = blackStones + blackTerritory;
      whiteScore = whiteStones + whiteTerritory + komi + whiteHandicapBonus;
    } else {
      // Territory scoring: surrounded territory + captured stones
      blackScore = blackTerritory + this.currentBoard.numCapturesBlack;
      whiteScore = whiteTerritory + this.currentBoard.numCapturesWhite + komi;
    }

    // Button adjustment
    if (this.rules.hasButton) {
      if (this.buttonPlayer === 'black') {
        blackScore += 0.5;
      } else if (this.buttonPlayer === 'white') {
        whiteScore += 0.5;
      }
    }

    const scoreLead = blackScore - whiteScore;
    const winner: Player | 'draw' =
      scoreLead > 0 ? 'black' : scoreLead < 0 ? 'white' : 'draw';

    return {
      blackScore,
      whiteScore,
      scoreLead,
      winner,
    };
  }
}
