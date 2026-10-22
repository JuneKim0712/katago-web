/**
 * Upstream-faithful Go board representation matching KataGo v1.18.1 (cpp/game/board.h, board.cpp).
 * Includes strict liberty tracking, chain maintenance, captures, and multi-stone suicide handling.
 */

import {
  type Color,
  type Player,
  EMPTY,
  BLACK,
  WHITE,
  DEFAULT_BOARD_SIZE,
  PASS_LOC,
  playerToColor,
  colorToPlayer,
  opponent,
} from '../core-contract/position.js';

// Deterministic Zobrist table initialization using splitmix64 / xorshift64
const ZOBRIST_BLACK = new BigUint64Array(DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE);
const ZOBRIST_WHITE = new BigUint64Array(DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE);
let ZOBRIST_PLAYER_BLACK = 0n;
let ZOBRIST_PLAYER_WHITE = 0n;

function initZobrist(): void {
  let state = 0x853c49e6748fea9bn;
  function nextRand64(): bigint {
    state ^= state >> 12n;
    state ^= (state << 25n) & 0xffffffffffffffffn;
    state ^= state >> 27n;
    return (state * 0x2545f4914f6cdd1dn) & 0xffffffffffffffffn;
  }

  for (let i = 0; i < DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE; i++) {
    ZOBRIST_BLACK[i] = nextRand64();
    ZOBRIST_WHITE[i] = nextRand64();
  }
  ZOBRIST_PLAYER_BLACK = nextRand64();
  ZOBRIST_PLAYER_WHITE = nextRand64();
}
initZobrist();

export { ZOBRIST_BLACK, ZOBRIST_WHITE, ZOBRIST_PLAYER_BLACK, ZOBRIST_PLAYER_WHITE };

export interface BoardCopy {
  width: number;
  height: number;
  stones: Uint8Array;
  numCapturesBlack: number;
  numCapturesWhite: number;
}

export class Board {
  readonly width: number;
  readonly height: number;
  readonly area: number;

  // 0: empty, 1: black, 2: white
  readonly stones: Uint8Array;

  // Chain tracking: parent pointer for disjoint set
  private readonly parent: Int16Array;
  // Size of chain (valid only at chain root)
  private readonly chainSize: Int16Array;
  // Next stone in chain circular linked-list
  private readonly nextInChain: Int16Array;

  numCapturesBlack = 0;
  numCapturesWhite = 0;

  constructor(width = DEFAULT_BOARD_SIZE, height = DEFAULT_BOARD_SIZE) {
    this.width = width;
    this.height = height;
    this.area = width * height;

    this.stones = new Uint8Array(this.area);
    this.parent = new Int16Array(this.area);
    this.chainSize = new Int16Array(this.area);
    this.nextInChain = new Int16Array(this.area);

    this.reset();
  }

  reset(): void {
    this.stones.fill(EMPTY);
    for (let i = 0; i < this.area; i++) {
      this.parent[i] = i;
      this.chainSize[i] = 1;
      this.nextInChain[i] = i;
    }
    this.numCapturesBlack = 0;
    this.numCapturesWhite = 0;
  }

  clone(): Board {
    const copy = new Board(this.width, this.height);
    copy.stones.set(this.stones);
    copy.parent.set(this.parent);
    copy.chainSize.set(this.chainSize);
    copy.nextInChain.set(this.nextInChain);
    copy.numCapturesBlack = this.numCapturesBlack;
    copy.numCapturesWhite = this.numCapturesWhite;
    return copy;
  }

  getColor(loc: number): Color {
    if (loc < 0 || loc >= this.area) return EMPTY;
    return this.stones[loc] as Color;
  }

  findRoot(loc: number): number {
    let curr = loc;
    while (this.parent[curr] !== curr) {
      this.parent[curr] = this.parent[this.parent[curr]]; // path compression
      curr = this.parent[curr];
    }
    return curr;
  }

  /**
   * Returns the 4 orthogonal neighbors of a location.
   */
  getNeighbors(loc: number, out: number[]): number {
    const x = loc % this.width;
    const y = Math.floor(loc / this.width);
    let count = 0;

    if (x > 0) out[count++] = loc - 1;
    if (x < this.width - 1) out[count++] = loc + 1;
    if (y > 0) out[count++] = loc - this.width;
    if (y < this.height - 1) out[count++] = loc + this.width;

    return count;
  }

  /**
   * Computes exact liberties for the chain containing `loc`.
   */
  getChainLiberties(loc: number): number {
    if (this.stones[loc] === EMPTY) return 0;
    const root = this.findRoot(loc);
    const seenLibs = new Set<number>();

    let curr = root;
    const neighbors: number[] = [0, 0, 0, 0];
    do {
      const numNeighbors = this.getNeighbors(curr, neighbors);
      for (let i = 0; i < numNeighbors; i++) {
        const nLoc = neighbors[i];
        if (this.stones[nLoc] === EMPTY) {
          seenLibs.add(nLoc);
        }
      }
      curr = this.nextInChain[curr];
    } while (curr !== root);

    return seenLibs.size;
  }

  /**
   * Computes liberties for all stones on the board into a destination Uint8Array.
   */
  computeLibertyMap(out: Uint8Array): void {
    out.fill(0);
    const visitedRoots = new Map<number, number>();

    for (let loc = 0; loc < this.area; loc++) {
      if (this.stones[loc] === EMPTY) continue;
      const root = this.findRoot(loc);
      let libs = visitedRoots.get(root);
      if (libs === undefined) {
        libs = this.getChainLiberties(root);
        visitedRoots.set(root, libs);
      }
      out[loc] = Math.min(255, libs);
    }
  }

  /**
   * Removes a chain of stones starting at root, returning the number of stones removed.
   */
  private removeChain(root: number): number {
    let count = 0;
    let curr = root;
    const toClear: number[] = [];
    do {
      toClear.push(curr);
      curr = this.nextInChain[curr];
      count++;
    } while (curr !== root);

    for (const loc of toClear) {
      this.stones[loc] = EMPTY;
      this.parent[loc] = loc;
      this.chainSize[loc] = 1;
      this.nextInChain[loc] = loc;
    }
    return count;
  }

  /**
   * Joins two adjacent chains of the same color.
   */
  private mergeChains(root1: number, root2: number): void {
    if (root1 === root2) return;

    // Swap if needed to attach smaller to larger
    let r1 = root1;
    let r2 = root2;
    if (this.chainSize[r1] < this.chainSize[r2]) {
      const tmp = r1;
      r1 = r2;
      r2 = tmp;
    }

    this.parent[r2] = r1;
    this.chainSize[r1] += this.chainSize[r2];

    // Splice circular linked lists
    const next1 = this.nextInChain[r1];
    const next2 = this.nextInChain[r2];
    this.nextInChain[r1] = next2;
    this.nextInChain[r2] = next1;
  }

  /**
   * Directly places a stone without legality checks (e.g. for initial stone setup).
   */
  setStone(loc: number, color: Color): void {
    if (loc < 0 || loc >= this.area) return;
    if (this.stones[loc] !== EMPTY) {
      this.removeChain(this.findRoot(loc));
    }
    if (color === EMPTY) return;

    this.stones[loc] = color;
    this.parent[loc] = loc;
    this.chainSize[loc] = 1;
    this.nextInChain[loc] = loc;

    const neighbors: number[] = [0, 0, 0, 0];
    const numNeighbors = this.getNeighbors(loc, neighbors);
    for (let i = 0; i < numNeighbors; i++) {
      const nLoc = neighbors[i];
      if (this.stones[nLoc] === color) {
        this.mergeChains(this.findRoot(loc), this.findRoot(nLoc));
      }
    }
  }

  /**
   * Checks if playing a move is legal without modifying the board.
   */
  isLegal(loc: number, player: Player, multiStoneSuicideLegal: boolean): boolean {
    if (loc === PASS_LOC) return true;
    if (loc < 0 || loc >= this.area) return false;
    if (this.stones[loc] !== EMPTY) return false;

    const color = playerToColor(player);
    const oppColor = color === BLACK ? WHITE : BLACK;

    const neighbors: number[] = [0, 0, 0, 0];
    const numNeighbors = this.getNeighbors(loc, neighbors);

    // 1. Direct liberty?
    for (let i = 0; i < numNeighbors; i++) {
      if (this.stones[neighbors[i]] === EMPTY) {
        return true;
      }
    }

    // 2. Captures opponent?
    const checkedOppRoots = new Set<number>();
    for (let i = 0; i < numNeighbors; i++) {
      const nLoc = neighbors[i];
      if (this.stones[nLoc] === oppColor) {
        const root = this.findRoot(nLoc);
        if (!checkedOppRoots.has(root)) {
          checkedOppRoots.add(root);
          if (this.getChainLiberties(root) === 1) {
            // This neighbor only had `loc` as its liberty -> playing here captures it!
            return true;
          }
        }
      }
    }

    // 3. Connects to friendly group with >1 liberty?
    const checkedFriendlyRoots = new Set<number>();
    for (let i = 0; i < numNeighbors; i++) {
      const nLoc = neighbors[i];
      if (this.stones[nLoc] === color) {
        const root = this.findRoot(nLoc);
        if (!checkedFriendlyRoots.has(root)) {
          checkedFriendlyRoots.add(root);
          if (this.getChainLiberties(root) > 1) {
            return true;
          }
        }
      }
    }

    // 4. Otherwise, this move has 0 liberties. Legal if and only if multiStoneSuicideLegal is true.
    return multiStoneSuicideLegal;
  }

  /**
   * Plays a move on the board.
   * Returns `{ success: boolean, numCaptures: number, isSuicide: boolean, singleCapturedLoc: number }`
   */
  playMove(
    loc: number,
    player: Player,
    multiStoneSuicideLegal = false
  ): {
    success: boolean;
    numCaptures: number;
    isSuicide: boolean;
    singleCapturedLoc: number;
  } {
    if (loc === PASS_LOC) {
      return { success: true, numCaptures: 0, isSuicide: false, singleCapturedLoc: PASS_LOC };
    }

    if (!this.isLegal(loc, player, multiStoneSuicideLegal)) {
      return { success: false, numCaptures: 0, isSuicide: false, singleCapturedLoc: PASS_LOC };
    }

    const color = playerToColor(player);
    const oppColor = color === BLACK ? WHITE : BLACK;

    // Place stone
    this.stones[loc] = color;
    this.parent[loc] = loc;
    this.chainSize[loc] = 1;
    this.nextInChain[loc] = loc;

    const neighbors: number[] = [0, 0, 0, 0];
    const numNeighbors = this.getNeighbors(loc, neighbors);

    // Merge with friendly neighbors
    for (let i = 0; i < numNeighbors; i++) {
      const nLoc = neighbors[i];
      if (this.stones[nLoc] === color) {
        this.mergeChains(this.findRoot(loc), this.findRoot(nLoc));
      }
    }

    // Capture opponent groups with 0 liberties
    let numCaptures = 0;
    let singleCapturedLoc = PASS_LOC;
    const checkedOppRoots = new Set<number>();

    for (let i = 0; i < numNeighbors; i++) {
      const nLoc = neighbors[i];
      if (this.stones[nLoc] === oppColor) {
        const root = this.findRoot(nLoc);
        if (!checkedOppRoots.has(root)) {
          checkedOppRoots.add(root);
          if (this.getChainLiberties(root) === 0) {
            const rootSize = this.chainSize[root];
            if (rootSize === 1) {
              singleCapturedLoc = root;
            } else {
              singleCapturedLoc = PASS_LOC;
            }
            const removed = this.removeChain(root);
            numCaptures += removed;
          }
        }
      }
    }

    if (player === 'black') {
      this.numCapturesBlack += numCaptures;
    } else {
      this.numCapturesWhite += numCaptures;
    }

    // Check suicide
    let isSuicide = false;
    const finalRoot = this.findRoot(loc);
    if (this.getChainLiberties(finalRoot) === 0) {
      // Must be multiStoneSuicideLegal
      isSuicide = true;
      const suicideStones = this.removeChain(finalRoot);
      if (player === 'black') {
        this.numCapturesWhite += suicideStones; // Suicide counts as captures for opponent in scoring
      } else {
        this.numCapturesBlack += suicideStones;
      }
      singleCapturedLoc = PASS_LOC;
    }

    if (numCaptures > 1) {
      singleCapturedLoc = PASS_LOC;
    }

    return {
      success: true,
      numCaptures,
      isSuicide,
      singleCapturedLoc,
    };
  }

  /**
   * Computes the 64-bit Zobrist hash of the current board stones (positional hash).
   */
  getPositionalHash(): bigint {
    let hash = 0n;
    for (let i = 0; i < this.area; i++) {
      const color = this.stones[i];
      if (color === BLACK) {
        hash ^= ZOBRIST_BLACK[i];
      } else if (color === WHITE) {
        hash ^= ZOBRIST_WHITE[i];
      }
    }
    return hash;
  }

  /**
   * Computes situational hash: stones + next player to move.
   */
  getSituationalHash(nextPlayer: Player): bigint {
    let hash = this.getPositionalHash();
    hash ^= nextPlayer === 'black' ? ZOBRIST_PLAYER_BLACK : ZOBRIST_PLAYER_WHITE;
    return hash;
  }
}
