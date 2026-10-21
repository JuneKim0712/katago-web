export const EMPTY: StoneColor = 0;
export const BLACK: StoneColor = 1;
export const WHITE: StoneColor = 2;

export type StoneColor = 0 | 1 | 2;

export const NUM_SYMMETRIES = 8;

export function buildSymPosMap(boardSize = 19): Int16Array {
  const n = boardSize;
  const boardArea = n * n;
  const map = new Int16Array(NUM_SYMMETRIES * boardArea);
  for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
    const symOff = sym * boardArea;
    const mirror = sym >= 4;
    const rot = sym & 3;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const sx = mirror ? n - 1 - x : x;
        const sy = y;
        let tx: number;
        let ty: number;
        if (rot === 0) {
          tx = sx;
          ty = sy;
        } else if (rot === 1) {
          tx = sy;
          ty = n - 1 - sx;
        } else if (rot === 2) {
          tx = n - 1 - sx;
          ty = n - 1 - sy;
        } else {
          tx = n - 1 - sy;
          ty = sx;
        }
        map[symOff + y * n + x] = ty * n + tx;
      }
    }
  }
  return map;
}

export function opponentOf(color: StoneColor): StoneColor {
  return color === BLACK ? WHITE : color === WHITE ? BLACK : EMPTY;
}

export const LADDER_BUF_SIZE = 120000;

export interface LadderSearchScratch {
  bufMoves: Int16Array;
  moveListStarts: Int32Array;
  moveListLens: Int32Array;
  moveListCur: Int32Array;
  recordMoves: Int16Array;
  recordPlayers: Uint8Array;
  recordKoPointBefore: Int16Array;
  recordCaptureStart: Int32Array;
  tmpKoPointBefore: Int16Array;
  tmpCaptureStart: Int32Array;
  captureStack: number[];
  tempLibs16: Int16Array;
  attackerLibs2: Int16Array;
}

export interface LadderFeaturesScratchV7 {
  copyPos: { stones: Uint8Array; koPoint: number };
  groupStones: Int16Array;
  workingMoves: number[];
}

export interface BoardScratch {
  visited: Int32Array;
  libVisited: Int32Array;
  bfsStamp: number;
  groupBuf: Int16Array;
  stackBuf: Int16Array;
  processedGroup: Int32Array;
  processedStamp: number;
  groupSeen: Int32Array;
  groupSeenStamp: number;

  // Area mapping scratch
  regionIdxByPos: Int16Array;
  nextEmptyOrOpp: Int16Array;
  bordersNonPassAliveByHeadPos: Uint8Array;
  groupIndexByPos: Int16Array;
  groupColorByGroup: Uint8Array;
  groupStartByGroup: Int16Array;
  groupLenByGroup: Int16Array;
  groupStonesFlat: Int16Array;
  regionHeads: Int16Array;
  vitalStart: Uint16Array;
  vitalLen: Uint8Array;
  numInternalSpacesMax2: Uint8Array;
  containsOpp: Uint8Array;
  vitalList: Int16Array;
  regionQueue: Int16Array;
  plaGroups: Int16Array;
  plaGroupKilled: Uint8Array;
  vitalCountByGroup: Int16Array;

  // Ladder search scratch
  ladderScratch: LadderSearchScratch;
  ladderGroupSeen: Int32Array;
  ladderOppGroupSeen: Int32Array;
  ladderGroupCopy: Int16Array;
  ladderConnectGroupSeen: Int32Array;
  ladderCaptured: Int32Array;
  ladderGroupSeenStamp: number;
  ladderOppGroupSeenStamp: number;
  ladderConnectGroupSeenStamp: number;
  ladderCapturedStamp: number;
  ladderFeaturesScratch: LadderFeaturesScratchV7;
}

export class BoardGeometry {
  readonly size: number;
  readonly area: number;
  readonly passMove: number;
  readonly neighborStarts: Int16Array;
  readonly neighborCounts: Int8Array;
  readonly neighbors: Int16Array;
  readonly symmetryMaps: Int16Array;

  constructor(size: number) {
    this.size = size;
    this.area = size * size;
    this.passMove = this.area;

    this.neighborStarts = new Int16Array(this.area);
    this.neighborCounts = new Int8Array(this.area);
    this.neighbors = new Int16Array(this.area * 4);

    let neighOffset = 0;
    // Neighbor ordering must be preserved: left, right, up, down
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const pos = y * size + x;
        this.neighborStarts[pos] = neighOffset;
        let count = 0;
        if (x > 0) {
          this.neighbors[neighOffset++] = pos - 1; // Left
          count++;
        }
        if (x + 1 < size) {
          this.neighbors[neighOffset++] = pos + 1; // Right
          count++;
        }
        if (y > 0) {
          this.neighbors[neighOffset++] = pos - size; // Up
          count++;
        }
        if (y + 1 < size) {
          this.neighbors[neighOffset++] = pos + size; // Down
          count++;
        }
        this.neighborCounts[pos] = count;
      }
    }

    this.symmetryMaps = buildSymPosMap(size);
  }
}

const geometryCache = new Map<number, BoardGeometry>();

export function getBoardGeometry(size = 19): BoardGeometry {
  let g = geometryCache.get(size);
  if (!g) {
    g = new BoardGeometry(size);
    geometryCache.set(size, g);
  }
  return g;
}

export function createBoardScratch(area: number): BoardScratch {
  const maxRegions = ((area + 1) / 2 + 1) | 0;
  const ladderStackSize = ((area * 3) / 2 + 2) | 0;

  return {
    visited: new Int32Array(area),
    libVisited: new Int32Array(area),
    bfsStamp: 0,
    groupBuf: new Int16Array(area),
    stackBuf: new Int16Array(area),
    processedGroup: new Int32Array(area),
    processedStamp: 0,
    groupSeen: new Int32Array(area),
    groupSeenStamp: 0,

    regionIdxByPos: new Int16Array(area),
    nextEmptyOrOpp: new Int16Array(area),
    bordersNonPassAliveByHeadPos: new Uint8Array(area),
    groupIndexByPos: new Int16Array(area),
    groupColorByGroup: new Uint8Array(area),
    groupStartByGroup: new Int16Array(area),
    groupLenByGroup: new Int16Array(area),
    groupStonesFlat: new Int16Array(area),
    regionHeads: new Int16Array(maxRegions),
    vitalStart: new Uint16Array(maxRegions),
    vitalLen: new Uint8Array(maxRegions),
    numInternalSpacesMax2: new Uint8Array(maxRegions),
    containsOpp: new Uint8Array(maxRegions),
    vitalList: new Int16Array(maxRegions * 4),
    regionQueue: new Int16Array(area),
    plaGroups: new Int16Array(area),
    plaGroupKilled: new Uint8Array(area),
    vitalCountByGroup: new Int16Array(area),

    ladderScratch: {
      bufMoves: new Int16Array(LADDER_BUF_SIZE),
      moveListStarts: new Int32Array(ladderStackSize),
      moveListLens: new Int32Array(ladderStackSize),
      moveListCur: new Int32Array(ladderStackSize),
      recordMoves: new Int16Array(ladderStackSize),
      recordPlayers: new Uint8Array(ladderStackSize),
      recordKoPointBefore: new Int16Array(ladderStackSize),
      recordCaptureStart: new Int32Array(ladderStackSize),
      tmpKoPointBefore: new Int16Array(1),
      tmpCaptureStart: new Int32Array(1),
      captureStack: [],
      tempLibs16: new Int16Array(16),
      attackerLibs2: new Int16Array(2),
    },
    ladderGroupSeen: new Int32Array(area),
    ladderOppGroupSeen: new Int32Array(area),
    ladderGroupCopy: new Int16Array(area),
    ladderConnectGroupSeen: new Int32Array(area),
    ladderCaptured: new Int32Array(area),
    ladderGroupSeenStamp: 0,
    ladderOppGroupSeenStamp: 0,
    ladderConnectGroupSeenStamp: 0,
    ladderCapturedStamp: 0,
    ladderFeaturesScratch: {
      copyPos: { stones: new Uint8Array(area), koPoint: -1 },
      groupStones: new Int16Array(area),
      workingMoves: [],
    },
  };
}

export const STAMP_ROLLOVER_THRESHOLD = 0x7ffff000;

export class BoardContext {
  readonly geometry: BoardGeometry;
  readonly scratch: BoardScratch;

  constructor(geometryOrSize: BoardGeometry | number = 19) {
    this.geometry = typeof geometryOrSize === 'number' ? getBoardGeometry(geometryOrSize) : geometryOrSize;
    this.scratch = createBoardScratch(this.geometry.area);
  }

  nextBfsStamp(): number {
    if (this.scratch.bfsStamp >= STAMP_ROLLOVER_THRESHOLD) {
      this.scratch.visited.fill(0);
      this.scratch.libVisited.fill(0);
      this.scratch.bfsStamp = 0;
    }
    return ++this.scratch.bfsStamp;
  }

  nextProcessedStamp(): number {
    if (this.scratch.processedStamp >= STAMP_ROLLOVER_THRESHOLD) {
      this.scratch.processedGroup.fill(0);
      this.scratch.processedStamp = 0;
    }
    return ++this.scratch.processedStamp;
  }

  nextGroupSeenStamp(): number {
    if (this.scratch.groupSeenStamp >= STAMP_ROLLOVER_THRESHOLD) {
      this.scratch.groupSeen.fill(0);
      this.scratch.groupSeenStamp = 0;
    }
    return ++this.scratch.groupSeenStamp;
  }

  nextLadderGroupSeenStamp(): number {
    if (this.scratch.ladderGroupSeenStamp >= STAMP_ROLLOVER_THRESHOLD) {
      this.scratch.ladderGroupSeen.fill(0);
      this.scratch.ladderGroupSeenStamp = 0;
    }
    return ++this.scratch.ladderGroupSeenStamp;
  }

  nextLadderOppGroupSeenStamp(): number {
    if (this.scratch.ladderOppGroupSeenStamp >= STAMP_ROLLOVER_THRESHOLD) {
      this.scratch.ladderOppGroupSeen.fill(0);
      this.scratch.ladderOppGroupSeenStamp = 0;
    }
    return ++this.scratch.ladderOppGroupSeenStamp;
  }

  nextLadderConnectGroupSeenStamp(): number {
    if (this.scratch.ladderConnectGroupSeenStamp >= STAMP_ROLLOVER_THRESHOLD) {
      this.scratch.ladderConnectGroupSeen.fill(0);
      this.scratch.ladderConnectGroupSeenStamp = 0;
    }
    return ++this.scratch.ladderConnectGroupSeenStamp;
  }

  nextLadderCapturedStamp(): number {
    if (this.scratch.ladderCapturedStamp >= STAMP_ROLLOVER_THRESHOLD) {
      this.scratch.ladderCaptured.fill(0);
      this.scratch.ladderCapturedStamp = 0;
    }
    return ++this.scratch.ladderCapturedStamp;
  }
}

let defaultContext = new BoardContext(19);

export function getDefaultBoardContext(): BoardContext {
  return defaultContext;
}

export function setDefaultBoardSize(size: number): void {
  const next = Math.max(2, Math.floor(size));
  if (next === defaultContext.geometry.size) return;
  defaultContext = new BoardContext(next);
}
