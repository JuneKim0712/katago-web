import { BOARD_AREA, BOARD_SIZE, type StoneColor } from '../fastBoard.js';

type Uint8Buf = Uint8Array<ArrayBufferLike>;

interface LadderCacheEntry {
  stones: Uint8Buf;
  koPoint: number;
  playerColor: StoneColor;
  ladderedStones: Uint8Buf;
  ladderWorkingMoves?: Uint8Buf;
}

interface AreaCacheEntry {
  stones: Uint8Buf;
  areaMap: Uint8Buf;
}

function stonesEqual(a: Uint8Buf, b: Uint8Buf): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function hashStones(stones: Uint8Buf, koPoint: number, color: number): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < stones.length; i++) {
    h ^= stones[i]!;
    h = Math.imul(h, 0x01000193);
  }
  h ^= (koPoint + 1);
  h = Math.imul(h, 0x01000193);
  h ^= color;
  h = Math.imul(h, 0x01000193);
  return h >>> 0;
}

export class FeatureCache {
  private readonly maxEntries: number;
  private readonly ladderCache = new Map<number, LadderCacheEntry[]>();
  private readonly areaCache = new Map<number, AreaCacheEntry[]>();
  private entryCount = 0;

  // Stats
  hits = 0;
  misses = 0;
  evictions = 0;

  constructor(maxEntries = 512) {
    this.maxEntries = maxEntries;
  }

  getLadderFeatures(
    stones: Uint8Buf,
    koPoint: number,
    playerColor: StoneColor,
    outLaddered: Uint8Buf,
    outWorkingMoves?: Uint8Buf
  ): boolean {
    const hash = hashStones(stones, koPoint, playerColor);
    const bucket = this.ladderCache.get(hash);
    if (bucket) {
      for (const entry of bucket) {
        if (entry.koPoint === koPoint && entry.playerColor === playerColor && stonesEqual(entry.stones, stones)) {
          if (outWorkingMoves && !entry.ladderWorkingMoves) {
            continue; // Need working moves but entry only has laddered stones
          }
          outLaddered.set(entry.ladderedStones);
          if (outWorkingMoves && entry.ladderWorkingMoves) {
            outWorkingMoves.set(entry.ladderWorkingMoves);
          }
          this.hits++;
          return true;
        }
      }
    }
    this.misses++;
    return false;
  }

  putLadderFeatures(
    stones: Uint8Buf,
    koPoint: number,
    playerColor: StoneColor,
    laddered: Uint8Buf,
    workingMoves?: Uint8Buf
  ): void {
    if (this.entryCount >= this.maxEntries) {
      this.evictOne();
    }

    const hash = hashStones(stones, koPoint, playerColor);
    let bucket = this.ladderCache.get(hash);
    if (!bucket) {
      bucket = [];
      this.ladderCache.set(hash, bucket);
    }

    // Check if existing
    for (const entry of bucket) {
      if (entry.koPoint === koPoint && entry.playerColor === playerColor && stonesEqual(entry.stones, stones)) {
        if (workingMoves && !entry.ladderWorkingMoves) {
          entry.ladderWorkingMoves = new Uint8Array(workingMoves);
        }
        return;
      }
    }

    const stonesCopy = new Uint8Array(stones);
    const ladderedCopy = new Uint8Array(laddered);
    const workingCopy = workingMoves ? new Uint8Array(workingMoves) : undefined;

    bucket.push({
      stones: stonesCopy,
      koPoint,
      playerColor,
      ladderedStones: ladderedCopy,
      ladderWorkingMoves: workingCopy,
    });
    this.entryCount++;
  }

  getAreaMap(stones: Uint8Buf, outArea: Uint8Buf): boolean {
    const hash = hashStones(stones, -1, 0);
    const bucket = this.areaCache.get(hash);
    if (bucket) {
      for (const entry of bucket) {
        if (stonesEqual(entry.stones, stones)) {
          outArea.set(entry.areaMap);
          this.hits++;
          return true;
        }
      }
    }
    this.misses++;
    return false;
  }

  putAreaMap(stones: Uint8Buf, areaMap: Uint8Buf): void {
    if (this.entryCount >= this.maxEntries) {
      this.evictOne();
    }

    const hash = hashStones(stones, -1, 0);
    let bucket = this.areaCache.get(hash);
    if (!bucket) {
      bucket = [];
      this.areaCache.set(hash, bucket);
    }

    for (const entry of bucket) {
      if (stonesEqual(entry.stones, stones)) {
        return;
      }
    }

    bucket.push({
      stones: new Uint8Array(stones),
      areaMap: new Uint8Array(areaMap),
    });
    this.entryCount++;
  }

  private evictOne(): void {
    this.evictions++;
    const firstLadderKey = this.ladderCache.keys().next().value;
    if (firstLadderKey !== undefined) {
      const bucket = this.ladderCache.get(firstLadderKey)!;
      bucket.shift();
      if (bucket.length === 0) {
        this.ladderCache.delete(firstLadderKey);
      }
      this.entryCount--;
      return;
    }

    const firstAreaKey = this.areaCache.keys().next().value;
    if (firstAreaKey !== undefined) {
      const bucket = this.areaCache.get(firstAreaKey)!;
      bucket.shift();
      if (bucket.length === 0) {
        this.areaCache.delete(firstAreaKey);
      }
      this.entryCount--;
    }
  }

  clear(): void {
    this.ladderCache.clear();
    this.areaCache.clear();
    this.entryCount = 0;
  }
}
