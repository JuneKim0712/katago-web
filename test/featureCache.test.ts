import { describe, expect, it } from 'vitest';
import { FeatureCache } from '../src/engine/features/cache.js';
import { BLACK, WHITE } from '../src/engine/fastBoard.js';

describe('FeatureCache', () => {
  it('stores and retrieves ladder features with defensive copying', () => {
    const cache = new FeatureCache(10);
    const stones = new Uint8Array(361);
    stones[10] = BLACK;
    const laddered = new Uint8Array(361);
    laddered[10] = 1;
    const working = new Uint8Array(361);
    working[11] = 1;

    cache.putLadderFeatures(stones, -1, BLACK, laddered, working);

    // Mutate source arrays
    stones[10] = WHITE;
    laddered[10] = 0;
    working[11] = 0;

    const outLaddered = new Uint8Array(361);
    const outWorking = new Uint8Array(361);

    // Original position lookup
    const origStones = new Uint8Array(361);
    origStones[10] = BLACK;
    const hit = cache.getLadderFeatures(origStones, -1, BLACK, outLaddered, outWorking);
    expect(hit).toBe(true);
    expect(outLaddered[10]).toBe(1);
    expect(outWorking[11]).toBe(1);

    // Different player should miss
    expect(cache.getLadderFeatures(origStones, -1, WHITE, outLaddered, outWorking)).toBe(false);

    // Different ko should miss
    expect(cache.getLadderFeatures(origStones, 5, BLACK, outLaddered, outWorking)).toBe(false);
  });

  it('bounds capacity and evicts LRU entries', () => {
    const cache = new FeatureCache(3);
    for (let i = 0; i < 5; i++) {
      const stones = new Uint8Array(361);
      stones[i] = BLACK;
      const laddered = new Uint8Array(361);
      cache.putLadderFeatures(stones, -1, BLACK, laddered);
    }
    expect(cache.evictions).toBe(2);
  });
});
