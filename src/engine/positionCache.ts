/**
 * § 1.4  Opening / position cache backed by IndexedDB.
 *
 * Before each MCTS search the worker checks whether a good-enough result is
 * already stored for this (board, player, komi, rules, model) combination.
 * After every completed search the result is written back.
 *
 * Cache key  : FNV-1a hash of the board stone array XOR'd with player/komi/rules,
 *              prefixed with the model name so model upgrades auto-partition.
 * Eviction   : LRU, capped at MAX_ENTRIES (500 positions).
 * Invalidation: Changing the model URL purges the old model's partition lazily
 *              on the next write cycle.
 */

const DB_NAME = 'katago-position-cache';
const STORE = 'positions';
const DB_VERSION = 1;
const MAX_ENTRIES = 500;

export type CachedPositionResult = {
  rootWinRate: number;
  rootScoreLead: number;
  rootScoreSelfplay: number;
  rootVisits: number;
};

type CacheEntry = {
  /** Composite key: "<modelName>:<hash>" */
  key: string;
  modelName: string;
  result: CachedPositionResult;
  visits: number;
  /** Unix ms — used for LRU eviction */
  lastUsed: number;
};

// ─── DB plumbing ────────────────────────────────────────────────────────────

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'key' });
        store.createIndex('modelName', 'modelName', { unique: false });
        store.createIndex('lastUsed', 'lastUsed', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => { dbPromise = null; reject(req.error); };
  });
  return dbPromise;
}

function idbGet(db: IDBDatabase, key: string): Promise<CacheEntry | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result as CacheEntry | undefined);
    req.onerror = () => reject(req.error);
  });
}

function idbPut(db: IDBDatabase, entry: CacheEntry): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function idbCountForModel(db: IDBDatabase, modelName: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).index('modelName').count(IDBKeyRange.only(modelName));
    req.onsuccess = () => resolve(req.result as number);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Delete the `toDelete` oldest entries for this model (by lastUsed ascending).
 */
function idbEvictOldest(db: IDBDatabase, modelName: string, toDelete: number): Promise<void> {
  if (toDelete <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    const index = store.index('lastUsed');
    // Index is on lastUsed only; filter by modelName inside the cursor handler.
    const req = index.openCursor();
    let deleted = 0;
    req.onsuccess = () => {
      const cursor = req.result as IDBCursorWithValue | null;
      if (!cursor || deleted >= toDelete) { return; }
      const entry = cursor.value as CacheEntry;
      if (entry.modelName === modelName) {
        cursor.delete();
        deleted++;
      }
      cursor.continue();
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ─── Hash ────────────────────────────────────────────────────────────────────

/**
 * FNV-1a 32-bit hash over the flat stone array, mixed with player/komi/rules.
 * Returns a base-36 string for compact keys.
 */
function hashPosition(
  stones: ArrayLike<number>,
  currentPlayer: 'black' | 'white',
  komi: number,
  rules: string,
): string {
  const FNV_PRIME = 16777619;
  const FNV_OFFSET = 2166136261;
  let h = FNV_OFFSET >>> 0;
  for (let i = 0; i < stones.length; i++) {
    h = (h ^ (stones[i]! & 0xff)) >>> 0;
    h = Math.imul(h, FNV_PRIME) >>> 0;
  }
  // Mix player
  h = (h ^ (currentPlayer === 'black' ? 0x01 : 0x02)) >>> 0;
  h = Math.imul(h, FNV_PRIME) >>> 0;
  // Mix komi (0.5 resolution)
  const komiFixed = Math.round(komi * 2) & 0xffff;
  h = (h ^ (komiFixed & 0xff)) >>> 0;
  h = Math.imul(h, FNV_PRIME) >>> 0;
  h = (h ^ ((komiFixed >> 8) & 0xff)) >>> 0;
  h = Math.imul(h, FNV_PRIME) >>> 0;
  // Mix rules (first 4 chars)
  for (let i = 0; i < Math.min(rules.length, 4); i++) {
    h = (h ^ rules.charCodeAt(i)) >>> 0;
    h = Math.imul(h, FNV_PRIME) >>> 0;
  }
  return h.toString(36);
}

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * Build the composite cache key for this position.
 * Stones should be a flat Uint8Array (BOARD_SIZE × BOARD_SIZE) as produced
 * by `boardStateToStones`.
 */
export function makeCacheKey(
  modelName: string,
  stones: ArrayLike<number>,
  currentPlayer: 'black' | 'white',
  komi: number,
  rules: string,
): string {
  return `${modelName}:${hashPosition(stones, currentPlayer, komi, rules)}`;
}

/**
 * Look up a cached result.  Returns `null` if the cache has no entry or if
 * the cached entry has fewer visits than `minVisits`.
 * Silently ignores IndexedDB errors (cache is best-effort).
 */
export async function getCachedPosition(
  key: string,
  minVisits: number,
): Promise<CachedPositionResult | null> {
  try {
    const db = await openDb();
    const entry = await idbGet(db, key);
    if (!entry || entry.visits < minVisits) return null;
    // Touch lastUsed in the background (fire-and-forget, don't block the caller)
    idbPut(db, { ...entry, lastUsed: Date.now() }).catch(() => {});
    return entry.result;
  } catch {
    return null;
  }
}

/**
 * Store a completed analysis result in the cache.
 * Runs eviction in the background if the per-model entry count exceeds MAX_ENTRIES.
 * Silently ignores errors.
 */
export async function setCachedPosition(
  key: string,
  modelName: string,
  result: CachedPositionResult,
  visits: number,
): Promise<void> {
  try {
    const db = await openDb();
    const entry: CacheEntry = { key, modelName, result, visits, lastUsed: Date.now() };
    await idbPut(db, entry);
    // Async eviction — don't await so the caller isn't delayed
    idbCountForModel(db, modelName).then(count => {
      const excess = count - MAX_ENTRIES;
      if (excess > 0) idbEvictOldest(db, modelName, excess + 50).catch(() => {});
    }).catch(() => {});
  } catch {
    // Cache is best-effort; ignore failures silently.
  }
}
