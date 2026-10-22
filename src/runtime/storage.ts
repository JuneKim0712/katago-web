/**
 * Model artifact caching and persistent file storage for KataGo Web.
 * Uses Browser Cache API as primary storage with IndexedDB fallback.
 */

const CACHE_NAME = 'katago-web-models-v1';
const DB_NAME = 'katago-web-storage';
const DB_VERSION = 1;
const STORE_NAME = 'models';

function openIndexedDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB is not available in this environment'));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('Failed to open IndexedDB'));
  });
}

export async function getCachedModelArtifact(sha256: string): Promise<Uint8Array | null> {
  // 1. Try Browser Cache API
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(CACHE_NAME);
      const resp = await cache.match(`/katago-models/${sha256}`);
      if (resp && resp.ok) {
        const buf = await resp.arrayBuffer();
        return new Uint8Array(buf);
      }
    } catch {
      // Fall through to IndexedDB
    }
  }

  // 2. Try IndexedDB
  if (typeof indexedDB !== 'undefined') {
    try {
      const db = await openIndexedDb();
      return await new Promise<Uint8Array | null>((resolve) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(sha256);
        req.onsuccess = () => {
          if (req.result instanceof Uint8Array) {
            resolve(req.result);
          } else if (req.result instanceof ArrayBuffer) {
            resolve(new Uint8Array(req.result));
          } else {
            resolve(null);
          }
        };
        req.onerror = () => resolve(null);
      });
    } catch {
      return null;
    }
  }

  return null;
}

export async function putCachedModelArtifact(sha256: string, bytes: Uint8Array): Promise<void> {
  // 1. Try Browser Cache API
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(CACHE_NAME);
      const resp = new Response(bytes as unknown as BodyInit, {
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(bytes.byteLength),
          'X-KataGo-Sha256': sha256,
        },
      });
      await cache.put(`/katago-models/${sha256}`, resp);
      return;
    } catch {
      // Fall through to IndexedDB
    }
  }

  // 2. Try IndexedDB
  if (typeof indexedDB !== 'undefined') {
    try {
      const db = await openIndexedDb();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const req = store.put(bytes, sha256);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    } catch {
      // Ignore caching failures gracefully
    }
  }
}

export async function isOpfsAvailable(): Promise<boolean> {
  if (typeof navigator !== 'undefined' && navigator.storage && typeof navigator.storage.getDirectory === 'function') {
    try {
      const root = await navigator.storage.getDirectory();
      return !!root;
    } catch {
      return false;
    }
  }
  return false;
}
