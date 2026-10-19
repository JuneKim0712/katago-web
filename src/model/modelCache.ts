import { KATAGO_MAX_CUSTOM_MODEL_BYTES } from '../worker/resourceProfile.js';

const DB_NAME = 'baduk-katago-models';
const DB_VERSION = 1;
const STORE_NAME = 'models';

type CachedModelRecord = {
  id: string;
  name: string;
  byteSize: number;
  buffer: ArrayBuffer;
  createdAt: number;
};

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Failed to open model cache'));
  });
  return dbPromise;
}

function makeModelId(file: File): string {
  return `custom-${file.name}-${file.size}-${file.lastModified}`;
}

const blobUrlById = new Map<string, string>();

export function validateCustomModelFile(file: File): string | null {
  if (!file.name.endsWith('.bin.gz') && !file.name.endsWith('.bin')) {
    return 'Model must be a .bin.gz or .bin file.';
  }
  if (file.size > KATAGO_MAX_CUSTOM_MODEL_BYTES) {
    return `Model exceeds ${Math.round(KATAGO_MAX_CUSTOM_MODEL_BYTES / 1_000_000)} MB limit.`;
  }
  if (file.size <= 0) {
    return 'Model file is empty.';
  }
  return null;
}

export async function cacheCustomModel(file: File): Promise<{ id: string; url: string; byteSize: number; name: string }> {
  const validationError = validateCustomModelFile(file);
  if (validationError) throw new Error(validationError);

  const id = makeModelId(file);
  const buffer = await file.arrayBuffer();
  const db = await openDb();

  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const record: CachedModelRecord = {
      id,
      name: file.name,
      byteSize: file.size,
      buffer,
      createdAt: Date.now(),
    };
    store.put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('Failed to cache model'));
  });

  revokeCachedModelUrl(id);
  const url = URL.createObjectURL(new Blob([buffer], { type: 'application/octet-stream' }));
  blobUrlById.set(id, url);

  return { id, url, byteSize: file.size, name: file.name };
}

/**
 * Restore a previously cached custom model from IndexedDB.
 * Creates a fresh blob URL from the stored buffer.
 * Returns null if the record no longer exists in IndexedDB.
 */
export async function restoreCustomModel(
  id: string,
): Promise<{ url: string; byteSize: number; name: string } | null> {
  try {
    const db = await openDb();
    const record = await new Promise<CachedModelRecord | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const request = tx.objectStore(STORE_NAME).get(id);
      request.onsuccess = () => resolve(request.result as CachedModelRecord | undefined);
      request.onerror = () => reject(request.error ?? new Error('Failed to read model cache'));
    });
    if (!record) return null;

    // Reuse existing blob URL if still alive; otherwise create a new one.
    let url = blobUrlById.get(id);
    if (!url) {
      url = URL.createObjectURL(new Blob([record.buffer], { type: 'application/octet-stream' }));
      blobUrlById.set(id, url);
    }
    return { url, byteSize: record.byteSize, name: record.name };
  } catch {
    return null;
  }
}

export function revokeCachedModelUrl(id: string): void {
  const existing = blobUrlById.get(id);
  if (existing) {
    URL.revokeObjectURL(existing);
    blobUrlById.delete(id);
  }
}

export function formatModelByteSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
  if (bytes >= 1_000_000) return `${Math.round(bytes / 1_000_000)} MB`;
  return `${Math.round(bytes / 1_000)} KB`;
}
