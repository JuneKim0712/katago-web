export function getWorkerConstructor(): typeof Worker | null {
  try {
    const workerConstructor = globalThis.Worker;
    return typeof workerConstructor === 'function' ? workerConstructor : null;
  } catch {
    return null;
  }
}

export function publicUrl(path: string): string {
  const base = typeof process !== 'undefined' && process.env?.BASE_URL ? process.env.BASE_URL : '/';
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  const trimmedPath = path.startsWith('/') ? path.slice(1) : path;
  return `${normalizedBase}${trimmedPath}`;
}
