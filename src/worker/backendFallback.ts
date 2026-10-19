import type { KataGoBackendPreference } from '../types/index.js';

export type KataGoModelLoadStage = 'fetch' | 'parse' | 'warmup' | 'runtime';

export const normalizeKataGoBackendPreference = (
  backend?: KataGoBackendPreference | null
): KataGoBackendPreference => (backend === 'wasm' || backend === 'cpu' ? backend : 'webgpu');

export function getKataGoFallbackBackend(args: {
  requestedBackend: KataGoBackendPreference;
  activeBackend: string | null | undefined;
}): KataGoBackendPreference | null {
  const activeBackend = args.activeBackend?.trim().toLowerCase();
  if (args.requestedBackend === 'webgpu' && activeBackend === 'webgpu') return 'wasm';
  if (args.requestedBackend !== 'cpu' && activeBackend === 'wasm') return 'cpu';
  return null;
}

export function getKataGoWarmupFallbackBackend(args: {
  requestedBackend: KataGoBackendPreference;
  activeBackend: string | null | undefined;
  stage: KataGoModelLoadStage;
}): KataGoBackendPreference | null {
  if (args.stage !== 'warmup' && args.stage !== 'runtime') return null;
  return getKataGoFallbackBackend(args);
}

export function shouldCacheKataGoFallbackForRequest(args: {
  requestedBackend: KataGoBackendPreference;
  fallbackBackend: string | null | undefined;
}): boolean {
  const fallbackBackend = args.fallbackBackend?.trim().toLowerCase();
  return !!fallbackBackend && fallbackBackend !== args.requestedBackend;
}
