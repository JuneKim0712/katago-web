/**
 * Toggle visit-speed optimizations for A/B benchmarking and regression tests.
 * Production defaults to the optimized profile; tests can switch to legacy.
 */

export type VisitSpeedFlags = {
  /** Skip cloning ownership/policy on progress updates. */
  lightProgress: boolean;
  /** Reuse GPU input tensors instead of tf.tensor4d per batch. */
  tensorReuse: boolean;
  /** Shared prepareBatchFeatures path (deduplicated CPU prep). */
  unifiedBatchPrep: boolean;
  /** Single run() with in-search reporting instead of sliced run loops. */
  singleRunReporting: boolean;
  /** Allow position cache hits for interactive analysis. */
  interactiveCache: boolean;
  /** Looser collectBatch attempt budget for fuller GPU batches. */
  widerBatchCollection: boolean;
};

export const LEGACY_VISIT_SPEED: VisitSpeedFlags = {
  lightProgress: false,
  tensorReuse: false,
  unifiedBatchPrep: false,
  singleRunReporting: false,
  interactiveCache: false,
  widerBatchCollection: false,
};

export const OPTIMIZED_VISIT_SPEED: VisitSpeedFlags = {
  lightProgress: true,
  tensorReuse: true,
  unifiedBatchPrep: true,
  singleRunReporting: true,
  interactiveCache: true,
  widerBatchCollection: true,
};

let activeFlags: VisitSpeedFlags = { ...OPTIMIZED_VISIT_SPEED };

export function getVisitSpeedFlags(): Readonly<VisitSpeedFlags> {
  return activeFlags;
}

export function setVisitSpeedFlags(flags: Partial<VisitSpeedFlags>): void {
  activeFlags = { ...activeFlags, ...flags };
}

export function resetVisitSpeedFlags(): void {
  activeFlags = { ...OPTIMIZED_VISIT_SPEED };
}

export function applyVisitSpeedProfile(profile: 'legacy' | 'optimized'): void {
  activeFlags = profile === 'legacy' ? { ...LEGACY_VISIT_SPEED } : { ...OPTIMIZED_VISIT_SPEED };
}

export async function withVisitSpeedProfile<T>(
  profile: 'legacy' | 'optimized',
  fn: () => T | Promise<T>
): Promise<T> {
  const prev = { ...activeFlags };
  applyVisitSpeedProfile(profile);
  try {
    return await fn();
  } finally {
    activeFlags = prev;
  }
}
