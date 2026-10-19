import type { KataGoBackendPreference } from '../types/index.js';
import type { ParsedKataGoModelV8 } from '../model/modelV8.js';

export const KATAGO_MAX_CUSTOM_MODEL_BYTES = 1_930_000_000;

export type ModelTier = 'small' | 'medium' | 'large' | 'xlarge';

export type ResourceProfile = {
  modelByteSize: number;
  modelName: string;
  numBlocks: number;
  trunkNumChannels: number;
  estimatedWeightBytes: number;
  tier: ModelTier;
  maxMctsBatchSize: number;
  maxRootSymmetrySamples: number;
  maxEvalBatchSize: number;
  defaultOwnershipMode: 'none' | 'root';
  /** How often (ms) to stream progress updates. Smaller = more responsive but more React renders. */
  progressIntervalMs: number;
};

export type ResourceProfilePayload = ResourceProfile & {
  backend: string;
  effectiveMctsBatchSize: number;
};

// ── Architecture-based complexity ─────────────────────────────────────────────

/**
 * Returns a FLOPs-proportional complexity score for the model's residual trunk.
 * Each residual block has roughly O(channels²) multiply-adds on the 19×19 spatial
 * map, so total cost scales with numBlocks × channels².
 *
 * Representative values for common KataGo model families:
 *   b6c96   →  55   (tiny eval/training net)
 *   b10c128 → 164   (small)
 *   b15c192 → 553   (medium)
 *   b20c256 → 1311  (large / b20c256 kata1)
 *   b40c256 → 2621  (xlarge / b40c256 kata1)
 */
export function computeArchComplexity(numBlocks: number, trunkNumChannels: number): number {
  return (numBlocks * trunkNumChannels * trunkNumChannels) / 1_000;
}

/**
 * Classify a model by architecture complexity (more accurate than file-size tiers).
 * Thresholds chosen so common KataGo model families map to the expected tier:
 *   small  : b6c96, b10c96, b10c128
 *   medium : b15c128, b15c192
 *   large  : b20c192, b20c256, b28c256
 *   xlarge : b40c256 and heavier
 */
export function computeModelTierFromArch(complexity: number): ModelTier {
  if (complexity >= 1_800) return 'xlarge';
  if (complexity >= 500)   return 'large';
  if (complexity >= 150)   return 'medium';
  return 'small';
}

/** File-size based tier — used where arch info is unavailable (e.g. before parsing). */
export function computeModelTier(modelByteSize: number): ModelTier {
  if (modelByteSize > 300_000_000) return 'xlarge';
  if (modelByteSize > 100_000_000) return 'large';
  if (modelByteSize > 40_000_000)  return 'medium';
  return 'small';
}

export function estimateWeightBytes(parsed: ParsedKataGoModelV8): number {
  return Math.max(parsed.trunk.numBlocks * parsed.trunk.trunkNumChannels * 10_000, 0);
}

export function buildResourceProfile(args: {
  modelByteSize: number;
  parsed: ParsedKataGoModelV8;
  backend: string;
}): ResourceProfile {
  // Use architecture-based tier — more accurate than byte-size alone.
  const complexity = computeArchComplexity(
    args.parsed.trunk.numBlocks,
    args.parsed.trunk.trunkNumChannels,
  );
  const tier = computeModelTierFromArch(complexity);

  const isWebGpu = args.backend === 'webgpu';
  const isWasm   = args.backend === 'wasm';

  let maxMctsBatchSize      = 4;
  let maxRootSymmetrySamples = 1;
  let maxEvalBatchSize      = 4;
  let defaultOwnershipMode: 'none' | 'root' = 'root';

  if (isWebGpu) {
    // These are the probe ceilings — the real batch size is found by
    // `probeMctsBatchSize` which tries descending from this value and caches
    // the best result.  Values are intentionally 2× the old hard-coded defaults
    // so modern GPUs get a meaningful throughput gain while weak GPUs fall back
    // gracefully via the probe (they cached 16/4/1/1 before and still will).
    switch (tier) {
      case 'small':
        maxMctsBatchSize       = 32;  // b6c96 / b10c128
        maxRootSymmetrySamples = 8;
        maxEvalBatchSize       = 32;
        break;
      case 'medium':
        maxMctsBatchSize       = 16;  // b15c192
        maxRootSymmetrySamples = 4;
        maxEvalBatchSize       = 16;
        break;
      case 'large':
        maxMctsBatchSize       = 8;   // b20c256 / b28c256
        maxRootSymmetrySamples = 2;
        maxEvalBatchSize       = 8;
        break;
      case 'xlarge':
        maxMctsBatchSize       = 4;   // b40c256+
        maxRootSymmetrySamples = 1;
        maxEvalBatchSize       = 4;
        defaultOwnershipMode   = 'none';
        break;
    }
  } else if (isWasm) {
    // WASM is not probed (no GPU crash risk); these values go straight to use.
    // Raised modestly from the previous defaults for better XNNPACK vectorisation.
    switch (tier) {
      case 'small':
        maxMctsBatchSize       = 16;
        maxRootSymmetrySamples = 4;
        maxEvalBatchSize       = 16;
        break;
      case 'medium':
        maxMctsBatchSize       = 8;
        maxRootSymmetrySamples = 2;
        maxEvalBatchSize       = 12;
        break;
      case 'large':
        maxMctsBatchSize       = 4;
        maxRootSymmetrySamples = 1;
        maxEvalBatchSize       = 6;
        defaultOwnershipMode   = 'none';
        break;
      case 'xlarge':
        maxMctsBatchSize       = 2;
        maxRootSymmetrySamples = 1;
        maxEvalBatchSize       = 2;
        defaultOwnershipMode   = 'none';
        break;
    }
  } else {
    // CPU fallback
    switch (tier) {
      case 'small':
        maxMctsBatchSize = 4;
        maxEvalBatchSize = 4;
        break;
      default:
        maxMctsBatchSize     = 1;
        maxEvalBatchSize     = 1;
        defaultOwnershipMode = 'none';
        break;
    }
  }

  // Faster progress callbacks for lighter models → smoother live UI updates.
  const progressIntervalMs =
    tier === 'small' ? 100 :
    tier === 'medium' ? 200 :
    400;

  return {
    modelByteSize: args.modelByteSize,
    modelName: args.parsed.modelName,
    numBlocks: args.parsed.trunk.numBlocks,
    trunkNumChannels: args.parsed.trunk.trunkNumChannels,
    estimatedWeightBytes: estimateWeightBytes(args.parsed),
    tier,
    maxMctsBatchSize,
    maxRootSymmetrySamples,
    maxEvalBatchSize,
    defaultOwnershipMode,
    progressIntervalMs,
  };
}

export function getAnalyzeTuning(args: {
  profile: ResourceProfile;
  backend: string;
  probedMctsBatchSize: number | null;
  msgBatchSize?: number;
  msgOwnershipMode?: 'none' | 'root' | 'tree';
  includeMovesOwnership?: boolean;
  nnRandomize: boolean;
}): {
  batchSize: number;
  rootSymmetrySamples: number;
  ownershipMode: 'none' | 'root' | 'tree';
} {
  const maxBatch = args.probedMctsBatchSize ?? args.profile.maxMctsBatchSize;
  const batchSize = Math.max(1, Math.min(args.msgBatchSize ?? maxBatch, 64));

  let rootSymmetrySamples = 1;
  if (args.backend === 'webgpu' && args.nnRandomize) {
    rootSymmetrySamples = args.profile.maxRootSymmetrySamples;
  } else if (args.backend === 'wasm' && args.nnRandomize) {
    // WASM can afford up to 4 symmetry samples for small models on a multi-core CPU.
    rootSymmetrySamples = Math.min(4, args.profile.maxRootSymmetrySamples);
  }

  if (args.includeMovesOwnership) {
    return { batchSize, rootSymmetrySamples, ownershipMode: 'tree' };
  }

  let ownershipMode = args.msgOwnershipMode ?? args.profile.defaultOwnershipMode;
  if (args.profile.tier === 'large' || args.profile.tier === 'xlarge') {
    ownershipMode = ownershipMode === 'tree' ? 'root' : ownershipMode === 'root' ? 'root' : 'none';
    if (args.profile.defaultOwnershipMode === 'none') ownershipMode = 'none';
  }

  return { batchSize, rootSymmetrySamples, ownershipMode };
}

export function toProfilePayload(
  profile: ResourceProfile,
  backend: string,
  effectiveMctsBatchSize: number
): ResourceProfilePayload {
  return { ...profile, backend, effectiveMctsBatchSize };
}

const BATCH_CACHE_KEY = 'katago-probed-batch-v1';

export function loadCachedBatchSize(modelName: string, backend: string): number | null {
  try {
    const raw = localStorage.getItem(BATCH_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Record<string, number>;
    return parsed[`${modelName}:${backend}`] ?? null;
  } catch {
    return null;
  }
}

export function saveCachedBatchSize(modelName: string, backend: string, batchSize: number): void {
  try {
    const raw = localStorage.getItem(BATCH_CACHE_KEY);
    const parsed = raw ? (JSON.parse(raw) as Record<string, number>) : {};
    parsed[`${modelName}:${backend}`] = batchSize;
    localStorage.setItem(BATCH_CACHE_KEY, JSON.stringify(parsed));
  } catch {
    // Ignore quota / private-mode errors.
  }
}

/**
 * Generate a probe sequence descending from `max` in a geometric series
 * (each step ≈ 75 % of the previous), so we find the true GPU limit with
 * fine granularity rather than jumping coarsely from max to 4 to 2 to 1.
 *
 * Example for max = 64:  64 → 48 → 36 → 27 → 20 → 15 → 11 → 8 → 6 → 4 → 3 → 2 → 1
 */
export function generateProbeCandidates(max: number): number[] {
  const seen = new Set<number>();
  const add = (v: number) => { const n = Math.max(1, Math.round(v)); seen.add(n); };

  add(max);
  let v = max;
  while (v > 1) {
    v = v * 0.75;
    add(v);
  }
  add(1);

  return Array.from(seen).sort((a, b) => b - a); // descending
}

export function recommendBackendForModelSize(
  modelByteSize: number,
  preference: KataGoBackendPreference | 'auto' = 'auto'
): KataGoBackendPreference {
  if (preference !== 'auto') return preference;
  const tier = computeModelTier(modelByteSize);
  if (tier === 'xlarge' || tier === 'large') return 'wasm';
  return 'webgpu';
}
