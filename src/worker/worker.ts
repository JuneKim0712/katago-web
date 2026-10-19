/// <reference lib="webworker" />

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import { setThreadsCount, setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import pako from 'pako';

import type { KataGoAnalyzeRequest, KataGoWorkerRequest, KataGoWorkerResponse } from '../engine/types.js';
import type { BoardState, GameRules, KataGoBackendPreference } from '../types/index.js';
import { publicUrl } from '../utils/browserWorker.js';
import { getAnimationNow } from '../utils/helpers.js';
import { parseKataGoModelV8 } from '../model/loadModelV8.js';
import { KataGoModelV8Tf } from '../model/modelV8.js';
import { ENGINE_MAX_TIME_MS, ENGINE_MAX_VISITS } from '../engine/limits.js';
import { MctsSearch, type OwnershipMode } from '../engine/analyzeMcts.js';
import {
  getKataGoFallbackBackend,
  getKataGoWarmupFallbackBackend,
  normalizeKataGoBackendPreference,
  shouldCacheKataGoFallbackForRequest,
} from './backendFallback.js';
import { isGpuDeviceLostError } from './gpuError';
import {
  buildResourceProfile,
  generateProbeCandidates,
  getAnalyzeTuning,
  loadCachedBatchSize,
  recommendBackendForModelSize,
  saveCachedBatchSize,
  toProfilePayload,
  type ResourceProfile,
} from './resourceProfile.js';
import {
  BLACK,
  BOARD_AREA,
  BOARD_SIZE,
  WHITE,
  setBoardSize,
} from '../engine/fastBoard.js';
import { makeCacheKey, getCachedPosition, setCachedPosition } from '../engine/positionCache.js';
import { getVisitSpeedFlags } from '../engine/visitSpeedConfig.js';

let model: KataGoModelV8Tf | null = null;
let loadedModelUrl: string | null = null;
let loadedModelName: string | undefined;
let resourceProfile: ResourceProfile | null = null;
let probedMctsBatchSize: number | null = null;
let backendPromise: Promise<void> | null = null;
let backendPreference: KataGoBackendPreference | null = null;
let prodModeEnabled = false;
let queue: Promise<void> = Promise.resolve();

let scratchBoardSize = BOARD_SIZE;
type ParsedKataGoModelV8 = ReturnType<typeof parseKataGoModelV8>;

/** Convert a 2-D BoardState to a flat Uint8Array suitable for cache-key hashing. */
function boardToFlatForHash(board: BoardState): Uint8Array {
  const stones = new Uint8Array(BOARD_AREA);
  for (let y = 0; y < board.length && y < BOARD_SIZE; y++) {
    const row = board[y];
    if (!row) continue;
    for (let x = 0; x < row.length && x < BOARD_SIZE; x++) {
      const v = row[x];
      if (v === 'black') stones[y * BOARD_SIZE + x] = BLACK;
      else if (v === 'white') stones[y * BOARD_SIZE + x] = WHITE;
    }
  }
  return stones;
}

let search: MctsSearch | null = null;
const latestAnalyzeByGroup = new Map<string, number>();
let interactiveToken = 0;
const analyzeMeta = new WeakMap<KataGoAnalyzeRequest, { analysisGroup: 'interactive' | 'background'; interactiveToken: number }>();

function ensureBoardSizeForWorker(boardSize: number): void {
  if (boardSize === scratchBoardSize) return;
  setBoardSize(boardSize);
  scratchBoardSize = BOARD_SIZE;
  search = null;
}

async function initWasmBackend(): Promise<void> {
  try {
    // Vite serves `public/` at the site root.
    setWasmPaths(publicUrl('tfjs/'));
    // Use a reasonable thread count for XNNPACK when cross-origin isolated (SharedArrayBuffer).
    // Without COOP/COEP headers, browsers disable threads and TFJS will fall back to single-threaded wasm.
    const isCrossOriginIsolated = (globalThis as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    if (isCrossOriginIsolated) {
      const hc = (globalThis as unknown as { navigator?: { hardwareConcurrency?: number } }).navigator?.hardwareConcurrency ?? 1;
      const numThreads = Math.max(1, Math.min(8, Math.floor(hc)));
      setThreadsCount(numThreads);
    }
    await tf.setBackend('wasm');
    await tf.ready();
    return;
  } catch {
    // Fall through to CPU below.
  }

  await tf.setBackend('cpu');
  await tf.ready();
}

async function initBackend(preferredBackend: KataGoBackendPreference): Promise<void> {
  if (preferredBackend === 'cpu') {
    await tf.setBackend('cpu');
    await tf.ready();
    return;
  }

  if (preferredBackend === 'webgpu') {
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
      return;
    } catch {
      // Fall back to WASM/CPU if WebGPU isn't available or fails to initialize.
    }
  }

  await initWasmBackend();
}

function maybeUngzip(data: Uint8Array): Uint8Array {
  // gzip magic bytes 0x1f8b
  if (data.length >= 2 && data[0] === 0x1f && data[1] === 0x8b) return pako.ungzip(data);
  return data;
}

async function yieldToGpu(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function disposeModelSafely(): Promise<void> {
  search = null;
  model?.dispose();
  model = null;
  loadedModelName = undefined;
  loadedModelUrl = null;
  resourceProfile = null;
  probedMctsBatchSize = null;
  await yieldToGpu();
}

function postLoadProgress(stage: 'fetch' | 'parse' | 'warmup' | 'probe', percent: number): void {
  post({ type: 'katago:load_progress', stage, percent });
}

function postResourceProfile(effectiveMctsBatchSize: number): void {
  if (!resourceProfile) return;
  post({
    type: 'katago:profile',
    profile: toProfilePayload(resourceProfile, tf.getBackend(), effectiveMctsBatchSize),
  });
}

async function ensureBackend(backend?: KataGoBackendPreference): Promise<void> {
  const preferredBackend = normalizeKataGoBackendPreference(backend);
  if (backendPromise && backendPreference === preferredBackend) {
    await backendPromise;
    return;
  }

  await disposeModelSafely();

  backendPreference = preferredBackend;
  backendPromise = initBackend(preferredBackend)
      .then(() => {
        if (!prodModeEnabled) {
          tf.enableProdMode();
          prodModeEnabled = true;
        }
      })
      .catch((err) => {
        backendPromise = null;
        backendPreference = null;
        throw err;
      });
  await backendPromise;
}

async function runForwardProbe(candidate: KataGoModelV8Tf, batch: number): Promise<void> {
  const spatial = tf.zeros([batch, 19, 19, 22], 'float32') as tf.Tensor4D;
  const global = tf.zeros([batch, 19], 'float32') as tf.Tensor2D;
  let out: ReturnType<KataGoModelV8Tf['forwardValueOnly']> | null = null;
  try {
    out = candidate.forwardValueOnly(spatial, global);
    const results = await Promise.allSettled([out.value.data(), out.scoreValue.data()]);
    for (const result of results) {
      if (result.status === 'rejected') throw result.reason;
    }
  } finally {
    spatial.dispose();
    global.dispose();
    out?.value.dispose();
    out?.scoreValue.dispose();
  }
}

async function warmupModel(candidate: KataGoModelV8Tf): Promise<void> {
  await runForwardProbe(candidate, 1);
}

/**
 * Time a forward pass at `batch` and return throughput (positions / ms).
 *
 * Uses `performance.now()` with a minimum-of-N strategy: the fastest run is the
 * truest signal because slower runs only add OS/scheduler jitter.  `data()`
 * already awaits GPU completion, so each timed span covers the full dispatch +
 * readback — no manual GPU-timestamp plumbing is needed (and the previous
 * timestamp-query path measured an empty command encoder, not the TF.js work).
 */
async function measureThroughput(candidate: KataGoModelV8Tf, batch: number, timedRuns = 3): Promise<number> {
  // Warmup: prime GPU caches / JIT before timing.
  await runForwardProbe(candidate, batch);
  await runForwardProbe(candidate, batch);

  let bestMs = Infinity;
  for (let run = 0; run < timedRuns; run++) {
    const t0 = getAnimationNow();
    await runForwardProbe(candidate, batch);
    bestMs = Math.min(bestMs, getAnimationNow() - t0);
  }
  return batch / Math.max(bestMs, 0.01);
}

async function probeMctsBatchSize(candidate: KataGoModelV8Tf, profile: ResourceProfile): Promise<number> {
  const backend = tf.getBackend();

  // Return the cached value immediately — no probe needed.
  const cached = loadCachedBatchSize(profile.modelName, backend);
  if (cached !== null) {
    const effective = Math.max(1, Math.min(cached, profile.maxMctsBatchSize));
    klog('probe', `Batch size from cache: ${effective} (model: ${profile.modelName}, backend: ${backend})`);
    return effective;
  }

  // WASM / CPU never crash on oversized batches so skip the safety probe entirely.
  // Use the profile max directly and cache it.
  if (backend !== 'webgpu') {
    klog('probe', `WASM/CPU backend — skipping probe, using max: ${profile.maxMctsBatchSize}`);
    saveCachedBatchSize(profile.modelName, backend, profile.maxMctsBatchSize);
    return profile.maxMctsBatchSize;
  }

  // ── Phase 1: safety probe ─────────────────────────────────────────────────
  // Descend through a dense geometric series until we find a batch that
  // doesn't trigger a GPU device-lost error.
  const safetyCandidates = generateProbeCandidates(profile.maxMctsBatchSize);
  klog('probe', `WebGPU safety probe — ceiling: ${profile.maxMctsBatchSize}, candidates: [${safetyCandidates.join(', ')}]`);
  let safeMax = 1;
  let safeProbeMs = 0;
  for (const batch of safetyCandidates) {
    const probeT0 = getAnimationNow();
    try {
      await runForwardProbe(candidate, batch);
      safeProbeMs = Math.round(getAnimationNow() - probeT0);
      klog('probe', `  batch=${batch} → OK (${safeProbeMs}ms) — safe ceiling found`);
      safeMax = batch;
      break; // found the highest safe batch
    } catch (err) {
      kwarn('probe', `  batch=${batch} → GPU crash — trying smaller…`);
      if (!isGpuDeviceLostError(err)) throw err;
    }
  }

  // ── Phase 2: throughput timing sweep ─────────────────────────────────────
  // Among safe candidates, find the batch size with the best positions/ms.
  // This catches the case where safeMax > the GPU's compute-saturation point
  // (e.g. memory-bandwidth-bound regime at very large batches on some GPUs).
  //
  // Skip timing for large/xlarge models: each forward pass already takes
  // hundreds of ms, so the sweep would add unacceptable load time.
  let bestBatch = safeMax;

  // A slow safe probe means the device is already at its limit. Timing five more
  // batch sizes here delays the first win rate by minutes and rarely changes the pick.
  if ((profile.tier === 'small' || profile.tier === 'medium') && safeProbeMs <= 1_500) {
    // Build 3-4 timing candidates spread geometrically below safeMax.
    const timingSet = new Set<number>();
    timingSet.add(safeMax);
    if (safeMax >= 4)  timingSet.add(Math.max(1, Math.round(safeMax / 2)));
    if (safeMax >= 8)  timingSet.add(Math.max(1, Math.round(safeMax / 4)));
    if (safeMax >= 16) timingSet.add(Math.max(1, Math.round(safeMax / 8)));
    timingSet.add(1);

    const timingCandidates = Array.from(timingSet).sort((a, b) => b - a);
    let bestThroughput = 0;
    klog('probe', `Throughput sweep — candidates: [${timingCandidates.join(', ')}]`);

    for (const batch of timingCandidates) {
      try {
        const tput = await measureThroughput(candidate, batch);
        klog('probe', `  batch=${batch} → ${tput.toFixed(2)} pos/ms${tput > bestThroughput ? ' ← best so far' : ''}`);
        if (tput > bestThroughput) {
          bestThroughput = tput;
          bestBatch = batch;
        }
      } catch {
        kwarn('probe', `  batch=${batch} → timing measurement failed, skipping`);
      }
    }
    klog('probe', `Best batch: ${bestBatch} @ ${bestThroughput.toFixed(2)} pos/ms — saving to cache`);
  } else {
    const why = safeProbeMs > 1_500 ? `probe took ${safeProbeMs}ms` : `${profile.tier} model`;
    klog('probe', `Skipping throughput sweep (${why}), using safe max: ${bestBatch}`);
  }

  saveCachedBatchSize(profile.modelName, backend, bestBatch);
  return bestBatch;
}

async function createWarmedModel(parsed: ParsedKataGoModelV8): Promise<KataGoModelV8Tf> {
  const candidate = new KataGoModelV8Tf(parsed, { fuseRotary: true, fuseRmsNorm: true });
  try {
    await warmupModel(candidate);
    return candidate;
  } catch (err) {
    candidate.dispose();
    throw err;
  }
}

function installModel(nextModel: KataGoModelV8Tf, parsed: ParsedKataGoModelV8, modelUrl: string, byteSize: number): void {
  model?.dispose();
  model = nextModel;
  loadedModelName = parsed.modelName;
  loadedModelUrl = modelUrl;
  search = null;
  resourceProfile = buildResourceProfile({
    modelByteSize: byteSize,
    parsed,
    backend: tf.getBackend(),
  });
  klog('init', `Model installed: ${resourceProfile.modelName} | tier: ${resourceProfile.tier} | backend: ${tf.getBackend()} | batch ceiling: ${resourceProfile.maxMctsBatchSize} | ownership: ${resourceProfile.defaultOwnershipMode}`);
}

async function switchToFallbackBackendForRequest(
  requestedBackend: KataGoBackendPreference,
  fallbackBackend: KataGoBackendPreference
): Promise<void> {
  backendPromise = null;
  backendPreference = null;
  await ensureBackend(fallbackBackend);
  if (shouldCacheKataGoFallbackForRequest({ requestedBackend, fallbackBackend: tf.getBackend() })) {
    backendPreference = requestedBackend;
  }
}

async function loadModelFromBuffer(
  buf: Uint8Array,
  modelUrl: string,
  requestedBackend: KataGoBackendPreference
): Promise<void> {
  postLoadProgress('parse', 40);
  const parseT0 = getAnimationNow();
  const data = maybeUngzip(buf);
  const parsed = parseKataGoModelV8(data);
  klog('init', `Parsed: ${parsed.modelName} | ${parsed.trunk.numBlocks}b × ${parsed.trunk.trunkNumChannels}ch | ${Math.round(getAnimationNow() - parseT0)}ms`);
  postLoadProgress('warmup', 60);

  const attemptedFallbacks = new Set<KataGoBackendPreference>();
  while (true) {
    try {
      klog('init', `Warming up on backend: ${tf.getBackend()} …`);
      const warmT0 = getAnimationNow();
      const warmed = await createWarmedModel(parsed);
      klog('init', `Warmup OK (${Math.round(getAnimationNow() - warmT0)}ms)`);
      installModel(warmed, parsed, modelUrl, buf.length);
      postLoadProgress('probe', 75); // safety probe starting
      probedMctsBatchSize = await probeMctsBatchSize(warmed, resourceProfile!);
      postLoadProgress('probe', 100); // timing sweep done
      postResourceProfile(probedMctsBatchSize);
      return;
    } catch (err) {
      const fallbackBackend = getKataGoWarmupFallbackBackend({
        requestedBackend,
        activeBackend: tf.getBackend(),
        stage: 'warmup',
      });
      if (!fallbackBackend || attemptedFallbacks.has(fallbackBackend)) {
        throw err;
      }

      kwarn('init', `Warmup failed on ${tf.getBackend()} — falling back to ${fallbackBackend}`);
      attemptedFallbacks.add(fallbackBackend);
      probedMctsBatchSize = null;
      await switchToFallbackBackendForRequest(requestedBackend, fallbackBackend);
    }
  }
}

async function ensureModel(
  modelUrl: string,
  backend?: KataGoBackendPreference,
  modelBuffer?: ArrayBuffer | null,
  modelByteSize?: number
): Promise<void> {
  const requestedBackend =
    typeof modelByteSize === 'number'
      ? recommendBackendForModelSize(modelByteSize, backend ?? 'webgpu')
      : normalizeKataGoBackendPreference(backend);
  await ensureBackend(requestedBackend);
  if (model && loadedModelUrl === modelUrl) return;

  await disposeModelSafely();

  let buf: Uint8Array;
  if (modelBuffer) {
    buf = new Uint8Array(modelBuffer);
    klog('init', `Model supplied as buffer (${(modelBuffer.byteLength / 1_000_000).toFixed(1)} MB) | backend: ${requestedBackend}`);
    postLoadProgress('fetch', 100);
  } else {
    const shortUrl = modelUrl.split('/').pop() ?? modelUrl;
    klog('init', `Fetching model: ${shortUrl} | backend: ${requestedBackend}`);
    postLoadProgress('fetch', 0);
    const t0 = getAnimationNow();
    const res = await fetch(modelUrl);
    if (!res.ok) throw new Error(`Failed to fetch model: ${res.status} ${res.statusText}`);
    buf = new Uint8Array(await res.arrayBuffer());
    klog('init', `Fetch complete: ${shortUrl} | ${(buf.byteLength / 1_000_000).toFixed(1)} MB | ${Math.round(getAnimationNow() - t0)}ms`);
    postLoadProgress('fetch', 100);
  }

  await loadModelFromBuffer(buf, modelUrl, requestedBackend);
}

async function recoverFromGpuError(
  requestedBackend: KataGoBackendPreference | undefined,
  modelUrl: string,
  modelBuffer?: ArrayBuffer | null,
  modelByteSize?: number
): Promise<boolean> {
  const normalizedRequested = normalizeKataGoBackendPreference(requestedBackend);
  const fallbackBackend = getKataGoFallbackBackend({
    requestedBackend: normalizedRequested,
    activeBackend: tf.getBackend(),
  });
  if (!fallbackBackend) return false;

  search = null;
  await disposeModelSafely();
  backendPromise = null;
  backendPreference = null;
  await ensureBackend(fallbackBackend);
  await ensureModel(modelUrl, fallbackBackend, modelBuffer, modelByteSize);
  return true;
}

async function withGpuRecovery<T>(
  args: {
    backend?: KataGoBackendPreference;
    modelUrl: string;
    modelBuffer?: ArrayBuffer | null;
    modelByteSize?: number;
  },
  fn: () => Promise<T>
): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    if (!isGpuDeviceLostError(err)) throw err;
    kwarn('backend', `GPU device-lost detected (backend: ${tf.getBackend()}) — attempting fallback recovery…`);
    const recovered = await recoverFromGpuError(args.backend, args.modelUrl, args.modelBuffer, args.modelByteSize);
    if (!recovered) {
      kwarn('backend', 'GPU recovery failed — no further fallback available');
      throw err;
    }
    klog('backend', `GPU recovery OK — now on: ${tf.getBackend()}`);
    return fn();
  }
}

function post(msg: KataGoWorkerResponse, transfer?: Transferable[]) {
  if (transfer && transfer.length > 0) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

// ── Structured console logger ─────────────────────────────────────────────────
// Tags: init | probe | analyze | cache | backend
const klog  = (tag: string, msg: string) => console.info(`[katago:${tag}] ${msg}`);
const kwarn = (tag: string, msg: string) => console.warn(`[katago:${tag}] ${msg}`);

async function handleMessage(msg: KataGoWorkerRequest): Promise<void> {
  if (msg.type === 'katago:init') {
    await ensureModel(msg.modelUrl, msg.backend, null, msg.modelByteSize);
    post({
      type: 'katago:init_result',
      ok: true,
      backend: tf.getBackend(),
      modelName: loadedModelName,
      profile: resourceProfile
        ? toProfilePayload(resourceProfile, tf.getBackend(), probedMctsBatchSize ?? resourceProfile.maxMctsBatchSize)
        : undefined,
    });
    return;
  }

  if (msg.type === 'katago:init_buffer') {
    await ensureModel(msg.modelUrl, msg.backend, msg.modelBuffer, msg.modelBuffer.byteLength);
    post({
      type: 'katago:init_result',
      ok: true,
      backend: tf.getBackend(),
      modelName: loadedModelName,
      profile: resourceProfile
        ? toProfilePayload(resourceProfile, tf.getBackend(), probedMctsBatchSize ?? resourceProfile.maxMctsBatchSize)
        : undefined,
    });
    return;
  }

  if (msg.type === 'katago:analyze') {
    const meta = analyzeMeta.get(msg);
    const analysisGroup = meta?.analysisGroup ?? msg.analysisGroup ?? 'background';
    const interactiveTokenAtEnqueue = meta?.interactiveToken ?? interactiveToken;
    const isStale = () => latestAnalyzeByGroup.get(analysisGroup) !== msg.id;
    const isPreemptedByInteractive =
      analysisGroup !== 'interactive' && interactiveToken !== interactiveTokenAtEnqueue;
    const shouldAbort = () => isStale() || isPreemptedByInteractive;
    const postCanceled = () =>
      post({
        type: 'katago:analyze_result',
        id: msg.id,
        ok: false,
        canceled: true,
        error: 'canceled',
      });

    if (shouldAbort()) {
      postCanceled();
      return;
    }

    await withGpuRecovery({ backend: msg.backend, modelUrl: msg.modelUrl }, async () => {
    await ensureModel(msg.modelUrl, msg.backend);
    if (!model) throw new Error('Model not loaded');
    if (!resourceProfile) throw new Error('Resource profile unavailable');
    if (shouldAbort()) {
      postCanceled();
      return;
    }

    ensureBoardSizeForWorker(msg.board.length);

    const maxVisits = Math.max(16, Math.min(msg.visits ?? 256, ENGINE_MAX_VISITS));
    const maxTimeMs = Math.max(25, Math.min(msg.maxTimeMs ?? 800, ENGINE_MAX_TIME_MS));
    const nnRandomize = msg.nnRandomize !== false;
    const includeMovesOwnership = msg.includeMovesOwnership === true;
    const tuning = getAnalyzeTuning({
      profile: resourceProfile,
      backend: tf.getBackend(),
      probedMctsBatchSize,
      msgBatchSize: msg.batchSize,
      msgOwnershipMode: msg.ownershipMode,
      includeMovesOwnership,
      nnRandomize,
    });
    const batchSize = tuning.batchSize;
    const rootSymmetrySamples = tuning.rootSymmetrySamples;
    const ownershipMode: OwnershipMode = tuning.ownershipMode;
    const maxChildren = Math.max(4, Math.min(msg.maxChildren ?? 64, BOARD_AREA));
    const topK = Math.max(1, Math.min(msg.topK ?? 10, 50));
    const analysisPvLen = Math.max(0, Math.min(msg.analysisPvLen ?? 15, 60));
    const wideRootNoise = Math.max(0, Math.min(msg.wideRootNoise ?? 0.04, 5));
    const rules: GameRules = msg.rules === 'chinese' ? 'chinese' : msg.rules === 'korean' ? 'korean' : 'japanese';
    const conservativePass = msg.conservativePass !== false;
    const reportEveryMsRaw = msg.reportDuringSearchEveryMs;
    const reportEveryMs =
      typeof reportEveryMsRaw === 'number' && Number.isFinite(reportEveryMsRaw)
        ? Math.max(0, reportEveryMsRaw)
        : 0;
    const shouldReport = reportEveryMs > 0;
    const speedFlags = getVisitSpeedFlags();
    const useLightProgress = shouldReport && speedFlags.lightProgress;

    const analyzeT0 = getAnimationNow();
    klog('analyze', `#${msg.id} start | ${analysisGroup} | visits: ${maxVisits} | time: ${maxTimeMs}ms | batch: ${batchSize}`);

    search = await MctsSearch.create({
      model,
      board: msg.board,
      previousBoard: msg.previousBoard,
      previousPreviousBoard: msg.previousPreviousBoard,
      previousPreviousPreviousBoard: msg.previousPreviousPreviousBoard,
      koPoint: msg.koPoint,
      currentPlayer: msg.currentPlayer,
      moveHistory: msg.moveHistory,
      komi: msg.komi,
      rules,
      nnRandomize,
      conservativePass,
      maxChildren,
      ownershipMode,
      wideRootNoise,
      rootSymmetrySamples,
      regionOfInterest: msg.regionOfInterest,
    });

    const releaseSearch = () => { search = null; };

    const postAnalysis = (analysis: ReturnType<MctsSearch['getAnalysis']>, type: 'katago:analyze_update' | 'katago:analyze_result') => {
      const transfer: Transferable[] = [];
      const seen = new Set<ArrayBufferLike>();
      const push = (value?: unknown) => {
        if (!value || !ArrayBuffer.isView(value) || (value as ArrayBufferView).byteLength <= 0) return;
        const buf = (value as ArrayBufferView).buffer;
        if (seen.has(buf) || buf.byteLength === 0) return;
        if ('detached' in buf && (buf as ArrayBuffer & { detached: boolean }).detached) return;
        seen.add(buf);
        transfer.push(buf as ArrayBuffer);
      };
      push(analysis.ownership);
      push(analysis.ownershipStdev);
      push(analysis.policy);
      for (const move of analysis.moves) push(move.ownership);

      post(
        {
          type,
          id: msg.id,
          ok: true,
          backend: tf.getBackend(),
          modelName: loadedModelName,
          analysis,
        },
        transfer
      );
    };

    const buildAnalysis = (forProgress = false) => {
      const args = { topK, includeMovesOwnership, analysisPvLen, ownershipRefreshIntervalMs: msg.ownershipRefreshIntervalMs };
      if (forProgress && useLightProgress) {
        return search!.getAnalysisLight(args);
      }
      // Always clone before postMessage transfer — transferring rootOwnership /
      // rootPolicy detaches the ArrayBuffer and breaks subsequent updates.
      return search!.getAnalysis({ ...args, cloneBuffers: true });
    };

    // Post root evaluation immediately so the UI shows candidate moves without
    // waiting for the first full GPU batch (2–5 s on large models).
    if (shouldReport && !shouldAbort()) {
      postAnalysis(buildAnalysis(true), 'katago:analyze_update');
    }

    // Position cache — only for non-reporting searches (graph generation).
    const cacheModelName = loadedModelName ?? '';
    // Interactive analysis needs move lists; cache stores scalars only.
    const shouldUseCache = !shouldReport && cacheModelName.length > 0;
    let cacheKey: string | null = null;
    if (shouldUseCache) {
      cacheKey = makeCacheKey(cacheModelName, boardToFlatForHash(msg.board), msg.currentPlayer, msg.komi, rules);
      const cached = await getCachedPosition(cacheKey, maxVisits);
      if (cached && !shouldAbort()) {
        klog('cache', `  Hit — visits: ${cached.rootVisits} | wr: ${cached.rootWinRate.toFixed(3)} | lead: ${cached.rootScoreLead.toFixed(1)} | saved ${Math.round(getAnimationNow() - analyzeT0)}ms`);
        post({
          type: 'katago:analyze_result',
          id: msg.id,
          ok: true,
          backend: tf.getBackend(),
          modelName: loadedModelName,
          analysis: {
            rootWinRate: cached.rootWinRate,
            rootScoreLead: cached.rootScoreLead,
            rootScoreSelfplay: cached.rootScoreSelfplay,
            rootScoreStdev: 0,
            rootVisits: cached.rootVisits,
            ownership: new Float32Array(BOARD_AREA),
            ownershipStdev: new Float32Array(BOARD_AREA),
            policy: new Float32Array(BOARD_AREA + 1).fill(-1),
            moves: [],
          },
        });
        releaseSearch();
        return;
      }
    }

    if (!shouldReport) {
      const aborted = await search!.run({ visits: maxVisits, maxTimeMs, batchSize, shouldAbort });
      if (aborted || shouldAbort()) {
        postCanceled();
        releaseSearch();
        return;
      }
      const finishedAnalysis = buildAnalysis();
      postAnalysis(finishedAnalysis, 'katago:analyze_result');
      klog('analyze', `#${msg.id} done | visits: ${finishedAnalysis.rootVisits} | wr: ${finishedAnalysis.rootWinRate.toFixed(3)} | lead: ${finishedAnalysis.rootScoreLead >= 0 ? '+' : ''}${finishedAnalysis.rootScoreLead.toFixed(1)} | ${Math.round(getAnimationNow() - analyzeT0)}ms`);
      // §1.4 Write result to position cache (scalars only; typed-array buffers may be transferred).
      if (cacheKey) {
        setCachedPosition(cacheKey, cacheModelName, {
          rootWinRate: finishedAnalysis.rootWinRate,
          rootScoreLead: finishedAnalysis.rootScoreLead,
          rootScoreSelfplay: finishedAnalysis.rootScoreSelfplay,
          rootVisits: finishedAnalysis.rootVisits,
        }, maxVisits).catch(() => {});
      }
      releaseSearch();
      return;
    }

    if (speedFlags.singleRunReporting) {
      let lastReportVisits = -1;
      const aborted = await search!.run({
        visits: maxVisits,
        maxTimeMs,
        batchSize,
        shouldAbort,
        reportEveryMs,
        onProgress: () => {
          if (shouldAbort()) return;
          const analysis = buildAnalysis(true);
          if (analysis.rootVisits > lastReportVisits) {
            lastReportVisits = analysis.rootVisits;
            postAnalysis(analysis, 'katago:analyze_update');
          }
        },
      });
      if (aborted || shouldAbort()) {
        postCanceled();
        releaseSearch();
        return;
      }
      const finalAnalysis = buildAnalysis();
      postAnalysis(finalAnalysis, 'katago:analyze_result');
      const reason = finalAnalysis.rootVisits >= maxVisits ? 'visit cap' : 'time limit';
      klog('analyze', `#${msg.id} done | visits: ${finalAnalysis.rootVisits} | wr: ${finalAnalysis.rootWinRate.toFixed(3)} | lead: ${finalAnalysis.rootScoreLead >= 0 ? '+' : ''}${finalAnalysis.rootScoreLead.toFixed(1)} | ${Math.round(getAnimationNow() - analyzeT0)}ms (${reason})`);
      if (cacheKey) {
        setCachedPosition(cacheKey, cacheModelName, {
          rootWinRate: finalAnalysis.rootWinRate,
          rootScoreLead: finalAnalysis.rootScoreLead,
          rootScoreSelfplay: finalAnalysis.rootScoreSelfplay,
          rootVisits: finalAnalysis.rootVisits,
        }, maxVisits).catch(() => {});
      }
      releaseSearch();
      return;
    }

    const deadline = getAnimationNow() + maxTimeMs;
    let lastReportVisits = -1;
    while (true) {
      if (shouldAbort()) {
        postCanceled();
        releaseSearch();
        return;
      }
      const now = getAnimationNow();
      const remaining = deadline - now;
      if (remaining <= 0) break;
      const sliceMs = Math.min(reportEveryMs, remaining);
      const aborted = await search!.run({ visits: maxVisits, maxTimeMs: sliceMs, batchSize, shouldAbort });
      if (aborted || shouldAbort()) {
        postCanceled();
        releaseSearch();
        return;
      }
      const analysis = buildAnalysis(true);
      const done = analysis.rootVisits >= maxVisits || getAnimationNow() >= deadline;
      if (done) {
        postAnalysis(buildAnalysis(), 'katago:analyze_result');
        const reason = analysis.rootVisits >= maxVisits ? 'visit cap' : 'time limit';
        klog('analyze', `#${msg.id} done | visits: ${analysis.rootVisits} | wr: ${analysis.rootWinRate.toFixed(3)} | lead: ${analysis.rootScoreLead >= 0 ? '+' : ''}${analysis.rootScoreLead.toFixed(1)} | ${Math.round(getAnimationNow() - analyzeT0)}ms (${reason})`);
        releaseSearch();
        return;
      }
      if (analysis.rootVisits > lastReportVisits) {
        lastReportVisits = analysis.rootVisits;
        postAnalysis(analysis, 'katago:analyze_update');
      }
    }

    const finalAnalysis = buildAnalysis();
    postAnalysis(finalAnalysis, 'katago:analyze_result');
    klog('analyze', `#${msg.id} done | visits: ${finalAnalysis.rootVisits} | wr: ${finalAnalysis.rootWinRate.toFixed(3)} | lead: ${finalAnalysis.rootScoreLead >= 0 ? '+' : ''}${finalAnalysis.rootScoreLead.toFixed(1)} | ${Math.round(getAnimationNow() - analyzeT0)}ms (time limit)`);
    releaseSearch();
    });
  }
}

self.onmessage = (ev: MessageEvent<KataGoWorkerRequest>) => {
  const msg = ev.data;

  // Cancellation runs immediately, outside the serial queue, so it preempts in-flight work.
  if (msg.type === 'katago:cancel') {
    latestAnalyzeByGroup.set(msg.analysisGroup, -1);
    if (msg.analysisGroup === 'interactive') interactiveToken++;
    return;
  }

  if (msg.type === 'katago:analyze') {
    const analysisGroup = msg.analysisGroup ?? 'background';
    latestAnalyzeByGroup.set(analysisGroup, msg.id);
    if (analysisGroup === 'interactive') interactiveToken++;
    analyzeMeta.set(msg, { analysisGroup, interactiveToken });
  }
  queue = queue
    .then(() => handleMessage(msg))
    .catch((err: unknown) => {
      if (msg.type === 'katago:init' || msg.type === 'katago:init_buffer') {
        post({
          type: 'katago:init_result',
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
      if (msg.type === 'katago:analyze') {
        post({
          type: 'katago:analyze_result',
          id: msg.id,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
    });
};
