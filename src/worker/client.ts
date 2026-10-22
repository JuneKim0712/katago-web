import type { KataGoLoadProgress, KataGoWorkerRequest, KataGoWorkerResponse } from '../engine/types.js';
import type { BoardState, GameRules, KataGoBackendPreference, Move, Player, RegionOfInterest } from '../types/index.js';
import { getWorkerConstructor } from '../utils/browserWorker.js';
import { formatKataGoUserError } from './gpuError.js';
import type { ResourceProfilePayload } from './resourceProfile.js';
import { recommendBackendForModelSize } from './resourceProfile.js';
import { ENGINE_MAX_TIME_MS } from '../engine/limits.js';

type Analysis = NonNullable<Extract<KataGoWorkerResponse, { type: 'katago:analyze_result' }>['analysis']>;

const takeLastMoves = (moves: Move[]): Move[] => (moves.length <= 5 ? moves : moves.slice(moves.length - 5));

export class KataGoCanceledError extends Error {
  readonly canceled = true;

  constructor(message = 'Analysis canceled') {
    super(message);
    this.name = 'KataGoCanceledError';
  }
}

export const isKataGoCanceledError = (err: unknown): err is KataGoCanceledError => {
  if (!err || typeof err !== 'object') return false;
  if ((err as { canceled?: boolean }).canceled) return true;
  return err instanceof Error && err.name === 'KataGoCanceledError';
};

/** Outstanding request bookkeeping shared by every request map. `timer` is a
 *  watchdog that restarts the worker if it goes silent, so callers never hang. */
type Pending<T> = {
  resolve: (value: T) => void;
  reject: (e: Error) => void;
  timer?: ReturnType<typeof setTimeout>;
  watchdogMs?: number;
  onProgress?: (a: Analysis) => void;
};

// Safety-net watchdogs. These are deliberately generous — they only fire on a
// genuine hang, never on legitimately slow work — and exist purely so a wedged
// or crashed worker can NEVER leave the UI frozen.
const INIT_WATCHDOG_MS = 300_000;

export class KataGoEngineClient {
  private worker: Worker;
  private nextId = 1;
  private pendingInit: Pending<void> | null = null;
  // Coalescing: multiple callers awaiting the same in-flight init share one promise.
  private initInFlight: Promise<void> | null = null;
  private initInFlightUrl: string | null = null;
  // The URL of the model that is currently fully loaded in the worker.
  private loadedModelUrl: string | null = null;
  private pending = new Map<number, Pending<Analysis>>();
  private backend: string | null = null;
  private modelName: string | null = null;
  private resourceProfile: ResourceProfilePayload | null = null;
  private loadProgress: KataGoLoadProgress | null = null;
  private lastLoggedEngineLabel: string | null = null;
  private readonly loadProgressListeners = new Set<(progress: KataGoLoadProgress) => void>();
  private readonly profileListeners = new Set<(profile: ResourceProfilePayload) => void>();
  // A worker that dies before it ever speaks (script blocked, bad bundle) must not
  // be respawned forever — each new Worker() fails the same way and wedges the UI.
  private consecutiveStartFailures = 0;

  constructor() {
    if (!getWorkerConstructor()) {
      throw new Error('Browser Worker API is unavailable; KataGo analysis cannot run in this browser context.');
    }
    this.worker = this.spawnWorker();
  }

  private spawnWorker(): Worker {
    let worker: Worker;
    try {
      worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    } catch (err) {
      throw formatWorkerError(err, 'KataGo worker failed to start');
    }
    worker.onmessage = (ev: MessageEvent<KataGoWorkerResponse>) => this.handleWorkerMessage(ev.data);
    // A crashed/erroring worker would otherwise leave every awaiter hanging forever.
    worker.onerror = (ev: ErrorEvent) => {
      const where = ev?.filename ? `${ev.filename}:${ev.lineno || 0}` : '';
      const detail = [ev?.message, where].filter(Boolean).join(' ');
      this.restartWorker(detail || 'worker error');
    };
    worker.onmessageerror = () => this.restartWorker('worker message deserialization failed');
    return worker;
  }

  private handleWorkerMessage(msg: KataGoWorkerResponse): void {
    this.consecutiveStartFailures = 0;
    if (msg.type === 'katago:load_progress') {
      this.loadProgress = msg;
      for (const listener of this.loadProgressListeners) listener(msg);
      return;
    }
    if (msg.type === 'katago:profile') {
      this.resourceProfile = msg.profile;
      for (const listener of this.profileListeners) listener(msg.profile);
      return;
    }
    if (msg.type === 'katago:init_result') {
      const pendingInit = this.pendingInit;
      if (!pendingInit) return;
      this.pendingInit = null;
      clearTimeout(pendingInit.timer);
      if (msg.ok) {
        this.syncEngineInfo(msg);
        if (msg.profile) {
          this.resourceProfile = msg.profile;
          for (const listener of this.profileListeners) listener(msg.profile);
        }
        pendingInit.resolve();
      } else {
        pendingInit.reject(new Error(formatKataGoUserError(msg.error ?? 'Init failed')));
      }
      return;
    }
    if (msg.type === 'katago:analyze_update') {
      const pending = this.pending.get(msg.id);
      if (!pending) return;
      if (msg.canceled || msg.error === 'canceled') return;
      // Progress = liveness: push the watchdog back so a long but healthy search
      // is never mistaken for a hang.
      this.rearmWatchdog(pending, msg.id, 'analyze');
      this.syncEngineInfo(msg);
      if (!msg.ok || !msg.analysis) return;
      pending.onProgress?.(msg.analysis);
      return;
    }
    if (msg.type === 'katago:analyze_result') {
      const pending = this.pending.get(msg.id);
      if (!pending) return;
      this.pending.delete(msg.id);
      clearTimeout(pending.timer);
      if (msg.canceled || msg.error === 'canceled') {
        pending.reject(new KataGoCanceledError());
        return;
      }
      this.syncEngineInfo(msg);
      if (!msg.ok || !msg.analysis) pending.reject(new Error(formatKataGoUserError(msg.error ?? 'Analysis failed')));
      else pending.resolve(msg.analysis);
    }
  }

  /** Arm a watchdog on a pending request. If the worker stays silent past the
   *  deadline it is presumed wedged and gets restarted, rejecting the request. */
  private armWatchdog<T>(entry: Pending<T>, id: number, kind: string, ms: number): void {
    entry.watchdogMs = ms;
    entry.timer = setTimeout(() => this.restartWorker(`${kind} #${id} timed out after ${ms}ms`), ms);
  }

  private rearmWatchdog<T>(entry: Pending<T>, id: number, kind: string): void {
    if (entry.watchdogMs == null) return;
    clearTimeout(entry.timer);
    entry.timer = setTimeout(() => this.restartWorker(`${kind} #${id} stalled`), entry.watchdogMs);
  }

  /** Reject every outstanding request so no awaiting caller is left hanging. */
  private failAllPending(err: Error): void {
    if (this.pendingInit) {
      clearTimeout(this.pendingInit.timer);
      this.pendingInit.reject(err);
      this.pendingInit = null;
    }
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(err); }
    this.pending.clear();
    this.initInFlight = null;
    this.initInFlightUrl = null;
    this.loadedModelUrl = null;
  }

  /** Tear down a crashed/wedged worker and spin up a fresh one. Outstanding
   *  requests reject as canceled so the UI recovers silently and the next
   *  request transparently re-loads the model into the new worker. */
  private restartWorker(reason: string): void {
    this.consecutiveStartFailures += 1;
    try { this.worker.terminate(); } catch { /* already dead */ }
    this.backend = null;
    this.modelName = null;
    this.resourceProfile = null;
    this.loadProgress = null;

    if (this.consecutiveStartFailures >= 3) {
      console.error(`[katago:recover] ${reason} — stopped after ${this.consecutiveStartFailures} failures`);
      this.failAllPending(new Error('AI engine could not start. Reload the page and try again.'));
      return;
    }

    console.warn(`[katago:recover] ${reason} — restarting worker`);
    this.failAllPending(new KataGoCanceledError(`engine restarted (${reason})`));
    try {
      this.worker = this.spawnWorker();
    } catch (err) {
      console.error('[katago:recover] failed to restart worker', err);
    }
  }

  dispose(): void {
    this.failAllPending(new KataGoCanceledError('engine disposed'));
    this.worker.terminate();
  }

  private postToWorker(message: KataGoWorkerRequest): void {
    try {
      this.worker.postMessage(message);
    } catch (err) {
      throw formatWorkerError(err, 'KataGo worker message failed');
    }
  }

  private syncEngineInfo(msg: { backend?: string; modelName?: string }): void {
    let changed = false;
    if (typeof msg.backend === 'string' && msg.backend !== this.backend) {
      this.backend = msg.backend;
      changed = true;
    }
    if (typeof msg.modelName === 'string' && msg.modelName !== this.modelName) {
      this.modelName = msg.modelName;
      changed = true;
    }
    if (!changed) return;

    const parts: string[] = [];
    if (this.backend) parts.push(this.backend);
    if (this.modelName) parts.push(this.modelName);
    const label = parts.join(' / ');
    if (!label || label === this.lastLoggedEngineLabel) return;
    this.lastLoggedEngineLabel = label;
    console.info(`[katago] engine: ${label}`);
  }

  getEngineInfo(): { backend: string | null; modelName: string | null } {
    return { backend: this.backend, modelName: this.modelName };
  }

  getResourceProfile(): ResourceProfilePayload | null {
    return this.resourceProfile;
  }

  getLoadProgress(): KataGoLoadProgress | null {
    return this.loadProgress;
  }

  onLoadProgress(listener: (progress: KataGoLoadProgress) => void): () => void {
    this.loadProgressListeners.add(listener);
    return () => this.loadProgressListeners.delete(listener);
  }

  onProfile(listener: (profile: ResourceProfilePayload) => void): () => void {
    this.profileListeners.add(listener);
    if (this.resourceProfile) listener(this.resourceProfile);
    return () => this.profileListeners.delete(listener);
  }

  cancelGroup(analysisGroup: 'interactive' | 'background'): void {
    try {
      this.worker.postMessage({ type: 'katago:cancel', analysisGroup });
    } catch {
      // Ignore if worker is already terminated.
    }
  }

  init(modelUrl: string, backend?: KataGoBackendPreference, modelByteSize?: number): Promise<void> {
    // Already loaded with the same model — nothing to do.
    if (this.loadedModelUrl === modelUrl && !this.pendingInit) {
      return Promise.resolve();
    }
    // An init for *this same* model is already in flight — share its promise.
    if (this.initInFlight && this.initInFlightUrl === modelUrl) {
      return this.initInFlight;
    }

    const resolvedBackend =
      typeof modelByteSize === 'number'
        ? recommendBackendForModelSize(modelByteSize, backend ?? 'auto')
        : (backend ?? 'webgpu');

    return this.startInit(modelUrl, () => {
      this.postToWorker({ type: 'katago:init', modelUrl, backend: resolvedBackend, modelByteSize });
    });
  }

  initFromBuffer(modelUrl: string, modelBuffer: ArrayBuffer, backend?: KataGoBackendPreference): Promise<void> {
    if (this.initInFlight && this.initInFlightUrl === modelUrl) return this.initInFlight;

    const resolvedBackend = recommendBackendForModelSize(modelBuffer.byteLength, backend ?? 'auto');
    return this.startInit(modelUrl, () => {
      try {
        this.worker.postMessage({ type: 'katago:init_buffer', modelUrl, modelBuffer, backend: resolvedBackend }, [modelBuffer]);
      } catch (err) {
        throw formatWorkerError(err, 'KataGo worker message failed');
      }
    });
  }

  /** Shared init machinery: supersede any different in-flight init, post the
   *  request, and track an init watchdog. */
  private startInit(modelUrl: string, send: () => void): Promise<void> {
    // Supersede any in-flight init for a different model so its awaiter never hangs.
    if (this.pendingInit) {
      clearTimeout(this.pendingInit.timer);
      this.pendingInit.reject(new KataGoCanceledError('init superseded'));
      this.pendingInit = null;
    }

    this.loadedModelUrl = null; // set on success
    this.loadProgress = null;
    this.initInFlightUrl = modelUrl;

    this.initInFlight = new Promise<void>((resolve, reject) => {
      const entry: Pending<void> = { resolve, reject };
      this.armWatchdog(entry, 0, 'init', INIT_WATCHDOG_MS);
      this.pendingInit = entry;
      try {
        send();
      } catch (err) {
        clearTimeout(entry.timer);
        this.pendingInit = null;
        this.initInFlight = null;
        this.initInFlightUrl = null;
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    }).then(() => {
      this.loadedModelUrl = modelUrl;
      this.initInFlight = null;
      this.initInFlightUrl = null;
    }).catch((err) => {
      this.loadedModelUrl = null;
      this.initInFlight = null;
      this.initInFlightUrl = null;
      throw err;
    });

    return this.initInFlight;
  }

  async analyze(args: {
    analysisGroup?: 'interactive' | 'background';
    modelUrl: string;
    backend?: KataGoBackendPreference;
    board: BoardState;
    previousBoard?: BoardState;
    previousPreviousBoard?: BoardState;
    previousPreviousPreviousBoard?: BoardState;
    koPoint?: number;
    currentPlayer: Player;
    moveHistory: Move[];
    komi: number;
    rules?: GameRules;
    regionOfInterest?: RegionOfInterest | null;
    topK?: number;
    analysisPvLen?: number;
    includeMovesOwnership?: boolean;
    wideRootNoise?: number;
    nnRandomize?: boolean;
    conservativePass?: boolean;
    visits?: number;
    maxTimeMs?: number;
    batchSize?: number;
    maxChildren?: number;
    reportDuringSearchEveryMs?: number;
    ownershipRefreshIntervalMs?: number;
    ownershipMode?: 'none' | 'root' | 'tree';
    onProgress?: (analysis: Analysis) => void;
  }): Promise<Analysis> {
    const id = this.nextId++;
    const req: KataGoWorkerRequest = {
      type: 'katago:analyze',
      id,
      analysisGroup: args.analysisGroup,
      modelUrl: args.modelUrl,
      backend: args.backend,
      board: args.board,
      previousBoard: args.previousBoard,
      previousPreviousBoard: args.previousPreviousBoard,
      previousPreviousPreviousBoard: args.previousPreviousPreviousBoard,
      koPoint: args.koPoint,
      currentPlayer: args.currentPlayer,
      moveHistory: takeLastMoves(args.moveHistory),
      komi: args.komi,
      rules: args.rules,
      regionOfInterest: args.regionOfInterest,
      topK: args.topK,
      analysisPvLen: args.analysisPvLen,
      includeMovesOwnership: args.includeMovesOwnership,
      wideRootNoise: args.wideRootNoise,
      nnRandomize: args.nnRandomize,
      conservativePass: args.conservativePass,
      visits: args.visits,
      maxTimeMs: args.maxTimeMs,
      batchSize: args.batchSize,
      maxChildren: args.maxChildren,
      reportDuringSearchEveryMs: args.reportDuringSearchEveryMs,
      ownershipRefreshIntervalMs: args.ownershipRefreshIntervalMs,
      ownershipMode: args.ownershipMode,
    };
    // Watchdog must outlast the request's own time budget plus model (re)load
    // and the first forward pass, which can take many seconds on large models.
    const watchdogMs = Math.min(ENGINE_MAX_TIME_MS, Math.max(args.maxTimeMs ?? 15_000, 1_000)) + 120_000;
    const promise = new Promise<Analysis>((resolve, reject) => {
      const entry: Pending<Analysis> = { resolve, reject, onProgress: args.onProgress };
      this.armWatchdog(entry, id, 'analyze', watchdogMs);
      this.pending.set(id, entry);
    });
    try {
      this.postToWorker(req);
    } catch (err) {
      const entry = this.pending.get(id);
      if (entry) clearTimeout(entry.timer);
      this.pending.delete(id);
      throw err;
    }
    return promise;
  }
}

let singleton: KataGoEngineClient | null = null;

function formatWorkerError(err: unknown, prefix: string): Error {
  const message = err instanceof Error ? err.message : String(err);
  return new Error(message ? `${prefix}: ${message}` : prefix);
}

export function getKataGoEngineClient(): KataGoEngineClient {
  if (!singleton) singleton = new KataGoEngineClient();
  return singleton;
}

/**
 * Returns the engine client used for full-game graph generation.
 *
 * NOTE: This intentionally shares the same singleton as `getKataGoEngineClient`.
 * A dedicated second worker (§ 1.3 of the optimisation plan) would double VRAM
 * usage — which causes a GPU device-lost crash on GPUs that can only hold one
 * copy of the model.  Resource dedication is achieved instead by
 * `useFullGameAnalysis` calling `client.cancelGroup('interactive')` at the start
 * of each graph-generation pass, ensuring the worker is fully dedicated to the
 * graph while it runs.
 */
export function getKataGoGraphEngineClient(): KataGoEngineClient {
  return getKataGoEngineClient();
}

/**
 * Hard-reset the engine: terminates the Web Worker, releases the GPU context,
 * and clears the singleton so the next analysis request creates a fresh worker.
 * Call this when the engine appears frozen and normal cancellation has no effect.
 */
export function resetKataGoEngineClient(): void {
  console.info('[katago:reset] Engine reset — terminating worker and clearing singleton');
  singleton?.dispose();
  singleton = null;
}

export function resetKataGoEngineClientForTests(): void {
  singleton?.dispose();
  singleton = null;
}
