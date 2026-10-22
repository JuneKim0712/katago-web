/**
 * Public Contract v1 types for KataGo Web.
 * Conforms strictly to C:/JuneKim0712/baduk/docs/katago-package-contract/v1.ts.
 */

export const API_VERSION = 1 as const;
export type Color = 'black' | 'white';
export type Stone = 0 | 1 | 2; // empty, black, white
export type Point = Readonly<{ x: number; y: number }>;
export type Move = Readonly<{ kind: 'play'; point: Point } | { kind: 'pass' }>;
export type PlayedMove = Readonly<{ player: Color; move: Move }>;
export type StonePlacement = Readonly<{ player: Color; point: Point }>;
export type Dimensions = Readonly<{ width: number; height: number }>;

export type Rules = Readonly<{
  ko: 'simple' | 'positional' | 'situational' | 'spight';
  scoring: 'area' | 'territory';
  tax: 'none' | 'seki' | 'all';
  multiStoneSuicideLegal: boolean;
  hasButton: boolean;
  whiteHandicapBonus: '0' | 'N' | 'N-1';
  friendlyPassOk: boolean;
}>;
export type RulesInput = Readonly<
  { kind: 'preset'; name: 'japanese' | 'korean' | 'chinese' } |
  { kind: 'explicit'; value: Rules }
>;

/** History starts from an explicitly declared normal-phase initial position. */
export type GamePosition = Readonly<{
  kind: 'game';
  dimensions: Dimensions;
  initialStones: readonly StonePlacement[];
  initialPlayer: Color;
  moves: readonly PlayedMove[];
  rules: RulesInput;
  komi: number; // points added to white
  handicap: Readonly<{
    countOverride: number | null;
    inferFromInitialBlackMoves: boolean;
  }>;
}>;

/** Explicitly incomplete history. Never advertised as full-game equivalence. */
export type SetupPosition = Readonly<{
  kind: 'setup';
  dimensions: Dimensions;
  stones: readonly StonePlacement[];
  nextPlayer: Color;
  rules: RulesInput;
  komi: number;
  simpleKoPoint: Point | null;
  acknowledgeMissingHistory: true;
}>;
export type Position = GamePosition | SetupPosition;

/** Immutable manifest, not a TFJS/ORT object. URLs are absolute and fetchable. */
export type ModelSource = Readonly<
  { kind: 'manifest-url'; url: string; expectedSha256: string | null } |
  { kind: 'manifest'; manifest: ModelManifest }
>;
export type ModelArtifact = Readonly<{
  path: string; // unique logical path, no traversal
  sha256: string;
  byteLength: number;
  source: Readonly<{ kind: 'url'; url: string } | { kind: 'bytes'; bytes: Uint8Array }>;
}>;
export type ModelManifest = Readonly<{
  schemaVersion: 1;
  modelId: string; // display identity, never sufficient for cache keys
  sourceModelSha256: string;
  variants: readonly Readonly<{
    variantId: string;
    format: 'katago-bin-gzip' | 'katago-bin' | 'onnx';
    entryPath: string;
    artifacts: readonly ModelArtifact[];
  }>[];
}>;

export type RuntimePreference = Readonly<{
  accelerator: 'webgpu' | 'wasm-cpu';
  fallback: 'forbid'; // no silent capability/precision/fidelity downgrade
}>;
export type Capabilities = Readonly<{
  apiVersion: 1;
  packageVersion: string;
  upstreamVersion: string;
  upstreamCommit: string;
  status: 'available' | 'unavailable';
  unavailableReasons: readonly string[];
  supportedAccelerators: readonly ('webgpu' | 'wasm-cpu')[];
  supportedModelFormats: readonly ModelManifest['variants'][number]['format'][];
  supportedRulePresets: readonly ('japanese' | 'korean' | 'chinese')[];
  maxBoardWidth: number;
  maxBoardHeight: number;
  supportsRectangularBoards: boolean;
  supportsSetupPositions: boolean;
  supportsRootMoveRestriction: boolean;
  supportsCandidateOwnership: boolean;
  supportsOwnershipStdev: boolean;
  crossOriginIsolationRequired: boolean;
  maxVisits: number;
  maxTimeMs: number;
  maxBatchSize: number;
  maxSearchThreads: number;
  maxPvLength: number;
}>;

export type EngineInfo = Readonly<{
  engineGeneration: number;
  packageVersion: string;
  upstreamCommit: string;
  modelId: string;
  modelFingerprint: string; // digest of verified model/variant/artifact identity
  sourceModelSha256: string;
  variantId: string;
  accelerator: 'webgpu' | 'wasm-cpu';
  inferenceImplementation: string; // informational, never application branching
  precision: string;
  modelVersion: number;
  blocks: number | null;
  channels: number | null;
  recommendedBatchSize: number;
  recommendedSearchThreads: number;
  recommendedProgressIntervalMs: number;
  capabilities: Capabilities;
}>;

export type Limit = Readonly<
  { kind: 'visits'; maxVisits: number } |
  { kind: 'time'; maxTimeMs: number } |
  { kind: 'either'; maxVisits: number; maxTimeMs: number }
>;
export type AnalyzeRequest = Readonly<{
  apiVersion: 1;
  requestId: string;
  position: Position;
  search: Readonly<{
    profile: 'upstream-analysis-v1';
    limits: Limit;
    seed: string | null;
    batchSize: number | 'auto';
    searchThreads: number | 'auto';
    conservativePass: boolean;
    wideRootNoise: number;
    rootSymmetrySamples: number;
    allowedRootMoves: readonly Move[] | null;
  }>;
  output: Readonly<{
    maxCandidates: number;
    pvLength: number;
    includePolicy: boolean;
    ownership: 'none' | 'root' | 'tree';
    includeOwnershipStdev: boolean;
    includeCandidateOwnership: boolean;
    progressIntervalMs: number | null;
  }>;
}>;

export type Evaluation = Readonly<{
  blackWinProbability: number;
  whiteWinProbability: number;
  noResultProbability: number;
  blackWinRate: number; // (1 + blackWinProbability - whiteWinProbability) / 2
  blackScoreLead: number; // black advantage in points
  blackScoreMean: number;
  scoreStdev: number;
}>;
export type Ownership = Readonly<{
  mode: 'root' | 'tree';
  mean: Float32Array; // width*height; +1 black, -1 white
  stdev: Float32Array | null;
}>;
export type Candidate = Readonly<{
  move: Move;
  order: number; // zero-based upstream report order
  visits: number;
  prior: number;
  evaluation: Evaluation | null; // null means not evaluated; never fake root statistics
  pv: readonly Move[]; // includes this candidate as element 0 when pvLength > 0
  ownership: Ownership | null; // after candidate, original board orientation
}>;
export type EffectiveSearch = Readonly<{
  profile: 'upstream-analysis-v1';
  profileFingerprint: string;
  seed: string;
  batchSize: number;
  searchThreads: number;
  conservativePass: boolean;
  wideRootNoise: number;
  rootSymmetrySamples: number;
  rootRestricted: boolean;
}>;
export type AnalysisSnapshot = Readonly<{
  apiVersion: 1;
  requestId: string;
  engineGeneration: number;
  sequence: number;
  positionFingerprint: string;
  modelFingerprint: string;
  dimensions: Dimensions;
  nextPlayer: Color;
  historyStatus: 'complete-from-declared-initial-state' | 'setup-only';
  resolvedRules: Rules;
  effectiveSearch: EffectiveSearch;
  rootVisits: number;
  evaluation: Evaluation;
  candidates: readonly Candidate[];
  policy: Float32Array | null; // width*height+1, last is pass; illegal exactly -1
  ownership: Ownership | null;
  accounting: Readonly<{
    elapsedMs: number;
    searchElapsedMs: number;
    completedNewVisits: number;
    reusedVisits: 0; // v1 always starts a fresh search
    neuralRows: number;
    terminalVisits: number;
    neuralCacheHits: number;
  }>;
}>;
export type AnalysisResult = Readonly<{
  kind: 'completed';
  stopReason: 'visit-limit' | 'time-limit' | 'terminal' | 'exhausted';
  snapshot: AnalysisSnapshot;
}>;

export type ErrorCode =
  | 'INVALID_REQUEST' | 'UNSUPPORTED_API_VERSION' | 'UNSUPPORTED_CAPABILITY'
  | 'INVALID_POSITION' | 'ILLEGAL_MOVE' | 'INCONSISTENT_HISTORY'
  | 'MODEL_FETCH_FAILED' | 'MODEL_HASH_MISMATCH' | 'MODEL_UNSUPPORTED'
  | 'BACKEND_UNAVAILABLE' | 'OUT_OF_MEMORY' | 'DEVICE_LOST'
  | 'BUSY' | 'NOT_INITIALIZED' | 'CANCELLED' | 'DISPOSED' | 'INTERNAL';

export type EngineFailure = Readonly<{
  code: ErrorCode;
  message: string;
  requestId: string | null;
  engineGeneration: number;
  details: Readonly<Record<string, string | number | boolean | null>>;
}>;
export type EngineEvent = Readonly<
  { type: 'load-progress'; operationId: string; engineGeneration: number;
    stage: 'fetch' | 'parse' | 'compile' | 'warmup'; fraction: number | null;
    bytesLoaded: number | null; bytesTotal: number | null } |
  { type: 'ready'; operationId: string; info: EngineInfo } |
  { type: 'analysis-progress'; snapshot: AnalysisSnapshot } |
  { type: 'fault'; error: EngineFailure }
>;

export class KataGoEngineError extends Error {
  readonly failure: EngineFailure;
  constructor(failure: EngineFailure) {
    super(failure.message);
    this.name = 'KataGoEngineError';
    this.failure = failure;
  }
}

export interface KataGoEngine {
  readonly apiVersion: 1;
  inspectCapabilities(): Promise<Capabilities>;
  subscribe(listener: (event: EngineEvent) => void): () => void;
  initialize(request: Readonly<{
    operationId: string;
    model: ModelSource;
    runtime: RuntimePreference;
  }>, options?: Readonly<{ signal?: AbortSignal }>): Promise<EngineInfo>;
  analyze(request: AnalyzeRequest, options?: Readonly<{ signal?: AbortSignal }>): Promise<AnalysisResult>;
  cancel(requestId: string): Promise<void>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
}

export type EngineOptions = Readonly<{
  assetBaseUrl?: string;
  workerFactory?: () => Worker;
}>;
