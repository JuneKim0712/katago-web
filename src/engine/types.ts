import type { BoardState, FloatArray, GameRules, KataGoBackendPreference, Move, Player, RegionOfInterest } from '../types/index.js';
import type { ResourceProfilePayload } from '../worker/resourceProfile.js';

export type KataGoLoadStage = 'fetch' | 'parse' | 'warmup' | 'probe';

export interface KataGoInitRequest {
  type: 'katago:init';
  modelUrl: string;
  backend?: KataGoBackendPreference;
  modelByteSize?: number;
}

export interface KataGoInitBufferRequest {
  type: 'katago:init_buffer';
  modelUrl: string;
  modelBuffer: ArrayBuffer;
  backend?: KataGoBackendPreference;
}

export interface KataGoInitResponse {
  type: 'katago:init_result';
  ok: boolean;
  backend?: string;
  modelName?: string;
  profile?: ResourceProfilePayload;
  error?: string;
}

export interface KataGoLoadProgress {
  type: 'katago:load_progress';
  stage: KataGoLoadStage;
  percent: number;
}

export interface KataGoProfileMessage {
  type: 'katago:profile';
  profile: ResourceProfilePayload;
}

export interface KataGoAnalyzeRequest {
  type: 'katago:analyze';
  id: number;
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
}

export interface KataGoAnalysisPayload {
  rootWinRate: number;
  rootScoreLead: number;
  rootScoreSelfplay: number;
  rootScoreStdev: number;
  rootVisits: number;
  ownership: FloatArray; // len 361, +1 black owns, -1 white owns
  ownershipStdev: FloatArray; // len 361
  policy: FloatArray; // len 362, illegal = -1, pass at index 361
  moves: Array<{
    x: number;
    y: number;
    winRate: number;
    winRateLost: number;
    scoreLead: number;
    scoreSelfplay: number;
    scoreStdev: number;
    visits: number;
    pointsLost: number;
    relativePointsLost: number;
    order: number;
    prior: number;
    pv: string[];
    ownership?: FloatArray; // len 361, +1 black owns, -1 white owns (position after this move)
  }>;
}

export interface KataGoAnalyzeUpdate {
  type: 'katago:analyze_update';
  id: number;
  ok: boolean;
  canceled?: boolean;
  backend?: string;
  modelName?: string;
  analysis?: KataGoAnalysisPayload;
  error?: string;
}

export interface KataGoAnalyzeResponse {
  type: 'katago:analyze_result';
  id: number;
  ok: boolean;
  canceled?: boolean;
  backend?: string;
  modelName?: string;
  analysis?: KataGoAnalysisPayload;
  error?: string;
}

export interface KataGoCancelRequest {
  type: 'katago:cancel';
  analysisGroup: 'interactive' | 'background';
}

export type KataGoWorkerRequest =
  | KataGoInitRequest
  | KataGoInitBufferRequest
  | KataGoAnalyzeRequest
  | KataGoCancelRequest;
export type KataGoWorkerResponse =
  | KataGoInitResponse
  | KataGoLoadProgress
  | KataGoProfileMessage
  | KataGoAnalyzeUpdate
  | KataGoAnalyzeResponse;
