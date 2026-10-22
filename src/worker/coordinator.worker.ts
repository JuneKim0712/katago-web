/**
 * Dedicated Coordinator Worker for KataGo Web (Pattern C).
 * Hosts ONNX Runtime WebGPU session and Emscripten Wasm instance.
 * Communicates with client via RPC messages; evaluator pthreads suspend via JSPI.
 */

import {
  AnalysisSnapshot,
  AnalyzeRequest,
  Candidate,
  Capabilities,
  EffectiveSearch,
  EngineFailure,
  EngineInfo,
  Evaluation,
  Move,
  Ownership,
  RuntimePreference,
} from '../api/types';
import { OrtWebGpuSession } from '../inference/ort/session';
import { createKatagoWasmInstance, KatagoNativeInstance } from '../runtime/loader';
import { BASELINE_B10C384_MANIFEST, computeModelFingerprint, validateManifest } from '../models/manifest';
import { makeEngineFailure, resolveRules } from '../api/validation';
import { getNextPlayer, mapAnalyzeRequestToNative } from './nativeRequest';
import { getCachedModelArtifact, putCachedModelArtifact } from '../runtime/storage';

export interface WorkerRpcRequest {
  id: string;
  type: 'inspect-capabilities' | 'initialize' | 'analyze' | 'cancel' | 'reset' | 'dispose';
  payload?: any;
}

export interface WorkerRpcResponse {
  id: string;
  type: 'response' | 'event';
  error?: EngineFailure;
  result?: any;
  event?: any;
}

let g_generation = 0;
let g_ortSession: OrtWebGpuSession | null = null;
let g_wasmInstance: KatagoNativeInstance | null = null;
let g_activeRequestId: string | null = null;
let g_isInitialized = false;
let g_modelFingerprint = '';

function parseMoveString(moveStr: string, boardHeight: number): Move {
  const s = String(moveStr).trim();
  if (s.toLowerCase() === 'pass') {
    return { kind: 'pass' };
  }
  const colChar = s[0].toUpperCase();
  const colCode = colChar.charCodeAt(0);
  let x = colCode - 65; // 'A'
  if (colCode > 73) { // 'I'
    x -= 1;
  }
  const rowNum = parseInt(s.slice(1), 10);
  const y = boardHeight - rowNum;
  return { kind: 'play', point: { x, y } };
}

function mapRawSnapshotToContractV1(
  data: any,
  analyzeReq: AnalyzeRequest,
  sequence: number,
  searchStartTime: number,
  modelFingerprint: string
): AnalysisSnapshot {
  const root = data.rootInfo || {};
  const dims = analyzeReq.position.dimensions;

  const rawWinrate = typeof root.winrate === 'number' ? root.winrate : 0.5;
  const noResult = typeof root.noResultValue === 'number' ? root.noResultValue : 0.0;
  // Upstream winrate = pWin + 0.5 * noResult (STAGE R5 FIX)
  const blackWin = Math.max(0, Math.min(1, rawWinrate - 0.5 * noResult));
  const whiteWin = Math.max(0, Math.min(1, 1.0 - rawWinrate - 0.5 * noResult));
  const blackWinRate = (1 + blackWin - whiteWin) / 2;
  const blackScoreLead = typeof root.scoreLead === 'number' ? root.scoreLead : 0;
  const blackScoreMean = typeof (root.scoreSelfplay ?? root.scoreMean) === 'number' ? (root.scoreSelfplay ?? root.scoreMean) : 0;
  const scoreStdev = typeof root.scoreStdev === 'number' ? root.scoreStdev : 0;

  const evaluation: Evaluation = {
    blackWinProbability: blackWin,
    whiteWinProbability: whiteWin,
    noResultProbability: noResult,
    blackWinRate,
    blackScoreLead,
    blackScoreMean,
    scoreStdev,
  };

  const candidates: Candidate[] = (data.moveInfos || []).map((m: any, idx: number) => {
    const mRawWin = typeof m.winrate === 'number' ? m.winrate : 0.5;
    const mNoResult = typeof m.noResultValue === 'number' ? m.noResultValue : 0;
    const mBlackWin = Math.max(0, Math.min(1, mRawWin - 0.5 * mNoResult));
    const mWhiteWin = Math.max(0, Math.min(1, 1.0 - mRawWin - 0.5 * mNoResult));
    const mLead = typeof m.scoreLead === 'number' ? m.scoreLead : 0;
    const mMean = typeof (m.scoreSelfplay ?? m.scoreMean) === 'number' ? (m.scoreSelfplay ?? m.scoreMean) : 0;
    const mStdev = typeof m.scoreStdev === 'number' ? m.scoreStdev : 0;

    const candEval: Evaluation = {
      blackWinProbability: mBlackWin,
      whiteWinProbability: mWhiteWin,
      noResultProbability: mNoResult,
      blackWinRate: (1 + mBlackWin - mWhiteWin) / 2,
      blackScoreLead: mLead,
      blackScoreMean: mMean,
      scoreStdev: mStdev,
    };

    const pv: Move[] = Array.isArray(m.pv) ? m.pv.map((p: string) => parseMoveString(p, dims.height)) : [];

    return {
      move: parseMoveString(m.move, dims.height),
      order: typeof m.order === 'number' ? m.order : idx,
      visits: m.visits || 0,
      prior: typeof m.prior === 'number' ? m.prior : 0,
      evaluation: candEval,
      pv,
      ownership: null,
    };
  });

  const rootVisits = root.visits || 0;
  const elapsed = Math.max(1, Math.round(performance.now() - searchStartTime));

  const resolvedRules = analyzeReq.position.rules.kind === 'preset'
    ? resolveRules(analyzeReq.position.rules, CAPABILITIES)
    : analyzeReq.position.rules.value;

  const effectiveSearch: EffectiveSearch = {
    profile: analyzeReq.search.profile,
    profileFingerprint: 'profile-v1',
    seed: analyzeReq.search.seed ?? 'default-seed',
    batchSize: typeof analyzeReq.search.batchSize === 'number' ? analyzeReq.search.batchSize : 4,
    searchThreads: typeof analyzeReq.search.searchThreads === 'number' ? analyzeReq.search.searchThreads : 2,
    conservativePass: analyzeReq.search.conservativePass,
    wideRootNoise: analyzeReq.search.wideRootNoise,
    rootSymmetrySamples: analyzeReq.search.rootSymmetrySamples,
    rootRestricted: analyzeReq.search.allowedRootMoves !== null,
  };

  let ownership: Ownership | null = null;
  if (Array.isArray(data.ownership)) {
    ownership = {
      mode: 'root',
      mean: new Float32Array(data.ownership),
      stdev: Array.isArray(data.ownershipStdev) ? new Float32Array(data.ownershipStdev) : null,
    };
  }

  let policy: Float32Array | null = null;
  if (Array.isArray(data.policy)) {
    policy = new Float32Array(data.policy);
  }

  const nextPlayer = getNextPlayer(analyzeReq.position);
  const historyStatus = analyzeReq.position.kind === 'game'
    ? 'complete-from-declared-initial-state'
    : 'setup-only';

  return {
    apiVersion: 1,
    requestId: analyzeReq.requestId,
    engineGeneration: g_generation,
    sequence,
    positionFingerprint: 'pos-' + analyzeReq.requestId,
    modelFingerprint,
    dimensions: dims,
    nextPlayer,
    historyStatus,
    resolvedRules,
    effectiveSearch,
    rootVisits,
    evaluation,
    candidates,
    policy,
    ownership,
    accounting: {
      elapsedMs: elapsed,
      searchElapsedMs: elapsed,
      completedNewVisits: rootVisits,
      reusedVisits: 0,
      neuralRows: rootVisits,
      terminalVisits: 0,
      neuralCacheHits: 0,
    },
  };
}

const CAPABILITIES: Capabilities = {
  apiVersion: 1,
  packageVersion: '1.0.0',
  upstreamVersion: '1.18.1',
  upstreamCommit: '92ee95c0a4b25fec214da00951ab69e97e207729',
  status: 'available',
  unavailableReasons: [],
  supportedAccelerators: ['webgpu'],
  supportedModelFormats: ['onnx'],
  supportedRulePresets: ['japanese', 'korean', 'chinese'],
  maxBoardWidth: 19,
  maxBoardHeight: 19,
  supportsRectangularBoards: true,
  supportsSetupPositions: true,
  supportsRootMoveRestriction: true,
  supportsCandidateOwnership: false,
  supportsOwnershipStdev: false,
  crossOriginIsolationRequired: true,
  maxVisits: 1000000,
  maxTimeMs: 3600000,
  maxBatchSize: 16,
  maxSearchThreads: 16,
  maxPvLength: 100,
};

self.onmessage = async (e: MessageEvent<WorkerRpcRequest>) => {
  const req = e.data;
  if (!req || !req.id || !req.type) return;

  const replySuccess = (result: any) => {
    self.postMessage({ id: req.id, type: 'response', result } as WorkerRpcResponse);
  };

  const replyError = (failure: EngineFailure) => {
    self.postMessage({ id: req.id, type: 'response', error: failure } as WorkerRpcResponse);
  };

  const emitEvent = (event: any) => {
    self.postMessage({ id: req.id, type: 'event', event } as WorkerRpcResponse);
  };

  try {
    switch (req.type) {
      case 'inspect-capabilities': {
        replySuccess(CAPABILITIES);
        break;
      }

      case 'initialize': {
        const { operationId, model, runtime } = req.payload;
        g_generation++;

        emitEvent({
          type: 'load-progress',
          operationId,
          engineGeneration: g_generation,
          stage: 'fetch',
          fraction: 0.1,
          bytesLoaded: null,
          bytesTotal: null,
        });

        // 1. Resolve model manifest and fetch ONNX model bytes
        let manifest = BASELINE_B10C384_MANIFEST;
        if (model && model.kind === 'manifest') {
          manifest = validateManifest(model.manifest);
        }

        const variant = manifest.variants.find(v => v.format === 'onnx') || manifest.variants[0];
        const artifact = variant.artifacts[0];

        let modelBytes: Uint8Array;
        if (artifact.source.kind === 'bytes') {
          modelBytes = artifact.source.bytes;
        } else {
          // Check persistent cache (Cache API / IndexedDB) before fetching over network
          const cached = await getCachedModelArtifact(artifact.sha256);
          if (cached) {
            modelBytes = cached;
          } else {
            emitEvent({
              type: 'load-progress',
              operationId,
              engineGeneration: g_generation,
              stage: 'fetch',
              fraction: 0.3,
              bytesLoaded: 0,
              bytesTotal: artifact.byteLength,
            });

            const resp = await fetch(artifact.source.url);
            if (!resp.ok) {
              replyError(makeEngineFailure('MODEL_FETCH_FAILED', `Fetch failed with status ${resp.status}`, null, g_generation));
              return;
            }
            const buf = await resp.arrayBuffer();
            modelBytes = new Uint8Array(buf);

            // Persist model artifact for subsequent sessions
            putCachedModelArtifact(artifact.sha256, modelBytes).catch(() => {});
          }
        }

        emitEvent({
          type: 'load-progress',
          operationId,
          engineGeneration: g_generation,
          stage: 'compile',
          fraction: 0.6,
          bytesLoaded: modelBytes.byteLength,
          bytesTotal: artifact.byteLength,
        });

        // 2. Initialize ORT WebGPU session
        if (g_ortSession) {
          await g_ortSession.dispose();
          g_ortSession = null;
        }

        g_ortSession = await OrtWebGpuSession.create(modelBytes, {
          disableCpuFallback: true,
        });

        emitEvent({
          type: 'load-progress',
          operationId,
          engineGeneration: g_generation,
          stage: 'warmup',
          fraction: 0.8,
          bytesLoaded: modelBytes.byteLength,
          bytesTotal: artifact.byteLength,
        });

        // Warm up WebGPU shaders & pipeline with a dummy inference
        await g_ortSession.warmup(1, 19, 19);

        // 3. Initialize KataGo Wasm Instance
        if (!g_wasmInstance) {
          // Dynamic import of compiled KataGo Emscripten module
          // @ts-ignore
          const katagoModuleFactory = (await import('../../dist/wasm/katago.js')).default;
          g_wasmInstance = await createKatagoWasmInstance(katagoModuleFactory, g_ortSession);
          g_wasmInstance.init();
        }

        // Load model in native core
        const modelJson = JSON.stringify({
          name: manifest.modelId,
          modelVersion: 17,
          numInputChannels: 22,
          numInputGlobalChannels: 19,
          numInputMetaChannels: 0,
          numPolicyChannels: 2,
          numValueChannels: 3,
          numScoreValueChannels: 6,
          numOwnershipChannels: 1,
        });

        g_wasmInstance.loadModel(manifest.modelId, modelJson, 4, 19, 19, 1);
        g_isInitialized = true;

        g_modelFingerprint = computeModelFingerprint(manifest, variant.variantId);
        const info: EngineInfo = {
          engineGeneration: g_generation,
          packageVersion: '1.0.0',
          upstreamCommit: '92ee95c0a4b25fec214da00951ab69e97e207729',
          modelId: manifest.modelId,
          modelFingerprint: g_modelFingerprint,
          sourceModelSha256: manifest.sourceModelSha256,
          variantId: variant.variantId,
          accelerator: 'webgpu',
          inferenceImplementation: 'onnxruntime-webgpu-jspi',
          precision: 'fp32',
          modelVersion: 17,
          blocks: 10,
          channels: 384,
          recommendedBatchSize: 4,
          recommendedSearchThreads: 2,
          recommendedProgressIntervalMs: 100,
          capabilities: CAPABILITIES,
        };

        emitEvent({
          type: 'ready',
          operationId,
          info,
        });

        replySuccess(info);
        break;
      }

      case 'analyze': {
        if (!g_isInitialized || !g_wasmInstance) {
          replyError(makeEngineFailure('NOT_INITIALIZED', 'Engine not initialized', null, g_generation));
          return;
        }
        if (g_activeRequestId !== null) {
          replyError(makeEngineFailure('BUSY', 'An analysis request is already active', req.payload?.requestId, g_generation));
          return;
        }

        const analyzeReq = req.payload as AnalyzeRequest;
        g_activeRequestId = analyzeReq.requestId;
        let sequence = 1;
        const searchStartTime = performance.now();

        // Dispatch search in native thread via explicit nativeRequest mapping (STAGE R1 FIX)
        const nativeQuery = mapAnalyzeRequestToNative(analyzeReq, CAPABILITIES);
        const queryJson = JSON.stringify(nativeQuery);
        const code = g_wasmInstance.postAnalyze(queryJson);
        if (code !== 0) {
          g_activeRequestId = null;
          replyError(makeEngineFailure('INTERNAL', `Failed to start analysis (code ${code})`, analyzeReq.requestId, g_generation));
          return;
        }

        // Asynchronously poll for results without blocking event loop
        const pollInterval = setInterval(() => {
          if (!g_wasmInstance) {
            clearInterval(pollInterval);
            return;
          }
          const line = g_wasmInstance.pollResponse();
          if (!line) return;

          try {
            const data = JSON.parse(line);
            if (data.error) {
              clearInterval(pollInterval);
              g_activeRequestId = null;
              replyError(makeEngineFailure('INTERNAL', data.error, analyzeReq.requestId, g_generation));
              return;
            }

            const snapshot = mapRawSnapshotToContractV1(
              data,
              analyzeReq,
              sequence++,
              searchStartTime,
              g_modelFingerprint
            );

            if (data.isDuringSearch) {
              emitEvent({
                type: 'analysis-progress',
                snapshot,
              });
            } else {
              clearInterval(pollInterval);
              g_activeRequestId = null;
              replySuccess({
                kind: 'completed',
                stopReason: 'visit-limit',
                snapshot,
              });
            }
          } catch (err: unknown) {
            clearInterval(pollInterval);
            g_activeRequestId = null;
            const msg = err instanceof Error ? err.message : String(err);
            replyError(makeEngineFailure('INTERNAL', `Parse error on engine response: ${msg}`, analyzeReq.requestId, g_generation));
          }
        }, 10);
        break;
      }

      case 'cancel': {
        const { requestId } = req.payload;
        if (g_wasmInstance && g_activeRequestId === requestId) {
          g_wasmInstance.cancel(requestId);
          g_activeRequestId = null;
        }
        replySuccess(undefined);
        break;
      }

      case 'reset': {
        g_generation++;
        if (g_wasmInstance) {
          g_wasmInstance.reset();
        }
        g_activeRequestId = null;
        g_isInitialized = false;
        replySuccess(undefined);
        break;
      }

      case 'dispose': {
        g_generation++;
        if (g_wasmInstance) {
          g_wasmInstance.destroy();
          g_wasmInstance = null;
        }
        if (g_ortSession) {
          await g_ortSession.dispose();
          g_ortSession = null;
        }
        g_activeRequestId = null;
        g_isInitialized = false;
        replySuccess(undefined);
        break;
      }

      default:
        replyError(makeEngineFailure('INVALID_REQUEST', `Unknown RPC request type: ${req.type}`, null, g_generation));
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    replyError(makeEngineFailure('INTERNAL', msg, null, g_generation));
  }
};
