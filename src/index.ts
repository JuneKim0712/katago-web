// Pattern C Public Engine Contract v1 Exports
export { createEngine } from './client/engine.js';
export {
  API_VERSION,
  KataGoEngineError,
  type KataGoEngine,
  type EngineOptions,
  type EngineInfo,
  type EngineEvent,
  type EngineFailure,
  type Capabilities,
  type AnalyzeRequest,
  type AnalysisSnapshot,
  type Candidate,
  type Evaluation,
  type Ownership,
  type EffectiveSearch,
  type Position,
  type GamePosition,
  type SetupPosition,
  type RulesInput,
  type PlayedMove,
  type StonePlacement,
  type Point,
  type Stone,
  type Dimensions,
  type Limit,
  type ModelSource,
  type ModelManifest,
  type ModelArtifact,
  type RuntimePreference,
  type Move as ContractMove,
  type AnalysisResult as ContractAnalysisResult,
  type Rules as ContractRules,
  type Color as ContractColor,
} from './api/types.js';

// Types (Legacy)
export * from './types/index.js';
export * from './engine/types.js';
export {
  type ResourceProfile,
  type ResourceProfilePayload,
  type ModelTier,
  buildResourceProfile,
  computeModelTier,
  computeModelTierFromArch,
  KATAGO_MAX_CUSTOM_MODEL_BYTES,
  recommendBackendForModelSize,
} from './worker/resourceProfile.js';
export {
  formatKataGoUserError,
} from './worker/gpuError.js';
export {
  validateCustomModelFile,
  cacheCustomModel,
  restoreCustomModel,
  revokeCachedModelUrl,
  formatModelByteSize,
} from './model/modelCache.js';
export {
  ENGINE_MAX_VISITS,
  ENGINE_MAX_TIME_MS,
} from './engine/limits.js';

// Fast Board & Game Rules
export {
  BLACK,
  WHITE,
  EMPTY,
  BOARD_SIZE,
  BOARD_AREA,
  PASS_MOVE,
  type StoneColor,
  type SimPosition,
  type UndoSnapshot,
  playMove,
  undoMove,
  computeLibertyMap,
  computeLibertyMapInto,
  computeAreaMapV7KataGo,
  computeAreaMapV7KataGoInto,
  computeLadderFeaturesV7KataGoInto,
  computeLadderedStonesV7KataGoInto,
} from './engine/fastBoard.js';

// MCTS Search Engine & GTP move string helpers
export {
  MctsSearch,
  analyzeMcts,
  moveToGtp,
  type OwnershipMode,
  type NeuralEval,
} from './engine/analyzeMcts.js';

// Feature Generator
export {
  fillInputsV7Fast,
  type RecentMove,
} from './engine/featuresV7Fast.js';

// Neural Network & Model Parser
export {
  KataGoModelV8Tf,
  type ParsedKataGoModelV8,
} from './model/modelV8.js';

export {
  KataGoBinModelParser,
} from './model/binModelParser.js';

export {
  parseKataGoModelV8,
} from './model/loadModelV8.js';

// Browser Worker & Client
export {
  KataGoEngineClient,
  KataGoCanceledError,
  isKataGoCanceledError,
  getKataGoEngineClient,
  getKataGoGraphEngineClient,
  resetKataGoEngineClient,
  resetKataGoEngineClientForTests,
} from './worker/client.js';

// Manifests
export { BASELINE_B10C384_MANIFEST } from './models/manifest.js';

// Upstream Core Contract & Runtime
export {
  RULES_PRESETS,
  parseRules,
  rulesToString,
  equalsRules,
  type RulesPresetName,
  type KoRule,
  type ScoringRule,
  type TaxRule,
  type WhiteHandicapBonusRule,
} from './core-contract/rules.js';

export {
  PASS_LOC,
  DEFAULT_BOARD_SIZE,
  createMove,
  createPassMove,
  type CanonicalPosition,
} from './core-contract/position.js';

export {
  validateAndNormalizePosition,
  createBoardHistoryFromPosition,
  PositionValidationError,
} from './core-contract/validation.js';

export {
  EvaluationCancelledError,
  type EvaluationBackend,
  type EvaluationCapabilities,
  type EvaluationBatch,
  type EvaluationResult,
  type SingleEvaluationValue,
} from './core-contract/evaluation.js';

export { Board } from './core-runtime/board.js';
export { BoardHistory, type ScoreResult } from './core-runtime/boardHistory.js';
export {
  extractFeatures,
  applySymmetry,
  NUM_SPATIAL_CHANNELS,
  NUM_GLOBAL_CHANNELS,
  type TensorLayout,
  type EncodedFeatures,
} from './core-runtime/features.js';
export {
  SearchSession,
  DEFAULT_SEARCH_PARAMS,
  type SearchParams,
  type SearchReport,
  type CandidateMoveResult,
} from './core-runtime/search.js';


