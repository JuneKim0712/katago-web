// Types
export * from './types/index.js';

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
  resetKataGoEngineClient,
} from './worker/client.js';
