import type { Player } from '../../types/index.js';
import type { RecentMove } from '../featuresV7Fast.js';

export interface EvalState {
  stones: Uint8Array;
  koPoint: number;
  prevStones: Uint8Array;
  prevKoPoint: number;
  prevPrevStones: Uint8Array;
  prevPrevKoPoint: number;
  currentPlayer: Player;
  recentMoves: RecentMove[];
  libertyMap?: Uint8Array;
  prevLibertyMap?: Uint8Array;
  prevPrevLibertyMap?: Uint8Array;
  komi?: number;
  conservativePassAndIsRoot?: boolean;
  symmetry?: number;
}

export interface NeuralEval {
  policy: Float32Array; // len 361, in symmetry space if symmetry != 0
  symmetry: number; // 0..7, where 0 is identity
  passLogit: number;
  blackWinProb: number;
  blackScoreLead: number;
  blackScoreMean: number;
  blackScoreStdev: number;
  blackNoResultProb: number;
  libertyMap: Uint8Array;
  areaMap: Uint8Array;
  ownership?: Float32Array; // len 361, raw logits (player-to-move perspective, symmetry space if symmetry != 0)
}

export interface EvalBatchScratch {
  spatialBatch: Float32Array;
  globalBatch: Float32Array;
  ladderedStonesScratch: Uint8Array;
  ladderWorkingMovesScratch: Uint8Array;
  prevLadderedStonesScratch: Uint8Array;
  prevPrevLadderedStonesScratch: Uint8Array;
  libertyMapScratch: Uint8Array;
  areaMapScratch: Uint8Array | null;
  symmetries: Uint8Array;
  policyScratch: Float32Array;
  passScratch: Float32Array;
}

export interface EvaluationLease {
  readonly slot: 0 | 1;
  readonly sequence: number;
  readonly result: Promise<readonly NeuralEval[]>;
  releaseAfterConsumption(): void;
}
