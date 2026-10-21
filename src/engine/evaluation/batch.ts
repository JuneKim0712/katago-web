import { BOARD_AREA, EMPTY_AREA_MAP } from '../fastBoard.js';
import { postprocessKataGoV8, type KataGoPostProcessParams as PostProcessParams } from '../evalV8.js';
import type { EvalBatchScratch, EvalState, NeuralEval } from './types.js';

export class BatchSlot {
  readonly slot: 0 | 1;
  state: 'free' | 'preparing' | 'submitted' | 'outputs-ready' | 'consumed' = 'free';
  private scratchWithArea: EvalBatchScratch | null = null;
  private scratchNoArea: EvalBatchScratch | null = null;

  constructor(slot: 0 | 1) {
    this.slot = slot;
  }

  getScratch(batch: number, includeAreaFeature: boolean): EvalBatchScratch {
    const neededSpatial = batch * BOARD_AREA * 22;
    const neededGlobal = batch * 19;
    const neededMaps = batch * BOARD_AREA;
    const neededPolicy = batch * BOARD_AREA;

    const existing = includeAreaFeature ? this.scratchWithArea : this.scratchNoArea;
    if (
      existing &&
      existing.spatialBatch.length >= neededSpatial &&
      existing.globalBatch.length >= neededGlobal &&
      existing.libertyMapScratch.length >= neededMaps &&
      existing.symmetries.length >= batch &&
      existing.policyScratch.length >= neededPolicy &&
      existing.passScratch.length >= batch &&
      (!includeAreaFeature || (existing.areaMapScratch && existing.areaMapScratch.length >= neededMaps))
    ) {
      return existing;
    }

    const scratch: EvalBatchScratch = {
      spatialBatch: new Float32Array(neededSpatial),
      globalBatch: new Float32Array(neededGlobal),
      libertyMapScratch: new Uint8Array(neededMaps),
      areaMapScratch: includeAreaFeature ? new Uint8Array(neededMaps) : null,
      ladderedStonesScratch: new Uint8Array(BOARD_AREA),
      ladderWorkingMovesScratch: new Uint8Array(BOARD_AREA),
      prevLadderedStonesScratch: new Uint8Array(BOARD_AREA),
      prevPrevLadderedStonesScratch: new Uint8Array(BOARD_AREA),
      symmetries: new Uint8Array(batch),
      policyScratch: new Float32Array(neededPolicy),
      passScratch: new Float32Array(batch),
    };

    if (includeAreaFeature) this.scratchWithArea = scratch;
    else this.scratchNoArea = scratch;
    return scratch;
  }
}

export class EvaluationContext {
  private readonly slots: [BatchSlot, BatchSlot] = [new BatchSlot(0), new BatchSlot(1)];

  getSlot(slot: 0 | 1): BatchSlot {
    return this.slots[slot];
  }

  getScratch(args: { batch: number; includeAreaFeature: boolean; slot?: number }): EvalBatchScratch {
    const slotIdx = ((args.slot ?? 0) & 1) as 0 | 1;
    return this.slots[slotIdx].getScratch(args.batch, args.includeAreaFeature);
  }
}

export function processNeuralOutputs(args: {
  model: {
    policyOutChannels: number;
    modelVersion: number;
    postProcessParams?: PostProcessParams;
  };
  states: EvalState[];
  batch: number;
  includeOwnership: boolean;
  policyOptimism: number;
  symmetries: Uint8Array;
  libertyMapScratch: Uint8Array;
  areaMapScratch: Uint8Array | null;
  includeAreaFeature: boolean;
  scratch: { policyScratch: Float32Array; passScratch: Float32Array };
  policyArr: Float32Array | Int32Array | Uint8Array;
  passArr: Float32Array | Int32Array | Uint8Array;
  valueArr: Float32Array | Int32Array | Uint8Array;
  scoreArr: Float32Array | Int32Array | Uint8Array;
  ownershipArr: Float32Array | Int32Array | Uint8Array | null;
}): NeuralEval[] {
  const {
    model,
    states,
    batch,
    includeOwnership,
    policyOptimism,
    symmetries,
    libertyMapScratch,
    areaMapScratch,
    includeAreaFeature,
    scratch,
    policyArr,
    passArr,
    valueArr,
    scoreArr,
    ownershipArr,
  } = args;

  const policyChannels = model.policyOutChannels;
  const usePolicyOptimism = policyChannels === 2 || (policyChannels === 4 && model.modelVersion >= 16);
  const mix = usePolicyOptimism ? policyOptimism : 0;
  let policyLogits = policyArr as Float32Array;
  let passLogits = passArr as Float32Array;

  if (policyChannels > 1) {
    const mixedPolicy = scratch.policyScratch.subarray(0, batch * BOARD_AREA);
    const mixedPass = scratch.passScratch.subarray(0, batch);
    for (let i = 0; i < batch; i++) {
      const baseOff = i * BOARD_AREA * policyChannels;
      const outOff = i * BOARD_AREA;
      for (let p = 0; p < BOARD_AREA; p++) {
        const src = baseOff + p * policyChannels;
        const base = policyArr[src]!;
        const opt = policyArr[src + 1]!;
        mixedPolicy[outOff + p] = base + (opt - base) * mix;
      }
      const passBase = passArr[i * policyChannels]!;
      const passOpt = passArr[i * policyChannels + 1]!;
      mixedPass[i] = passBase + (passOpt - passBase) * mix;
    }
    policyLogits = mixedPolicy;
    passLogits = mixedPass;
  }

  const results: NeuralEval[] = [];
  for (let i = 0; i < batch; i++) {
    const pOff = i * BOARD_AREA;
    const sym = symmetries[i]!;
    const policy = policyLogits.subarray(pOff, pOff + BOARD_AREA);
    const ownership = includeOwnership && ownershipArr
      ? (ownershipArr as Float32Array).subarray(pOff, pOff + BOARD_AREA)
      : undefined;
    const passLogit = passLogits[i]!;
    const vOff = i * 3;
    const sOff = i * 4;
    const evaled = postprocessKataGoV8({
      nextPlayer: states[i]!.currentPlayer,
      valueLogits: valueArr.subarray(vOff, vOff + 3),
      scoreValue: scoreArr.subarray(sOff, sOff + 4),
      postProcessParams: model.postProcessParams,
    });

    results.push({
      policy,
      symmetry: sym,
      passLogit,
      blackWinProb: evaled.blackWinProb,
      blackScoreLead: evaled.blackScoreLead,
      blackScoreMean: evaled.blackScoreMean,
      blackScoreStdev: evaled.blackScoreStdev,
      blackNoResultProb: evaled.blackNoResultProb,
      libertyMap: libertyMapScratch.subarray(i * BOARD_AREA, (i + 1) * BOARD_AREA),
      areaMap: includeAreaFeature && areaMapScratch ? areaMapScratch.subarray(i * BOARD_AREA, (i + 1) * BOARD_AREA) : EMPTY_AREA_MAP,
      ownership,
    });
  }

  return results;
}
