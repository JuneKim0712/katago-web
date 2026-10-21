import { expectedWhiteScoreValue, getSqrtBoardArea } from '../scoreValue.js';

export const WIN_LOSS_UTILITY_FACTOR: number = 1.0;
export const STATIC_SCORE_UTILITY_FACTOR: number = 0.1;
export const DYNAMIC_SCORE_UTILITY_FACTOR: number = 0.3;
export const DYNAMIC_SCORE_CENTER_ZERO_WEIGHT: number = 0.2;
export const DYNAMIC_SCORE_CENTER_SCALE: number = 0.75;
export const NO_RESULT_UTILITY_FOR_WHITE: number = 0.0;

export const VALUE_WEIGHT_EXPONENT: number = 0.25;
export const USE_NOISE_PRUNING = true;
export const NOISE_PRUNE_UTILITY_SCALE = 0.15;
export const NOISE_PRUNING_CAP = 1e50;

export const CPUCT_EXPLORATION = 1.0;
export const CPUCT_EXPLORATION_LOG = 0.45;
export const CPUCT_EXPLORATION_BASE = 500;
export const CPUCT_UTILITY_STDEV_PRIOR = 0.4;
export const CPUCT_UTILITY_STDEV_PRIOR_WEIGHT = 2.0;
export const CPUCT_UTILITY_STDEV_SCALE = 0.85;
export const FPU_REDUCTION_MAX = 0.2;
export const ROOT_FPU_REDUCTION_MAX = 0.1;
export const FPU_LOSS_PROP = 0.0;
export const ROOT_FPU_LOSS_PROP = 0.0;
export const FPU_PARENT_WEIGHT_BY_VISITED_POLICY = true;
export const FPU_PARENT_WEIGHT_BY_VISITED_POLICY_POW = 2.0;
export const FPU_PARENT_WEIGHT = 0.0;
export const TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

export function computeRecentScoreCenter(expectedWhiteScore: number): number {
  let recentScoreCenter = expectedWhiteScore * (1.0 - DYNAMIC_SCORE_CENTER_ZERO_WEIGHT);
  const cap = getSqrtBoardArea() * DYNAMIC_SCORE_CENTER_SCALE;
  if (recentScoreCenter > expectedWhiteScore + cap) recentScoreCenter = expectedWhiteScore + cap;
  if (recentScoreCenter < expectedWhiteScore - cap) recentScoreCenter = expectedWhiteScore - cap;
  return recentScoreCenter;
}

export function computeBlackUtilityFromEval(args: {
  blackWinProb: number;
  blackNoResultProb: number;
  blackScoreMean: number;
  blackScoreStdev: number;
  recentScoreCenter: number; // white score center
}): number {
  const sqrtBoardArea = getSqrtBoardArea();
  const blackLossProb = 1.0 - args.blackWinProb - args.blackNoResultProb;
  const whiteWinLossValue = blackLossProb - args.blackWinProb;
  const whiteScoreMean = -args.blackScoreMean;
  const whiteScoreStdev = args.blackScoreStdev;

  const staticScoreValue = expectedWhiteScoreValue({
    whiteScoreMean,
    whiteScoreStdev,
    center: 0.0,
    scale: 2.0,
    sqrtBoardArea,
  });

  const dynamicScoreValue =
    DYNAMIC_SCORE_UTILITY_FACTOR === 0.0
      ? 0.0
      : expectedWhiteScoreValue({
          whiteScoreMean,
          whiteScoreStdev,
          center: args.recentScoreCenter,
          scale: DYNAMIC_SCORE_CENTER_SCALE,
          sqrtBoardArea,
        });

  const whiteUtility =
    whiteWinLossValue * WIN_LOSS_UTILITY_FACTOR +
    args.blackNoResultProb * NO_RESULT_UTILITY_FOR_WHITE +
    staticScoreValue * STATIC_SCORE_UTILITY_FACTOR +
    dynamicScoreValue * DYNAMIC_SCORE_UTILITY_FACTOR;

  return -whiteUtility;
}

export type ChildWeightStats = {
  weightAdjusted: number;
  selfUtility: number;
  policy: number;
  value: number;
  scoreLead: number;
  scoreMean: number;
  scoreMeanSq: number;
};

export type RootSelfStats = {
  value: number;
  scoreLead: number;
  scoreMean: number;
  scoreMeanSq: number;
  utility: number;
  weight: number;
};

const SQRT_3 = Math.sqrt(3);

export function tDistCdf3(z: number): number {
  const u = z / SQRT_3;
  const term = u / (1 + u * u);
  return 0.5 + (Math.atan(u) + term) / Math.PI;
}

export function pruneNoiseWeight(stats: ChildWeightStats[]): number {
  if (stats.length <= 1) return stats.reduce((acc, s) => acc + s.weightAdjusted, 0);
  stats.sort((a, b) => b.policy - a.policy);

  let utilitySumSoFar = 0;
  let weightSumSoFar = 0;
  let rawPolicySumSoFar = 0;

  for (const s of stats) {
    const utility = s.selfUtility;
    const oldWeight = s.weightAdjusted;
    const rawPolicy = Math.max(1e-30, s.policy);
    let newWeight = oldWeight;

    if (weightSumSoFar > 0 && rawPolicySumSoFar > 0) {
      const avgUtilitySoFar = utilitySumSoFar / weightSumSoFar;
      const utilityGap = avgUtilitySoFar - utility;
      if (utilityGap > 0) {
        const weightShareFromRawPolicy = (weightSumSoFar * rawPolicy) / rawPolicySumSoFar;
        const lenientWeightShareFromRawPolicy = 2.0 * weightShareFromRawPolicy;
        if (oldWeight > lenientWeightShareFromRawPolicy) {
          const excessWeight = oldWeight - lenientWeightShareFromRawPolicy;
          let weightToSubtract = excessWeight * (1.0 - Math.exp(-utilityGap / NOISE_PRUNE_UTILITY_SCALE));
          if (weightToSubtract > NOISE_PRUNING_CAP) weightToSubtract = NOISE_PRUNING_CAP;
          newWeight = oldWeight - weightToSubtract;
          s.weightAdjusted = newWeight;
        }
      }
    }

    utilitySumSoFar += utility * newWeight;
    weightSumSoFar += newWeight;
    rawPolicySumSoFar += rawPolicy;
  }

  return weightSumSoFar;
}

export function downweightBadChildrenAndNormalizeWeight(args: {
  stats: ChildWeightStats[];
  currentTotalWeight: number;
  desiredTotalWeight: number;
  amountToSubtract: number;
  amountToPrune: number;
}): void {
  const stats = args.stats;
  const desiredTotalWeight = args.desiredTotalWeight;
  if (stats.length === 0 || args.currentTotalWeight <= 0) return;

  if (VALUE_WEIGHT_EXPONENT === 0) {
    let currentTotalWeight = args.currentTotalWeight;
    for (const s of stats) {
      if (s.weightAdjusted < args.amountToPrune) {
        currentTotalWeight -= s.weightAdjusted;
        s.weightAdjusted = 0;
        continue;
      }
      const newWeight = s.weightAdjusted - args.amountToSubtract;
      if (newWeight <= 0) {
        currentTotalWeight -= s.weightAdjusted;
        s.weightAdjusted = 0;
      } else {
        currentTotalWeight -= args.amountToSubtract;
        s.weightAdjusted = newWeight;
      }
    }

    if (currentTotalWeight > 0 && currentTotalWeight !== desiredTotalWeight) {
      const factor = desiredTotalWeight / currentTotalWeight;
      for (const s of stats) s.weightAdjusted *= factor;
    }
    return;
  }

  const stdevs: number[] = new Array(stats.length);
  let simpleValueSum = 0;
  for (let i = 0; i < stats.length; i++) {
    const s = stats[i]!;
    const weight = s.weightAdjusted;
    if (weight <= 0) continue;
    const precision = 1.5 * Math.sqrt(weight);
    stdevs[i] = Math.sqrt(1e-8 + 1.0 / precision);
    simpleValueSum += s.selfUtility * weight;
  }

  const simpleValue = simpleValueSum / args.currentTotalWeight;
  let totalNewUnnormWeight = 0;

  for (let i = 0; i < stats.length; i++) {
    const s = stats[i]!;
    if (s.weightAdjusted < args.amountToPrune) {
      s.weightAdjusted = 0;
      continue;
    }
    const newWeight = s.weightAdjusted - args.amountToSubtract;
    if (newWeight <= 0) {
      s.weightAdjusted = 0;
      continue;
    }
    s.weightAdjusted = newWeight;

    const stdev = stdevs[i];
    if (!stdev || stdev <= 0) continue;
    const z = (s.selfUtility - simpleValue) / stdev;
    const p = tDistCdf3(z) + 0.0001;
    s.weightAdjusted *= Math.pow(p, VALUE_WEIGHT_EXPONENT);
    totalNewUnnormWeight += s.weightAdjusted;
  }

  if (totalNewUnnormWeight <= 0) return;
  const factor = desiredTotalWeight / totalNewUnnormWeight;
  for (const s of stats) s.weightAdjusted *= factor;
}

export function computeWeightedRootStats(args: { children: ChildWeightStats[]; rootSelf: RootSelfStats }): {
  rootValue: number;
  rootWinRate: number;
  rootScoreLead: number;
  rootScoreSelfplay: number;
  rootScoreStdev: number;
} {
  const stats = args.children;
  if (stats.length === 0) {
    const rootValue = args.rootSelf.value;
    const rootScoreSelfplay = args.rootSelf.scoreMean;
    const rootScoreMeanSq = args.rootSelf.scoreMeanSq;
    const rootScoreStdev = Math.sqrt(Math.max(0, rootScoreMeanSq - rootScoreSelfplay * rootScoreSelfplay));
    return {
      rootValue,
      rootWinRate: (rootValue + 1) * 0.5,
      rootScoreLead: args.rootSelf.scoreLead,
      rootScoreSelfplay,
      rootScoreStdev,
    };
  }

  let totalWeight = 0;
  for (const s of stats) totalWeight += s.weightAdjusted;
  if (USE_NOISE_PRUNING) totalWeight = pruneNoiseWeight(stats);

  downweightBadChildrenAndNormalizeWeight({
    stats,
    currentTotalWeight: totalWeight,
    desiredTotalWeight: totalWeight,
    amountToSubtract: 0,
    amountToPrune: 0,
  });

  let weightSum = 0;
  let valueSum = 0;
  let scoreMeanSum = 0;
  let scoreMeanSqSum = 0;
  let scoreLeadSum = 0;

  for (const s of stats) {
    if (s.weightAdjusted <= 0) continue;
    weightSum += s.weightAdjusted;
    valueSum += s.weightAdjusted * s.value;
    scoreMeanSum += s.weightAdjusted * s.scoreMean;
    scoreMeanSqSum += s.weightAdjusted * s.scoreMeanSq;
    scoreLeadSum += s.weightAdjusted * s.scoreLead;
  }

  weightSum += args.rootSelf.weight;
  valueSum += args.rootSelf.weight * args.rootSelf.value;
  scoreMeanSum += args.rootSelf.weight * args.rootSelf.scoreMean;
  scoreMeanSqSum += args.rootSelf.weight * args.rootSelf.scoreMeanSq;
  scoreLeadSum += args.rootSelf.weight * args.rootSelf.scoreLead;

  if (weightSum <= 0) {
    const rootValue = args.rootSelf.value;
    const rootScoreSelfplay = args.rootSelf.scoreMean;
    const rootScoreMeanSq = args.rootSelf.scoreMeanSq;
    const rootScoreStdev = Math.sqrt(Math.max(0, rootScoreMeanSq - rootScoreSelfplay * rootScoreSelfplay));
    return {
      rootValue,
      rootWinRate: (rootValue + 1) * 0.5,
      rootScoreLead: args.rootSelf.scoreLead,
      rootScoreSelfplay,
      rootScoreStdev,
    };
  }

  const rootValue = valueSum / weightSum;
  const rootScoreSelfplay = scoreMeanSum / weightSum;
  const rootScoreMeanSq = scoreMeanSqSum / weightSum;
  const rootScoreStdev = Math.sqrt(Math.max(0, rootScoreMeanSq - rootScoreSelfplay * rootScoreSelfplay));
  return {
    rootValue,
    rootWinRate: (rootValue + 1) * 0.5,
    rootScoreLead: scoreLeadSum / weightSum,
    rootScoreSelfplay,
    rootScoreStdev,
  };
}

export function cpuctExploration(totalChildWeight: number): number {
  return (
    CPUCT_EXPLORATION +
    CPUCT_EXPLORATION_LOG * Math.log((totalChildWeight + CPUCT_EXPLORATION_BASE) / CPUCT_EXPLORATION_BASE)
  );
}

export function exploreScaling(totalChildWeight: number, parentUtilityStdevFactor: number): number {
  return (
    cpuctExploration(totalChildWeight) *
    Math.sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET) *
    parentUtilityStdevFactor
  );
}
