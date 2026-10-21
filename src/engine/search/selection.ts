import { BLACK } from '../board/context.js';
import type { Rand } from '../../utils/random.js';
import type { Node, Edge } from './tree.js';
import {
  WIN_LOSS_UTILITY_FACTOR,
  STATIC_SCORE_UTILITY_FACTOR,
  DYNAMIC_SCORE_UTILITY_FACTOR,
  CPUCT_UTILITY_STDEV_PRIOR,
  CPUCT_UTILITY_STDEV_PRIOR_WEIGHT,
  CPUCT_UTILITY_STDEV_SCALE,
  FPU_REDUCTION_MAX,
  ROOT_FPU_REDUCTION_MAX,
  FPU_LOSS_PROP,
  ROOT_FPU_LOSS_PROP,
  FPU_PARENT_WEIGHT_BY_VISITED_POLICY,
  FPU_PARENT_WEIGHT_BY_VISITED_POLICY_POW,
  FPU_PARENT_WEIGHT,
  exploreScaling,
} from './utility.js';

export function selectEdge(node: Node, isRoot: boolean, wideRootNoise: number, rand: Rand): Edge {
  const edges = node.edges;
  if (!edges || edges.length === 0) throw new Error('selectEdge called on unexpanded node');

  const pla = node.playerToMove;
  const sign = pla === BLACK ? 1 : -1;

  let totalChildWeight = 0;
  let policyProbMassVisited = 0;

  // Maintain original edge summation order exactly
  for (const e of edges) {
    const child = e.child;
    if (!child) continue;
    const w = child.visits + child.inFlight;
    if (w <= 0) continue;
    totalChildWeight += w;
    policyProbMassVisited += e.prior;
  }

  const visits = node.visits;
  const weightSum = visits;
  const parentUtility = visits > 0 ? node.utilitySum / visits : 0;
  const parentUtilitySqAvg = visits > 0 ? node.utilitySqSum / visits : parentUtility * parentUtility;

  const variancePrior = CPUCT_UTILITY_STDEV_PRIOR * CPUCT_UTILITY_STDEV_PRIOR;
  const variancePriorWeight = CPUCT_UTILITY_STDEV_PRIOR_WEIGHT;
  let parentUtilityStdev: number;
  if (visits <= 0 || weightSum <= 1) {
    parentUtilityStdev = CPUCT_UTILITY_STDEV_PRIOR;
  } else {
    const utilitySq = parentUtility * parentUtility;
    let utilitySqAvg = parentUtilitySqAvg;
    if (utilitySqAvg < utilitySq) utilitySqAvg = utilitySq;
    parentUtilityStdev = Math.sqrt(
      Math.max(
        0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum) /
          (variancePriorWeight + weightSum - 1.0) -
          utilitySq
      )
    );
  }

  const parentUtilityStdevFactor =
    1.0 + CPUCT_UTILITY_STDEV_SCALE * (parentUtilityStdev / CPUCT_UTILITY_STDEV_PRIOR - 1.0);

  let parentUtilityForFPU = parentUtility;
  const parentNNUtility = node.nnUtility ?? parentUtility;
  if (FPU_PARENT_WEIGHT_BY_VISITED_POLICY) {
    const avgWeight = Math.min(1.0, Math.pow(policyProbMassVisited, FPU_PARENT_WEIGHT_BY_VISITED_POLICY_POW));
    parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * parentNNUtility;
  } else if (FPU_PARENT_WEIGHT > 0.0) {
    parentUtilityForFPU = FPU_PARENT_WEIGHT * parentNNUtility + (1.0 - FPU_PARENT_WEIGHT) * parentUtility;
  }

  const fpuReductionMax = isRoot ? ROOT_FPU_REDUCTION_MAX : FPU_REDUCTION_MAX;
  const fpuLossProp = isRoot ? ROOT_FPU_LOSS_PROP : FPU_LOSS_PROP;
  const reduction = fpuReductionMax * Math.sqrt(Math.max(0, policyProbMassVisited));
  let fpuValue = pla === BLACK ? parentUtilityForFPU - reduction : parentUtilityForFPU + reduction;

  const utilityRadius = WIN_LOSS_UTILITY_FACTOR + STATIC_SCORE_UTILITY_FACTOR + DYNAMIC_SCORE_UTILITY_FACTOR;
  const lossValue = pla === BLACK ? -utilityRadius : utilityRadius;
  fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;

  const scaling = exploreScaling(totalChildWeight, parentUtilityStdevFactor);

  let bestEdge = edges[0]!;
  let bestScore = Number.NEGATIVE_INFINITY;

  const applyWideRootNoise = isRoot && wideRootNoise > 0;
  const wideRootNoisePolicyExponent = applyWideRootNoise ? 1.0 / (4.0 * wideRootNoise + 1.0) : 1.0;

  for (const e of edges) {
    const child = e.child;
    const childWeight = child ? child.visits + child.inFlight : 0;
    let childUtility = child && child.visits > 0 ? child.utilitySum / child.visits : fpuValue;
    let prior = e.prior;

    if (applyWideRootNoise) {
      prior = Math.pow(prior, wideRootNoisePolicyExponent);
      if (rand.nextBool(0.5)) {
        const bonus = wideRootNoise * Math.abs(rand.nextGaussian());
        childUtility += pla === BLACK ? bonus : -bonus;
      }
    }

    const explore = (scaling * prior) / (1.0 + childWeight);
    const score = explore + sign * childUtility;
    if (score > bestScore) {
      bestScore = score;
      bestEdge = e;
    }
  }

  return bestEdge;
}
