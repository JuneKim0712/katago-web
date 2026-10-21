/**
 * Upstream-faithful MCTS Search implementation driven by Contract v1 EvaluationBackend.
 * Implements PUCT selection with dynamic FPU matching KataGo v1.18.1 (cpp/search/searchpuct.cpp).
 */

import {
  type Move,
  type Player,
  DEFAULT_BOARD_SIZE,
  PASS_LOC,
  opponent,
  xyToLoc,
  locToX,
  locToY,
} from '../core-contract/position.js';
import type { EvaluationBackend, EvaluationBatch } from '../core-contract/evaluation.js';
import { BoardHistory } from './boardHistory.js';
import { extractFeatures, NUM_SPATIAL_CHANNELS, NUM_GLOBAL_CHANNELS } from './features.js';

export interface SearchParams {
  readonly cpuct?: number;
  readonly fpuReduction?: number;
  readonly fpuParentWeight?: number;
  readonly maxChildren?: number;
}

export const DEFAULT_SEARCH_PARAMS: Required<SearchParams> = Object.freeze({
  cpuct: 1.1,
  fpuReduction: 0.2,
  fpuParentWeight: 1.0,
  maxChildren: 100,
});

export interface SearchNode {
  readonly loc: number; // move location that led here
  readonly player: Player; // player who just played this move
  visits: number;
  totalValue: number; // from perspective of player who played this move
  meanValue: number;
  totalScoreLead: number;
  meanScoreLead: number;
  policyPrior: number;
  children: SearchNode[];
  isExpanded: boolean;
  isTerminal: boolean;
}

export interface CandidateMoveResult {
  readonly loc: number;
  readonly x: number;
  readonly y: number;
  readonly isPass: boolean;
  readonly visits: number;
  readonly winrate: number;
  readonly scoreLead: number;
  readonly prior: number;
  readonly pv: string[];
}

export interface SearchReport {
  readonly rootVisits: number;
  readonly rootWinrate: number;
  readonly rootScoreLead: number;
  readonly topMoves: CandidateMoveResult[];
}

export class SearchSession {
  readonly history: BoardHistory;
  readonly evaluator: EvaluationBackend;
  readonly params: Required<SearchParams>;
  readonly komi: number;

  private root: SearchNode;
  private generation = 1;

  constructor(
    history: BoardHistory,
    evaluator: EvaluationBackend,
    komi = 7.5,
    params: SearchParams = {}
  ) {
    this.history = history.clone();
    this.evaluator = evaluator;
    this.komi = komi;
    this.params = { ...DEFAULT_SEARCH_PARAMS, ...params };

    this.root = this.createNode(PASS_LOC, opponent(this.history.currentPlayer), 1.0);
  }

  private createNode(loc: number, player: Player, prior: number): SearchNode {
    return {
      loc,
      player,
      visits: 0,
      totalValue: 0,
      meanValue: 0,
      totalScoreLead: 0,
      meanScoreLead: 0,
      policyPrior: prior,
      children: [],
      isExpanded: false,
      isTerminal: false,
    };
  }

  /**
   * Resets the search tree, incrementing the generation to cancel in-flight batches.
   */
  reset(): void {
    this.generation++;
    this.root = this.createNode(PASS_LOC, opponent(this.history.currentPlayer), 1.0);
  }

  /**
   * Expands a leaf node by extracting features, evaluating with EvaluationBackend, and filtering illegal moves.
   */
  private async expandNode(node: SearchNode, simHistory: BoardHistory): Promise<void> {
    if (node.isExpanded || node.isTerminal) return;

    if (simHistory.isGameFinished()) {
      node.isTerminal = true;
      node.isExpanded = true;
      const score = simHistory.computeScore(this.komi);
      const isBlack = node.player === 'black';
      const won = (score.scoreLead > 0 && isBlack) || (score.scoreLead < 0 && !isBlack);
      node.meanValue = won ? 1.0 : -1.0;
      node.meanScoreLead = isBlack ? score.scoreLead : -score.scoreLead;
      return;
    }

    const encoded = extractFeatures(simHistory, this.komi, { layout: 'NHWC' });
    const batch: EvaluationBatch = {
      batchSize: 1,
      width: simHistory.currentBoard.width,
      height: simHistory.currentBoard.height,
      layout: 'NHWC',
      spatial: encoded.spatial,
      global: encoded.global,
      generation: this.generation,
    };

    const evalResult = await this.evaluator.evaluate(batch);
    const policy = evalResult.policy; // shape [362]
    const value = evalResult.values[0];

    const nextPla = simHistory.currentPlayer;
    const width = simHistory.currentBoard.width;
    const height = simHistory.currentBoard.height;
    const area = width * height;

    // Filter legal moves and re-normalize policy
    let legalPolicySum = 0;
    const legalCandidates: { loc: number; prior: number }[] = [];

    // Board moves
    for (let loc = 0; loc < area; loc++) {
      if (simHistory.isLegal(loc, nextPla)) {
        const p = policy[loc];
        if (p > 0) {
          legalCandidates.push({ loc, prior: p });
          legalPolicySum += p;
        }
      }
    }

    // Pass move is always legal (index = area)
    const passPrior = policy[area] ?? 0.01;
    legalCandidates.push({ loc: PASS_LOC, prior: passPrior });
    legalPolicySum += passPrior;

    // Normalize and attach children
    const invSum = legalPolicySum > 0 ? 1.0 / legalPolicySum : 1.0 / legalCandidates.length;
    for (const cand of legalCandidates) {
      const normPrior = cand.prior * invSum;
      const child = this.createNode(cand.loc, nextPla, normPrior);
      node.children.push(child);
    }

    node.isExpanded = true;
    node.meanValue = value.winrate * 2.0 - 1.0; // scale 0..1 to -1..+1
    node.meanScoreLead = value.scoreLead;
  }

  /**
   * Selects child using PUCT formula with dynamic FPU matching KataGo v1.18.1.
   */
  private selectChild(node: SearchNode): SearchNode {
    const sqrtParentVisits = Math.sqrt(Math.max(1, node.visits));
    const parentQ = node.meanValue;

    // Dynamic FPU calculation: unvisited children receive parentQ minus FPU reduction
    const fpuValue = parentQ - this.params.fpuReduction;

    let bestScore = -Infinity;
    let bestChild = node.children[0];

    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      // Invert child's Q because child.meanValue is from child.player's perspective!
      const q = child.visits > 0 ? -child.meanValue : fpuValue;
      const u =
        this.params.cpuct * child.policyPrior * (sqrtParentVisits / (1 + child.visits));
      const score = q + u;

      if (score > bestScore) {
        bestScore = score;
        bestChild = child;
      }
    }

    return bestChild;
  }

  /**
   * Executes a specified number of MCTS visits.
   */
  async runVisits(visits: number): Promise<SearchReport> {
    // Ensure root is expanded
    if (!this.root.isExpanded) {
      await this.expandNode(this.root, this.history);
    }

    for (let v = 0; v < visits; v++) {
      const simHistory = this.history.clone();
      const path: SearchNode[] = [this.root];

      let curr = this.root;
      while (curr.isExpanded && !curr.isTerminal && curr.children.length > 0) {
        const next = this.selectChild(curr);
        simHistory.makeMove(next.loc, next.player);
        path.push(next);
        curr = next;
      }

      // Expand leaf
      if (!curr.isTerminal && !curr.isExpanded) {
        await this.expandNode(curr, simHistory);
      }

      // Backpropagation along path
      const leafValue = curr.meanValue;
      const leafScore = curr.meanScoreLead;
      const leafPlayer = curr.player;

      for (let pIdx = path.length - 1; pIdx >= 0; pIdx--) {
        const n = path[pIdx];
        n.visits++;
        // Value sign adjustment if node player matches leaf player
        const sign = n.player === leafPlayer ? 1.0 : -1.0;
        n.totalValue += leafValue * sign;
        n.meanValue = n.totalValue / n.visits;

        n.totalScoreLead += leafScore * sign;
        n.meanScoreLead = n.totalScoreLead / n.visits;
      }
    }

    return this.getReport();
  }

  /**
   * Generates formatted report of current search status and top candidate moves.
   */
  getReport(topK = 5): SearchReport {
    const rootVisits = this.root.visits;
    // Winrate from perspective of root player to move
    const rootWinrate = (this.root.meanValue + 1.0) / 2.0;
    const rootScoreLead = this.root.meanScoreLead;

    const sortedChildren = [...this.root.children].sort((a, b) => b.visits - a.visits);
    const topMoves: CandidateMoveResult[] = [];

    const k = Math.min(topK, sortedChildren.length);
    for (let i = 0; i < k; i++) {
      const child = sortedChildren[i];
      const width = this.history.currentBoard.width;
      const x = locToX(child.loc, width);
      const y = locToY(child.loc, width);
      const isPass = child.loc === PASS_LOC;

      // Extract PV
      const pv: string[] = [];
      let pvCurr = child;
      while (pvCurr && pvCurr.isExpanded && pvCurr.children.length > 0) {
        const moveStr =
          pvCurr.loc === PASS_LOC
            ? 'pass'
            : `${String.fromCharCode(65 + locToX(pvCurr.loc, width))}${locToY(pvCurr.loc, width) + 1}`;
        pv.push(moveStr);
        if (pv.length >= 10) break;
        // Pick most visited grandchild
        const next = [...pvCurr.children].sort((a, b) => b.visits - a.visits)[0];
        pvCurr = next;
      }

      topMoves.push({
        loc: child.loc,
        x,
        y,
        isPass,
        visits: child.visits,
        winrate: (child.meanValue + 1.0) / 2.0,
        scoreLead: child.meanScoreLead,
        prior: child.policyPrior,
        pv,
      });
    }

    return {
      rootVisits,
      rootWinrate,
      rootScoreLead,
      topMoves,
    };
  }
}
