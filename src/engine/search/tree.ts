import type { StoneColor } from '../board/context.js';

export type Edge = {
  move: number; // 0..360 or PASS_MOVE
  prior: number;
  child: Node | null;
  pvCache?: { visits: number; depth: number; pv: string[] };
};

export class Node {
  readonly playerToMove: StoneColor;
  visits = 0;
  valueSum = 0; // [-1,1] where +1 is black win
  scoreLeadSum = 0; // black lead
  scoreMeanSum = 0; // black score mean
  scoreMeanSqSum = 0; // sum of (stdev^2 + mean^2) for mixture stdev
  utilitySum = 0; // from black perspective
  utilitySqSum = 0; // from black perspective
  nnUtility: number | null = null; // direct NN eval utility, from black perspective
  ownership: Float32Array | null = null; // len 361, +1 black owns, -1 white owns
  inFlight = 0;
  pendingEval = false;
  edges: Edge[] | null = null;

  constructor(playerToMove: StoneColor) {
    this.playerToMove = playerToMove;
  }
}
