export type Player = 'black' | 'white';

export type BoardState = (Player | null)[][];

export type Move = {
  x: number;
  y: number;
  player: Player;
};

export type GameRules = 'japanese' | 'chinese' | 'korean';

export type KataGoBackendPreference = 'webgpu' | 'wasm' | 'cpu';

export type FloatArray = Float32Array | number[];

export type RegionOfInterest = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

export type CandidateMove = {
  x: number;
  y: number;
  visits: number;
  winRate: number;
  scoreLead: number;
  scoreMean: number;
  scoreStdev?: number;
  prior: number;
  utility?: number;
  pv: string[];
};

export type AnalysisResult = {
  rootVisits: number;
  rootWinRate: number;
  rootScoreLead: number;
  moves: CandidateMove[];
  policy: Float32Array;
  ownership: Float32Array;
  movesOwnership?: Float32Array[];
};

export type AnalysisProgress = {
  rootVisits: number;
  rootWinRate: number;
  rootScoreLead: number;
  moves: CandidateMove[];
  policy: Float32Array;
  ownership: Float32Array;
};
