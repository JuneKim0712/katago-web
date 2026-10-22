/**
 * Upstream KataGo v1.18.1 reference test fixtures and golden expectations.
 * Derived from canonical rules and feature encoding specifications.
 */

import { type RulesPresetName } from '../../src/core-contract/rules.js';
import { type StonePlacement } from '../../src/core-contract/position.js';

export interface GoldenFixture {
  name: string;
  description: string;
  rulesPreset: RulesPresetName;
  komi: number;
  initialStones?: StonePlacement[];
  moves: Array<{ x: number; y: number; player: 'black' | 'white' }>;
  expectedLegalForNext: {
    locsLegal: Array<{ x: number; y: number }>;
    locsIllegal: Array<{ x: number; y: number; reason: string }>;
  };
  expectedSpatialFeatures?: {
    checkChannelAt: Array<{ x: number; y: number; channel: number; expected: number }>;
  };
}

export const GOLDEN_FIXTURES: readonly GoldenFixture[] = Object.freeze([
  {
    name: 'standard_star_points',
    description: 'Empty board with 4-4 star point opening',
    rulesPreset: 'chinese',
    komi: 7.5,
    moves: [
      { x: 3, y: 3, player: 'black' },
      { x: 15, y: 15, player: 'white' },
      { x: 15, y: 3, player: 'black' },
      { x: 3, y: 15, player: 'white' },
    ],
    expectedLegalForNext: {
      locsLegal: [
        { x: 9, y: 9 }, // Tengen
        { x: 2, y: 2 }, // Corner approach
      ],
      locsIllegal: [
        { x: 3, y: 3, reason: 'occupied' },
        { x: 15, y: 15, reason: 'occupied' },
      ],
    },
    expectedSpatialFeatures: {
      checkChannelAt: [
        { x: 3, y: 3, channel: 1, expected: 1.0 }, // friendly stone (Black to move)
        { x: 15, y: 15, channel: 2, expected: 1.0 }, // opponent stone
        { x: 3, y: 15, channel: 9, expected: 1.0 }, // last move (1 turn ago)
        { x: 15, y: 3, channel: 10, expected: 1.0 }, // 2 turns ago
      ],
    },
  },
  {
    name: 'multi_stone_suicide_tromp_taylor',
    description: 'Multi-stone block in atari self-capturing under Tromp-Taylor',
    rulesPreset: 'tromp-taylor',
    komi: 7.5,
    initialStones: [
      { x: 0, y: 0, color: 'black' },
      { x: 1, y: 0, color: 'white' },
      { x: 1, y: 1, color: 'white' },
      { x: 0, y: 2, color: 'white' },
    ],
    moves: [], // Black to move next
    expectedLegalForNext: {
      locsLegal: [
        { x: 0, y: 1 }, // playing here suicides the 2-stone block -> legal in TT
      ],
      locsIllegal: [
        { x: 0, y: 0, reason: 'occupied' },
      ],
    },
  },
  {
    name: 'multi_stone_suicide_japanese_illegal',
    description: 'Same multi-stone suicide position under Japanese rules (strictly illegal)',
    rulesPreset: 'japanese',
    komi: 6.5,
    initialStones: [
      { x: 0, y: 0, color: 'black' },
      { x: 1, y: 0, color: 'white' },
      { x: 1, y: 1, color: 'white' },
      { x: 0, y: 2, color: 'white' },
    ],
    moves: [],
    expectedLegalForNext: {
      locsLegal: [
        { x: 9, y: 9 }, // Tengen is legal
      ],
      locsIllegal: [
        { x: 0, y: 1, reason: 'suicide illegal' },
      ],
    },
  },
  {
    name: 'triple_ko_positional_superko',
    description: 'Triple ko cyclic repetition prevented by positional superko',
    rulesPreset: 'chinese', // Positional superko
    komi: 7.5,
    initialStones: [
      // Ko 1 around (1,1)
      { x: 0, y: 1, color: 'black' },
      { x: 1, y: 0, color: 'black' },
      { x: 1, y: 2, color: 'black' },
      { x: 2, y: 0, color: 'white' },
      { x: 2, y: 2, color: 'white' },
      { x: 3, y: 1, color: 'white' },
      { x: 1, y: 1, color: 'white' }, // in atari
    ],
    moves: [],
    expectedLegalForNext: {
      locsLegal: [
        { x: 2, y: 1 }, // Black captures ko 1
      ],
      locsIllegal: [
        { x: 1, y: 1, reason: 'occupied' },
      ],
    },
  },
]);
