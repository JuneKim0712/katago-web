import { describe, expect, it } from 'vitest';
import { fillInputsV7Fast, type RecentMove, getSymPosMap, NUM_SYMMETRIES } from '../../src/engine/featuresV7Fast.js';
import { assertFloat32BitsEqual } from '../../src/utils/parityComparators.js';
import { BLACK, WHITE, BOARD_SIZE } from '../../src/engine/fastBoard.js';

describe('featuresExact parity', () => {
  it('generates bitwise identical spatial and global features for reference position', () => {
    const stones = new Uint8Array(361);
    stones[3 * 19 + 3] = BLACK;
    stones[3 * 19 + 15] = WHITE;
    stones[15 * 19 + 3] = BLACK;
    stones[15 * 19 + 15] = WHITE;

    const recentMoves: RecentMove[] = [
      { move: 3 * 19 + 3, player: 'black' },
      { move: 3 * 19 + 15, player: 'white' },
      { move: 15 * 19 + 3, player: 'black' },
      { move: 15 * 19 + 15, player: 'white' },
    ];

    const spatial1 = new Float32Array(361 * 22);
    const global1 = new Float32Array(19);
    fillInputsV7Fast({
      stones,
      koPoint: -1,
      currentPlayer: 'black',
      recentMoves,
      komi: 7.5,
      rules: 'chinese',
      outSpatial: spatial1,
      outGlobal: global1,
    });

    const spatial2 = new Float32Array(361 * 22);
    const global2 = new Float32Array(19);
    fillInputsV7Fast({
      stones,
      koPoint: -1,
      currentPlayer: 'black',
      recentMoves,
      komi: 7.5,
      rules: 'chinese',
      outSpatial: spatial2,
      outGlobal: global2,
    });

    assertFloat32BitsEqual(spatial1, spatial2, 'spatial deterministic baseline');
    assertFloat32BitsEqual(global1, global2, 'global deterministic baseline');
  });

  it('handles ko and pass moves deterministically', () => {
    const stones = new Uint8Array(361);
    const recentMoves: RecentMove[] = [
      { move: 361, player: 'black' }, // pass
      { move: 100, player: 'white' },
    ];

    const spatial = new Float32Array(361 * 22);
    const global = new Float32Array(19);
    fillInputsV7Fast({
      stones,
      koPoint: 101,
      currentPlayer: 'black',
      recentMoves,
      komi: 6.5,
      rules: 'japanese',
      conservativePassAndIsRoot: true,
      outSpatial: spatial,
      outGlobal: global,
    });

    expect(spatial[101 * 22 + 6]).toBe(1.0); // ko plane
    expect(global[5]).toBeCloseTo((6.5 / 20.0) * (-1.0), 5);
  });

  it('preserves bitwise exactness across all 8 symmetries vs legacy encode-then-permute', () => {
    const stones = new Uint8Array(361);
    stones[2 * 19 + 4] = BLACK;
    stones[4 * 19 + 2] = WHITE;
    stones[10 * 19 + 10] = BLACK;
    stones[16 * 19 + 14] = WHITE;

    const recentMoves: RecentMove[] = [
      { move: 361, player: 'white' }, // pass
      { move: 2 * 19 + 4, player: 'black' },
      { move: 4 * 19 + 2, player: 'white' },
      { move: 10 * 19 + 10, player: 'black' },
    ];

    const fakeLadders = new Uint8Array(361);
    fakeLadders[2 * 19 + 4] = 1;
    const fakeWorking = new Uint8Array(361);
    fakeWorking[2 * 19 + 5] = 1;

    const rulesList = ['japanese', 'korean', 'chinese'] as const;
    const players = ['black', 'white'] as const;
    const symPosMap = getSymPosMap();

    for (const rules of rulesList) {
      for (const pla of players) {
        for (let sym = 0; sym < NUM_SYMMETRIES; sym++) {
          const symOff = sym * 361;
          const symMap = sym === 0 ? null : symPosMap.subarray(symOff, symOff + 361);

          // 1. Legacy encode-then-permute method
          const unpermutedSpatial = new Float32Array(361 * 22);
          const legacyGlobal = new Float32Array(19);
          fillInputsV7Fast({
            stones,
            koPoint: 45,
            currentPlayer: pla,
            recentMoves,
            komi: 7.5,
            rules,
            conservativePassAndIsRoot: true,
            ladderedStones: fakeLadders,
            ladderWorkingMoves: fakeWorking,
            outSpatial: unpermutedSpatial,
            outGlobal: legacyGlobal,
          });

          const legacyPermutedSpatial = new Float32Array(361 * 22);
          if (sym === 0) {
            legacyPermutedSpatial.set(unpermutedSpatial);
          } else {
            for (let pos = 0; pos < 361; pos++) {
              const dstPos = symPosMap[symOff + pos]!;
              const srcBase = pos * 22;
              const dstBase = dstPos * 22;
              for (let c = 0; c < 22; c++) {
                legacyPermutedSpatial[dstBase + c] = unpermutedSpatial[srcBase + c]!;
              }
            }
          }

          // 2. Direct symmetry-aware encoding
          const directSpatial = new Float32Array(361 * 22);
          const directGlobal = new Float32Array(19);
          fillInputsV7Fast({
            stones,
            koPoint: 45,
            currentPlayer: pla,
            recentMoves,
            komi: 7.5,
            rules,
            conservativePassAndIsRoot: true,
            ladderedStones: fakeLadders,
            ladderWorkingMoves: fakeWorking,
            symmetryMap: symMap,
            outSpatial: directSpatial,
            outGlobal: directGlobal,
          });

          assertFloat32BitsEqual(
            directSpatial,
            legacyPermutedSpatial,
            `spatial symmetry ${sym} rules ${rules} player ${pla}`
          );
          assertFloat32BitsEqual(
            directGlobal,
            legacyGlobal,
            `global symmetry ${sym} rules ${rules} player ${pla}`
          );
        }
      }
    }
  });
});
