import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import {
  BLACK,
  WHITE,
  EMPTY,
  PASS_MOVE,
  BOARD_SIZE,
  BOARD_AREA,
  playMove,
  undoMove,
  computeLibertyMap,
  computeLadderFeaturesV7KataGoInto,
  type SimPosition,
  type StoneColor,
} from '../src/engine/fastBoard.js';

describe('Board Property Invariants (fast-check)', () => {
  it('preserves board and ko invariant under legal play and undo', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            x: fc.integer({ min: 0, max: BOARD_SIZE - 1 }),
            y: fc.integer({ min: 0, max: BOARD_SIZE - 1 }),
          }),
          { minLength: 1, maxLength: 20 }
        ),
        (moveList) => {
          const pos: SimPosition = { stones: new Uint8Array(BOARD_AREA), koPoint: -1 };
          const captureStack: number[] = [];
          const playedMoves: number[] = [];
          const playedColors: StoneColor[] = [];
          const snapshots = [];

          let color: StoneColor = BLACK;
          for (const m of moveList) {
            const loc = m.y * BOARD_SIZE + m.x;
            try {
              const snap = playMove(pos, loc, color, captureStack);
              playedMoves.push(loc);
              playedColors.push(color);
              snapshots.push(snap);
              color = color === BLACK ? WHITE : BLACK;
            } catch {
              // Illegal moves in random sequence are expected and skipped
            }
          }

          // Now undo all played moves in reverse order
          for (let i = playedMoves.length - 1; i >= 0; i--) {
            undoMove(pos, playedMoves[i]!, playedColors[i]!, snapshots[i]!, captureStack);
          }

          // Board should be completely empty and ko reset
          expect(pos.koPoint).toBe(-1);
          for (let i = 0; i < BOARD_AREA; i++) {
            expect(pos.stones[i]).toBe(EMPTY);
          }
          expect(captureStack.length).toBe(0);
        }
      ),
      { numRuns: 100 }
    );
  });
});
