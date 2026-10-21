import { describe, expect, it } from 'vitest';
import {
  BoardContext,
  BoardGeometry,
  getBoardGeometry,
  STAMP_ROLLOVER_THRESHOLD,
  BLACK,
  WHITE,
  EMPTY,
} from '../src/engine/board/context.js';
import { playMove, undoMove, type SimPosition } from '../src/engine/board/moves.js';
import { computeLibertyMap, collectGroupAndLiberties } from '../src/engine/board/liberties.js';

describe('BoardContext and Geometry', () => {
  it('preserves neighbor ordering: left, right, up, down', () => {
    const geom19 = getBoardGeometry(19);
    // Center point (9, 9) -> pos = 9 * 19 + 9 = 180
    const centerPos = 9 * 19 + 9;
    const nStart = geom19.neighborStarts[centerPos]!;
    const nCount = geom19.neighborCounts[centerPos]!;
    expect(nCount).toBe(4);
    expect(geom19.neighbors[nStart + 0]).toBe(centerPos - 1); // left
    expect(geom19.neighbors[nStart + 1]).toBe(centerPos + 1); // right
    expect(geom19.neighbors[nStart + 2]).toBe(centerPos - 19); // up
    expect(geom19.neighbors[nStart + 3]).toBe(centerPos + 19); // down

    // Corner point (0, 0)
    const cornerStart = geom19.neighborStarts[0]!;
    const cornerCount = geom19.neighborCounts[0]!;
    expect(cornerCount).toBe(2);
    expect(geom19.neighbors[cornerStart + 0]).toBe(1); // right
    expect(geom19.neighbors[cornerStart + 1]).toBe(19); // down
  });

  it('safely rolls over stamps near 0x7ffff000 overflow threshold without integer collision', () => {
    const ctx = new BoardContext(19);
    ctx.scratch.visited[50] = 0x7ffff000;
    ctx.scratch.libVisited[50] = 0x7ffff000;
    ctx.scratch.bfsStamp = STAMP_ROLLOVER_THRESHOLD;

    const nextStamp = ctx.nextBfsStamp();
    expect(nextStamp).toBe(1);
    expect(ctx.scratch.bfsStamp).toBe(1);
    expect(ctx.scratch.visited[50]).toBe(0);
    expect(ctx.scratch.libVisited[50]).toBe(0);

    // Processed stamp rollover
    ctx.scratch.processedGroup[20] = 0x7ffff000;
    ctx.scratch.processedStamp = STAMP_ROLLOVER_THRESHOLD;
    const nextProcessed = ctx.nextProcessedStamp();
    expect(nextProcessed).toBe(1);
    expect(ctx.scratch.processedGroup[20]).toBe(0);

    // Group seen stamp rollover
    ctx.scratch.groupSeen[30] = 0x7ffff000;
    ctx.scratch.groupSeenStamp = STAMP_ROLLOVER_THRESHOLD;
    const nextGroupSeen = ctx.nextGroupSeenStamp();
    expect(nextGroupSeen).toBe(1);
    expect(ctx.scratch.groupSeen[30]).toBe(0);
  });

  it('maintains complete isolation between independent board contexts', () => {
    const ctxA = new BoardContext(19);
    const ctxB = new BoardContext(19);

    const posA: SimPosition = { stones: new Uint8Array(361), koPoint: -1 };
    const posB: SimPosition = { stones: new Uint8Array(361), koPoint: -1 };
    const capturesA: number[] = [];
    const capturesB: number[] = [];

    // Play move 200 (center of board) in context A
    playMove(posA, 200, BLACK, capturesA, ctxA);
    expect(posA.stones[200]).toBe(BLACK);
    expect(posB.stones[200]).toBe(EMPTY);

    // Compute liberties in context B
    const libsB = computeLibertyMap(posB.stones, ctxB);
    expect(libsB[200]).toBe(0);

    // Compute liberties in context A
    const libsA = computeLibertyMap(posA.stones, ctxA);
    expect(libsA[200]).toBe(4);

    // Verify scratch stamps do not interfere
    ctxA.scratch.bfsStamp = 100;
    ctxB.scratch.bfsStamp = 500;
    expect(ctxA.nextBfsStamp()).toBe(101);
    expect(ctxB.nextBfsStamp()).toBe(501);
  });
});
