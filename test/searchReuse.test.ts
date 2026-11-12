import { describe, it, expect } from 'vitest';
import { SearchSession } from '../src/core-runtime/search.js';
import { BoardHistory } from '../src/core-runtime/boardHistory.js';
import { RULES_PRESETS } from '../src/core-contract/rules.js';
import { FakeEvaluator } from './core-upstream/fakeEvaluator.js';

describe('Search Tree Reuse & Visit Carryover', () => {
  it('preserves subtree visits when advancing move via makeMove', async () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const evaluator = new FakeEvaluator();
    const session = new SearchSession(history, evaluator, 6.5);

    // Initial search of 40 visits
    const initialReport = await session.runVisits(40);
    expect(initialReport.rootVisits).toBe(40);
    expect(initialReport.topMoves.length).toBeGreaterThan(0);

    const topMove = initialReport.topMoves[0]!;
    const topMoveVisits = topMove.visits;
    expect(topMoveVisits).toBeGreaterThan(0);

    // Advance to the top candidate move
    const reusedVisits = session.makeMove(topMove.loc);
    expect(reusedVisits).toBe(topMoveVisits);
    expect(session.getRootVisits()).toBe(topMoveVisits);

    // Run until visits = 60. Should execute only (60 - reusedVisits) additional visits!
    const evalsBefore = evaluator.positionsEvaluated;
    const secondReport = await session.runUntilVisits(60);

    expect(secondReport.rootVisits).toBe(60);
    const additionalEvals = evaluator.positionsEvaluated - evalsBefore;
    expect(additionalEvals).toBe(60 - topMoveVisits);
  });

  it('runs 0 additional visits when reusedVisits already exceeds requested maxVisits', async () => {
    const history = new BoardHistory(RULES_PRESETS.chinese, 'black', 19, 19);
    const evaluator = new FakeEvaluator();
    const session = new SearchSession(history, evaluator, 7.5);

    await session.runVisits(50);
    const topMove = session.getReport().topMoves[0]!;
    const topMoveVisits = topMove.visits;

    const reusedVisits = session.makeMove(topMove.loc);
    expect(reusedVisits).toBe(topMoveVisits);

    // Request target visits less than already present
    const evalsBefore = evaluator.positionsEvaluated;
    const report = await session.runUntilVisits(Math.max(1, topMoveVisits - 5));

    // Zero new evaluations should occur
    expect(evaluator.positionsEvaluated - evalsBefore).toBe(0);
    expect(report.rootVisits).toBe(topMoveVisits);
  });

  it('falls back to fresh root with 0 reusedVisits when move was unvisited', async () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const evaluator = new FakeEvaluator();
    const session = new SearchSession(history, evaluator, 6.5);

    await session.runVisits(10);
    // Find an unvisited location
    const topLocs = new Set(session.getReport().topMoves.map((m) => m.loc));
    let unvisitedLoc = 0;
    while (topLocs.has(unvisitedLoc)) unvisitedLoc++;

    const reused = session.makeMove(unvisitedLoc);
    expect(reused).toBe(0);
    expect(session.getRootVisits()).toBe(0);

    await session.runUntilVisits(25);
    expect(session.getRootVisits()).toBe(25);
  });
});
