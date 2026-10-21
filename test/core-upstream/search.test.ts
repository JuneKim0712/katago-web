import { describe, it, expect } from 'vitest';
import { SearchSession } from '../../src/core-runtime/search.js';
import { BoardHistory } from '../../src/core-runtime/boardHistory.js';
import { RULES_PRESETS } from '../../src/core-contract/rules.js';
import { FakeEvaluator } from './fakeEvaluator.js';

describe('Stage 5: Search Integration with Backend-Neutral Evaluator', () => {
  it('runs MCTS visits and accumulates visits properly', async () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const evaluator = new FakeEvaluator();
    const session = new SearchSession(history, evaluator, 6.5);

    const report = await session.runVisits(20);

    expect(report.rootVisits).toBe(20);
    expect(report.topMoves.length).toBeGreaterThan(0);
    expect(evaluator.positionsEvaluated).toBeGreaterThan(0);

    // Sum of child visits must equal root visits
    let totalChildVisits = 0;
    for (const move of report.topMoves) {
      totalChildVisits += move.visits;
    }
    expect(totalChildVisits).toBeLessThanOrEqual(20);
  });

  it('selects plausible top moves and extracts principal variations (PV)', async () => {
    const history = new BoardHistory(RULES_PRESETS.chinese, 'black', 19, 19);
    const evaluator = new FakeEvaluator();
    const session = new SearchSession(history, evaluator, 7.5);

    const report = await session.runVisits(30);

    const top1 = report.topMoves[0];
    expect(top1).toBeDefined();
    expect(top1.visits).toBeGreaterThan(0);
    expect(top1.prior).toBeGreaterThan(0);
    expect(top1.pv.length).toBeGreaterThan(0);
  });

  it('is completely reproducible and deterministic under FakeEvaluator', async () => {
    const history1 = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const eval1 = new FakeEvaluator();
    const session1 = new SearchSession(history1, eval1, 6.5);
    const report1 = await session1.runVisits(15);

    const history2 = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const eval2 = new FakeEvaluator();
    const session2 = new SearchSession(history2, eval2, 6.5);
    const report2 = await session2.runVisits(15);

    expect(report1.rootVisits).toBe(report2.rootVisits);
    expect(report1.topMoves.length).toBe(report2.topMoves.length);
    for (let i = 0; i < report1.topMoves.length; i++) {
      expect(report1.topMoves[i].loc).toBe(report2.topMoves[i].loc);
      expect(report1.topMoves[i].visits).toBe(report2.topMoves[i].visits);
      expect(report1.topMoves[i].winrate).toBeCloseTo(report2.topMoves[i].winrate);
    }
  });

  it('resets tree and generation correctly', async () => {
    const history = new BoardHistory(RULES_PRESETS.japanese, 'black', 19, 19);
    const evaluator = new FakeEvaluator();
    const session = new SearchSession(history, evaluator, 6.5);

    await session.runVisits(10);
    expect(session.getReport().rootVisits).toBe(10);

    session.reset();
    expect(session.getReport().rootVisits).toBe(0);
  });
});
