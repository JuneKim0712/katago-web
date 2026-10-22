import { test, expect } from '@playwright/test';
import { startServer } from './concurrency/server.mjs';

let server: any;
const PORT = 8897;

test.beforeAll(async () => {
  server = await startServer(PORT);
});

test.afterAll(async () => {
  if (server) {
    server.close();
  }
});

test.describe('Pattern C Full Lifecycle & Performance Qualification (Stages R1, R2, R5, R7, R8)', () => {
  test('qualifies position moves, live cancellation, bot reuse, and multi-run memory stability', async ({ page }) => {
    test.setTimeout(90000);

    const consoleLogs: string[] = [];
    page.on('console', (msg) => {
      const line = `[BROWSER ${msg.type()}] ${msg.text()}`;
      consoleLogs.push(line);
      console.log(line);
    });

    await page.goto(`http://localhost:${PORT}/test/browser/engine_qualification.html`);

    // Wait for all tests to complete
    await page.waitForFunction(() => (window as any).qualResults?.finished === true, { timeout: 80000 });

    const results = await page.evaluate(() => (window as any).qualResults);

    if (!results.success) {
      console.error('Qualification failed with errors:', results.errors);
    }
    expect(results.success).toBe(true);
    expect(results.errors.length).toBe(0);

    // 1. Verify Test 1 (Opening with real moves & probability math)
    const res1 = results.test1_opening;
    expect(res1).toBeDefined();
    expect(res1.snapshot.rootVisits).toBeGreaterThanOrEqual(15);
    expect(res1.snapshot.nextPlayer).toBe('black');
    expect(res1.snapshot.candidates.length).toBeGreaterThan(0);

    const eval1 = res1.snapshot.evaluation;
    expect(eval1.blackWinProbability).toBeGreaterThanOrEqual(0);
    expect(eval1.whiteWinProbability).toBeGreaterThanOrEqual(0);
    expect(eval1.noResultProbability).toBeGreaterThanOrEqual(0);
    const sum1 = eval1.blackWinProbability + eval1.whiteWinProbability + eval1.noResultProbability;
    expect(Math.abs(sum1 - 1.0)).toBeLessThan(1e-3);

    // 2. Verify Test 2 (Live cancellation)
    const cancelInfo = results.test2_cancel;
    expect(cancelInfo).toBeDefined();
    expect(cancelInfo.latencyMs).toBeLessThan(1000); // Cancellation responded within 1s

    // 3. Verify Test 3 (Bot reuse after cancel)
    const res3 = results.test3_reuse;
    expect(res3).toBeDefined();
    expect(res3.snapshot.rootVisits).toBeGreaterThanOrEqual(10);

    // 4. Verify Test 4 (Multi-run stability)
    expect(results.test4_multiRun?.runsCompleted).toBe(3);
  });
});
