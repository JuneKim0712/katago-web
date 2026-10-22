import { test, expect } from '@playwright/test';
import { startServer } from './concurrency/server.mjs';

let server: any;
const PORT = 8898;

test.beforeAll(async () => {
  server = await startServer(PORT);
});

test.afterAll(async () => {
  if (server) {
    server.close();
  }
});

test.describe('Pattern C KataGo Hybrid Engine End-to-End', () => {
  test('executes end-to-end initialization, ORT WebGPU model inference, and MCTS analysis', async ({ page }) => {
    test.setTimeout(90000);

    const consoleLogs: string[] = [];
    page.on('console', (msg) => {
      const line = `[BROWSER ${msg.type()}] ${msg.text()}`;
      console.log(line);
      consoleLogs.push(line);
    });
    page.on('pageerror', (err) => {
      console.error(`[PAGE ERROR]`, err);
      consoleLogs.push(`[PAGE ERROR] ${err}`);
    });

    await page.goto(`http://localhost:${PORT}/test/browser/engine_e2e.html`);

    await page.waitForFunction(() => (window as any).testFinished === true, { timeout: 80000 });

    const testSuccess = await page.evaluate(() => (window as any).testSuccess);
    const testError = await page.evaluate(() => (window as any).testError);
    const testLogs = await page.evaluate(() => (window as any).testLogs);
    const testResult = await page.evaluate(() => (window as any).testResult);
    const testEvents = await page.evaluate(() => (window as any).testEvents);

    console.log('Browser Logs:\n', testLogs.join('\n'));
    if (consoleLogs.length > 0) {
      console.log('Console Logs:\n', consoleLogs.join('\n'));
    }

    if (!testSuccess) {
      console.error('Test failed with error:', testError);
    }
    expect(testSuccess).toBe(true);

    // Verify lifecycle events
    expect(testEvents.some((e: any) => e.type === 'load-progress' && e.stage === 'fetch')).toBe(true);
    expect(testEvents.some((e: any) => e.type === 'load-progress' && e.stage === 'compile')).toBe(true);
    expect(testEvents.some((e: any) => e.type === 'ready')).toBe(true);

    // Verify analysis result
    expect(testResult).toBeDefined();
    expect(testResult.kind).toBe('completed');
    expect(testResult.snapshot.rootVisits).toBeGreaterThanOrEqual(10);
    expect(testResult.snapshot.candidates.length).toBeGreaterThan(0);

    const evalRes = testResult.snapshot.evaluation;
    expect(evalRes.blackWinProbability).toBeGreaterThanOrEqual(0);
    expect(evalRes.whiteWinProbability).toBeGreaterThanOrEqual(0);
    expect(evalRes.noResultProbability).toBeGreaterThanOrEqual(0);

    const probSum = evalRes.blackWinProbability + evalRes.whiteWinProbability + evalRes.noResultProbability;
    expect(Math.abs(probSum - 1.0)).toBeLessThan(1e-3);
  });
});
