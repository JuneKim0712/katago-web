import { test, expect } from '@playwright/test';
import { startServer } from './concurrency/server.mjs';

let server: any;
const PORT = 8899;

test.beforeAll(async () => {
  server = await startServer(PORT);
});

test.afterAll(async () => {
  if (server) {
    server.close();
  }
});

test.describe('Pattern C Concurrency Gate: Pthreads + JSPI + Coordinator Worker', () => {
  test('evaluator pthread suspends on async JSPI import while concurrent native thread progresses', async ({ page }) => {
    // Collect browser console messages for diagnostics
    const consoleLogs: string[] = [];
    page.on('console', (msg) => {
      consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
    });

    await page.goto(`http://localhost:${PORT}/test/browser/concurrency/index.html`);

    // Wait for the test to complete or fail
    await page.waitForFunction(() => (window as any).testFinished === true, { timeout: 30000 });

    const testSuccess = await page.evaluate(() => (window as any).testSuccess);
    const events = await page.evaluate(() => (window as any).testEvents);

    console.log('Console logs:', consoleLogs.join('\n'));
    console.log('Received events:', JSON.stringify(events, null, 2));

    expect(testSuccess).toBe(true);

    // Verify event ordering and concurrency:
    const evalStarted = events.find((e: any) => e.type === 'eval_started');
    const evalCompleted = events.find((e: any) => e.type === 'eval_completed');
    const successEvent = events.find((e: any) => e.type === 'success');

    expect(evalStarted).toBeDefined();
    expect(evalCompleted).toBeDefined();
    expect(successEvent).toBeDefined();

    // Verify background thread made progress during JSPI suspension
    expect(successEvent.ticks).toBeGreaterThan(0);
  });

  test('cancellation while async JSPI import is suspended resolves cleanly without hanging', async ({ page }) => {
    // Collect console messages
    const consoleLogs: string[] = [];
    page.on('console', (msg) => {
      consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
    });

    await page.goto(`http://localhost:${PORT}/test/browser/concurrency/index.html`);

    // Wait until worker is ready
    await page.waitForFunction(() => (window as any).testEvents?.some((e: any) => e.type === 'ready'), { timeout: 10000 });

    // When eval_started event appears, immediately dispatch cancel to the worker while JSPI is suspended
    await page.evaluate(() => {
      return new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Timed out waiting for eval_started')), 10000);
        const interval = setInterval(() => {
          const evalStart = (window as any).testEvents?.find((e: any) => e.type === 'eval_started');
          if (evalStart) {
            clearInterval(interval);
            clearTimeout(timeout);
            // Post cancel directly to coordinator worker while evaluator pthread is suspended in JSPI
            (window as any).worker.postMessage({ cmd: 'cancel' });
            resolve();
          }
        }, 5);
      });
    });

    // Wait for cancel to be acknowledged and eval_cancelled to be recorded
    await page.waitForFunction(() => (window as any).testEvents?.some((e: any) => e.type === 'eval_cancelled'), { timeout: 10000 });

    // Wait for the test to complete cleanly (the suspended pthread must wake up with result -999 and terminate without hanging)
    await page.waitForFunction(() => (window as any).testFinished === true, { timeout: 15000 });

    const events = await page.evaluate(() => (window as any).testEvents);
    const cancelAck = events.find((e: any) => e.type === 'cancel_acknowledged');
    const evalCancelled = events.find((e: any) => e.type === 'eval_cancelled');

    expect(cancelAck).toBeDefined();
    expect(evalCancelled).toBeDefined();

    // Verify evaluator pthread did NOT hang and finished execution
    const finishedEvent = events.find((e: any) => e.type === 'failure' || e.type === 'success');
    expect(finishedEvent).toBeDefined();
    expect(finishedEvent.ticks).toBeGreaterThan(0);
  });
});
