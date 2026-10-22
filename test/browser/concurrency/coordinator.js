import createModule from '../../../native/tests/concurrency_gate/concurrency_test.js';

let Module = null;
let evalCallCount = 0;
let cancelRequested = false;

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.cmd === 'init') {
    try {
      Module = await createModule({
        onPthreadEvalBatch: async (slotId, threadId) => {
          evalCallCount++;
          self.postMessage({ type: 'eval_started', slotId, threadId, count: evalCallCount });

          // Simulate async inference latency (e.g. WebGPU dispatch)
          await new Promise((resolve) => setTimeout(resolve, 80));

          if (cancelRequested) {
            self.postMessage({ type: 'eval_cancelled', slotId, threadId });
            const worker = Module.PThread?.pthreads?.[threadId];
            if (worker) {
              worker.postMessage({ target: 'setimmediate', cmd: 'eval_response', slot_id: slotId, result: -999 });
            }
            return;
          }

          const result = slotId * 2;
          self.postMessage({ type: 'eval_completed', slotId, threadId, result });

          const worker = Module.PThread?.pthreads?.[threadId];
          if (worker) {
            worker.postMessage({ target: 'setimmediate', cmd: 'eval_response', slot_id: slotId, result });
          } else {
            self.postMessage({ type: 'error', error: `Worker for threadId ${threadId} not found` });
          }
        },
        print: (text) => self.postMessage({ type: 'stdout', text }),
        printErr: (text) => self.postMessage({ type: 'stderr', text }),
      });

      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', error: String(err) });
    }
  } else if (msg.cmd === 'start') {
    cancelRequested = false;
    try {
      Module._start_concurrency_test();

      // Poll test status until completion or timeout
      const startTime = Date.now();
      const interval = setInterval(() => {
        const status = Module._check_test_status();
        const ticks = Module._get_thread2_ticks();
        self.postMessage({ type: 'progress', status, ticks, elapsed: Date.now() - startTime });

        if (status === 1) {
          clearInterval(interval);
          self.postMessage({ type: 'success', ticks, elapsed: Date.now() - startTime });
        } else if (status === -1 || Date.now() - startTime > 10000) {
          clearInterval(interval);
          self.postMessage({
            type: 'failure',
            status,
            ticks,
            error: status === -1 ? 'Assertion failed in native thread' : 'Test timed out'
          });
        }
      }, 20);
    } catch (err) {
      self.postMessage({ type: 'error', error: String(err) });
    }
  } else if (msg.cmd === 'cancel') {
    cancelRequested = true;
    self.postMessage({ type: 'cancel_acknowledged' });
  } else if (msg.cmd === 'terminate') {
    self.close();
  }
};
