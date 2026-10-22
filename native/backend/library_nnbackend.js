addToLibrary({
  js_eval_batch_async__deps: ['$PThread'],
  js_eval_batch_async__async: true,
  js_eval_batch_async: function(slot_id, generation, seq_low, seq_high) {
    var threadId = typeof _pthread_self === 'function' ? _pthread_self() : 0;
    if (threadId === 0) {
      if (typeof Module !== 'undefined' && typeof Module.onPthreadEvalBatch === 'function') {
        return Module.onPthreadEvalBatch(slot_id, 0, generation, seq_low, seq_high);
      }
      return Promise.resolve(0);
    }
    return new Promise(function(resolve, reject) {
      if (typeof globalThis._pendingResolvers === 'undefined') {
        globalThis._pendingResolvers = {};
        self.addEventListener('message', function(e) {
          if (e.data && e.data.cmd === 'eval_response') {
            e.stopImmediatePropagation();
            var key = e.data.slot_id;
            var entry = globalThis._pendingResolvers[key];
            if (entry) {
              if (typeof e.data.generation === 'number' && e.data.generation !== entry.generation) {
                // Drop stale generation response
                return;
              }
              delete globalThis._pendingResolvers[key];
              entry.resolve(e.data.result !== undefined ? e.data.result : 0);
            }
          }
        }, true);
      }
      globalThis._pendingResolvers[slot_id] = {
        resolve: resolve,
        generation: generation,
        seq_low: seq_low,
        seq_high: seq_high
      };
      postMessage({
        cmd: 'callHandler',
        handler: 'onPthreadEvalBatch',
        args: [slot_id, threadId, generation, seq_low, seq_high]
      });
    });
  }
});
