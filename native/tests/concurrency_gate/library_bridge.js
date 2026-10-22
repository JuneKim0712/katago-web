addToLibrary({
  js_eval_batch_async__deps: ['$PThread'],
  js_eval_batch_async__async: true,
  js_eval_batch_async: function(slot_id) {
    var threadId = typeof _pthread_self === 'function' ? _pthread_self() : 0;
    return new Promise(function(resolve, reject) {
      if (typeof globalThis._pendingResolvers === 'undefined') {
        globalThis._pendingResolvers = {};
        self.addEventListener('message', function(e) {
          if (e.data && e.data.cmd === 'eval_response') {
            e.stopImmediatePropagation();
            var res = globalThis._pendingResolvers[e.data.slot_id];
            if (res) {
              delete globalThis._pendingResolvers[e.data.slot_id];
              res(e.data.result);
            }
          }
        }, true);
      }
      globalThis._pendingResolvers[slot_id] = resolve;
      postMessage({ cmd: 'callHandler', handler: 'onPthreadEvalBatch', args: [slot_id, threadId] });
    });
  }
});
