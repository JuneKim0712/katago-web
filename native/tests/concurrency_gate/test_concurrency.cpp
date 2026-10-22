#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <emscripten.h>
#include <emscripten/threading.h>

#ifdef __cplusplus
extern "C" {
#endif

// JSPI imported async function
extern int js_eval_batch_async(int slot_id);

// Atomic progress flags
static volatile int g_thread2_ticks = 0;
static volatile int g_thread1_result = 0;
static volatile int g_test_status = 0; // 0 = running, 1 = success, -1 = failure

void* evaluator_thread_func(void* arg) {
    printf("[Evaluator Thread] Starting. Calling js_eval_batch_async(42)...\n");
    int res = js_eval_batch_async(42);
    printf("[Evaluator Thread] Resumed from js_eval_batch_async, returned: %d\n", res);
    g_thread1_result = res;

    // Verify that thread 2 made progress while we were suspended
    int t2_ticks = g_thread2_ticks;
    printf("[Evaluator Thread] Concurrent thread2 ticks observed: %d\n", t2_ticks);
    if (res == 84 && t2_ticks > 0) {
        g_test_status = 1;
        printf("[Evaluator Thread] SUCCESS: JSPI suspension and concurrent progress verified!\n");
    } else {
        g_test_status = -1;
        printf("[Evaluator Thread] FAILURE: res=%d, t2_ticks=%d\n", res, t2_ticks);
    }
    return NULL;
}

void* background_thread_func(void* arg) {
    printf("[Background Thread] Starting progress ticks...\n");
    for (int i = 0; i < 20; ++i) {
        usleep(10000); // 10ms
        g_thread2_ticks++;
        if (g_test_status != 0) break;
    }
    printf("[Background Thread] Finished %d ticks.\n", g_thread2_ticks);
    return NULL;
}

EMSCRIPTEN_KEEPALIVE
int start_concurrency_test() {
    printf("[Main Wasm Thread] Starting concurrency test...\n");
    g_thread2_ticks = 0;
    g_thread1_result = 0;
    g_test_status = 0;

    pthread_t t1, t2;
    pthread_create(&t1, NULL, evaluator_thread_func, NULL);
    pthread_create(&t2, NULL, background_thread_func, NULL);

    pthread_detach(t1);
    pthread_detach(t2);
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int check_test_status() {
    return g_test_status;
}

EMSCRIPTEN_KEEPALIVE
int get_thread2_ticks() {
    return g_thread2_ticks;
}

int main() {
    printf("[Wasm main] Module loaded.\n");
    return 0;
}

#ifdef __cplusplus
}
#endif
