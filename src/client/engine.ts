import {
  API_VERSION,
  AnalysisResult,
  AnalyzeRequest,
  Capabilities,
  EngineEvent,
  EngineInfo,
  EngineOptions,
  KataGoEngine,
  KataGoEngineError,
  ModelSource,
  RuntimePreference,
} from '../api/types.js';
import { failRequest, validateAnalyzeRequest, validateIdentifier } from '../api/validation.js';
import { WorkerRpcRequest, WorkerRpcResponse } from '../worker/coordinator.worker.js';

class KataGoEngineImpl implements KataGoEngine {
  readonly apiVersion = API_VERSION;
  private worker: Worker | null = null;
  private subscribers = new Set<(event: EngineEvent) => void>();
  private pendingRpc = new Map<
    string,
    {
      resolve: (val: any) => void;
      reject: (err: any) => void;
    }
  >();
  private activeRequestId: string | null = null;
  private activeRequestSignalCleanup: (() => void) | null = null;
  private activeRequestAbortHandler: (() => void) | null = null;
  private isDisposed = false;
  private rpcSeq = 0;
  private cachedCapabilities: Capabilities | null = null;
  private activeIds = new Set<string>();

  constructor(private readonly options: EngineOptions = {}) {}

  private getWorker(): Worker {
    if (this.isDisposed) {
      failRequest('DISPOSED', 'Engine has been disposed');
    }
    if (!this.worker) {
      if (this.options.workerFactory) {
        this.worker = this.options.workerFactory();
      } else {
        // Default Web Worker instantiation
        this.worker = new Worker(new URL('../worker/coordinator.worker.js', import.meta.url), /* @vite-ignore */ {
          type: 'module',
        });
      }

      this.worker.onmessage = (e: MessageEvent<WorkerRpcResponse>) => {
        const msg = e.data;
        if (!msg) return;

        if (msg.type === 'event' && msg.event) {
          for (const sub of this.subscribers) {
            try {
              sub(msg.event);
            } catch (err) {
              console.error('Error in subscriber callback:', err);
            }
          }
          return;
        }

        if (msg.type === 'response' && msg.id) {
          const handler = this.pendingRpc.get(msg.id);
          if (handler) {
            this.pendingRpc.delete(msg.id);
            if (msg.error) {
              handler.reject(new KataGoEngineError(msg.error));
            } else {
              handler.resolve(msg.result);
            }
          }
        }
      };

      this.worker.onerror = (err: ErrorEvent) => {
        console.error('Worker error:', err);
      };
    }
    return this.worker;
  }

  private postRpc<T>(type: WorkerRpcRequest['type'], payload?: any): Promise<T> {
    const worker = this.getWorker();
    const id = `rpc-${++this.rpcSeq}`;
    return new Promise<T>((resolve, reject) => {
      this.pendingRpc.set(id, { resolve, reject });
      worker.postMessage({ id, type, payload } as WorkerRpcRequest);
    });
  }

  public async inspectCapabilities(): Promise<Capabilities> {
    if (this.cachedCapabilities) {
      return this.cachedCapabilities;
    }
    const caps = await this.postRpc<Capabilities>('inspect-capabilities');
    this.cachedCapabilities = caps;
    return caps;
  }

  public subscribe(listener: (event: EngineEvent) => void): () => void {
    if (this.isDisposed) {
      return () => {};
    }
    this.subscribers.add(listener);
    return () => {
      this.subscribers.delete(listener);
    };
  }

  public async initialize(
    request: Readonly<{
      operationId: string;
      model: ModelSource;
      runtime: RuntimePreference;
    }>,
    options?: Readonly<{ signal?: AbortSignal }>
  ): Promise<EngineInfo> {
    if (this.isDisposed) {
      failRequest('DISPOSED', 'Engine has been disposed');
    }
    validateIdentifier(request.operationId, 'operationId');

    if (options?.signal?.aborted) {
      failRequest('CANCELLED', 'Operation was aborted', null, 0);
    }

    if (options?.signal) {
      const abortHandler = () => {
        // Abort requested
      };
      options.signal.addEventListener('abort', abortHandler, { once: true });
    }

    return this.postRpc<EngineInfo>('initialize', request);
  }

  public async analyze(
    request: AnalyzeRequest,
    options?: Readonly<{ signal?: AbortSignal }>
  ): Promise<AnalysisResult> {
    if (this.isDisposed) {
      failRequest('DISPOSED', 'Engine has been disposed');
    }

    const caps = await this.inspectCapabilities();
    validateAnalyzeRequest(request, caps, this.activeIds);

    if (this.activeRequestId !== null) {
      failRequest('BUSY', 'An analysis is already in flight', request.requestId);
    }

    this.activeIds.add(request.requestId);
    this.activeRequestId = request.requestId;

    if (options?.signal?.aborted) {
      this.activeRequestId = null;
      failRequest('CANCELLED', 'Analyze operation was aborted', request.requestId);
    }

    let abortPromise: Promise<never> | null = null;
    if (options?.signal) {
      abortPromise = new Promise((_, reject) => {
        const onAbort = () => {
          this.cancel(request.requestId).finally(() => {
            reject(new KataGoEngineError({
              code: 'CANCELLED',
              message: 'Analyze operation was aborted by signal',
              requestId: request.requestId,
              engineGeneration: 0,
              details: {},
            }));
          });
        };
        options.signal?.addEventListener('abort', onAbort, { once: true });
        this.activeRequestSignalCleanup = () => {
          options.signal?.removeEventListener('abort', onAbort);
        };
      });
    }

    const analyzePromise = this.postRpc<AnalysisResult>('analyze', request);

    try {
      if (abortPromise) {
        return await Promise.race([analyzePromise, abortPromise]);
      }
      return await analyzePromise;
    } finally {
      if (this.activeRequestSignalCleanup) {
        this.activeRequestSignalCleanup();
        this.activeRequestSignalCleanup = null;
      }
      this.activeRequestId = null;
    }
  }

  public async cancel(requestId: string): Promise<void> {
    if (this.isDisposed) return;
    await this.postRpc<void>('cancel', { requestId });
    if (this.activeRequestId === requestId) {
      this.activeRequestId = null;
    }
  }

  public async reset(): Promise<void> {
    if (this.isDisposed) return;
    await this.postRpc<void>('reset');
    this.activeRequestId = null;
    this.activeIds.clear();
  }

  public async dispose(): Promise<void> {
    if (this.isDisposed) return;
    try {
      if (this.worker) {
        await this.postRpc<void>('dispose');
        this.worker.terminate();
        this.worker = null;
      }
    } finally {
      this.isDisposed = true;
      this.subscribers.clear();
      this.pendingRpc.clear();
      this.activeRequestId = null;
    }
  }
}

export function createEngine(options?: EngineOptions): KataGoEngine {
  return new KataGoEngineImpl(options);
}
