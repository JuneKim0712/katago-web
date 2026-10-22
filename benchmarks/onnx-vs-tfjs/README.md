# KataGo TFJS vs ONNX Runtime WebGPU Benchmark

This benchmark performs an isolated, fair comparison of neural inference throughput and latency between **TensorFlow.js WebGPU** and **ONNX Runtime WebGPU** (`onnxruntime-web/webgpu`) using the exact same KataGo model (`kata1-b10c384`) and legal board position on Google Chrome.

---

## 1. Executive Summary & Verdict

- **Which runtime is faster?**
  **TensorFlow.js (TFJS) WebGPU is decisively faster than ONNX Runtime (ORT) WebGPU across all evaluated batch sizes.**
  - **Batch 1**: TFJS is **3.76x faster** (Median latency: **24.90 ms** vs **93.60 ms**; observed throughput: **39.13 pos/s** vs **10.64 pos/s**).
  - **Batch 4 (provisional)**: TFJS is **3.51x faster** (Median latency: **84.20 ms** vs **295.30 ms**; observed throughput: **49.28 pos/s** vs **13.73 pos/s**).
  - **Batch 8 (actual production batch size)**: TFJS is **3.02x faster** (Median latency: **132.50 ms** vs **399.50 ms**; observed throughput: **60.05 pos/s** vs **20.01 pos/s**).
- **Numerical Parity**: 100% agreement on legal policy top-5 moves (`[320, 306, 263, 78, 62]`), winrate difference $\le 1.82 \times 10^{-7}$, ownership difference $\le 1.16 \times 10^{-6}$.
- **WebGPU EP Coverage**: 100% pure WebGPU execution on both runtimes; 0 neural operators fell back to CPU / WASM.
- **Important Domain Note**: Neural inference throughput measures pure forward-pass evaluation rate and does **not** directly translate to MCTS visits/second or playing strength.

---

## 2. Environment & Hardware

- **OS**: Windows 11 Home (win32, x64)
- **Browser**: Google Chrome 155.0.8059.86 (Official Build)
- **GPU Adapter**: Intel Arc 140V GPU (`xe-2lpg`, 16 GB unified memory, max buffer size 2,147,483,648 bytes, `subgroups` supported)
- **Runtimes**:
  - Production TFJS: `@tensorflow/tfjs` & `@tensorflow/tfjs-backend-webgpu` `4.22.0` (with custom kernels `fuseRotary: true`, `fuseRmsNorm: true`)
  - Benchmark ORT: `onnxruntime-web` `1.22.0` (`onnxruntime-web/webgpu`)
- **Source Model**:
  - `kata1-tf2-b10c384-s2941M-d5872M.bin.gz`
  - SHA-256: `8015d49bdf45854c94e5f9e22b2b70e8738d3e56a6edca730e5b82509998a5b5`
  - Size: 38,245,507 bytes
- **Exported ONNX Model**:
  - `kata1-b10c384.onnx`
  - SHA-256: `c306a252a2909859c52d4d2ab433ac89ba051f8f2606850df2f747203e5974e3`
  - Size: 64,605,132 bytes

---

## 3. Model Export

The ONNX model was exported using the official KataGo v1.18.1 DirectML/ONNX release:

```powershell
& "C:\Users\junwo\.gemini\antigravity\scratch\katago-onnx\katago.exe" dumponnx `
  -model "C:\Users\junwo\Downloads\kata1-tf2-b10c384-s2941M-d5872M.bin.gz" `
  -out "C:\Users\junwo\.gemini\antigravity\scratch\kata1-b10c384.onnx"
```

Export settings:
- 19x19 FP32 board dimensions
- `requireExactNNLen = false` (dynamic batch size)
- `transformerNHWC = true`
- Spatial input: `[batch, 22, 19, 19]` (NCHW)
- Global input: `[batch, 19]`

---

## 4. Execution Provider & Numerical Verification

### 100% WebGPU Placement
- TFJS selected `webgpu` backend with production kernel fusions.
- ORT initialized with `executionProviders: ['webgpu']`. Provider diagnostics confirmed 100% of nodes allocated to WebGPU EP with 0 fallback nodes.
- **WGSL Vectorization Patch**: Addressed an ORT WebGPU WGSL shader compilation bug in RoPE / RMSNorm squaring nodes where `pow_v(vec4<f32>, f32)` mismatched scalar and vector types by intercepting `GPUDevice.prototype.createShaderModule` to vectorize the scalar exponent.

### Numerical Agreement on Non-Empty Board
Evaluated on position `opening-white` (5 legal moves played, Japanese rules, komi 6.5, White to move):
- **Top-5 Legal Moves**:
  - Native KataGo v1.18.1: `[320, 306, 263, 78, 62]`
  - TFJS WebGPU: `[320, 306, 263, 78, 62]`
  - ORT WebGPU: `[320, 306, 263, 78, 62]`
  - **Agreement**: **100% MATCH**
- **Winrate (White)**:
  - TFJS: `0.524550`
  - ORT: `0.524550`
  - Native: `0.524553`
  - TFJS vs ORT diff: `1.8169e-7`
- **Score Lead**:
  - TFJS: `0.14243` pts
  - ORT: `0.14244` pts
  - Native: `0.14200` pts
  - TFJS vs ORT diff: `3.0030e-6`
- **Policy Probability Max Error**: `3.8134e-7` (Mean: `3.7109e-9`)
- **Ownership Max Error (post-tanh)**: `1.1611e-6` (Mean: `2.1064e-7`)

---

## 5. Benchmark Protocol & Results

- **Run Order**: `TFJS -> ORT -> ORT -> TFJS` for each batch size.
- **Isolation**: Each block runs in an independent Web Worker with isolated model memory, doing 10 untimed warmups + 50 sequential timed iterations, terminated before the next block starts.
- **Timing Interval**: Strictly measures CPU feature array upload $\rightarrow$ WebGPU execution $\rightarrow$ CPU output readback resolution (`Promise.all` on `.data()`).

### Detailed Results Table

| Batch Size | Runtime | Block | Median (ms) | p95 (ms) | Observed Pos/s | Median-derived Pos/s |
|---|---|---|---|---|---|---|
| **Batch 1** | **TFJS** | 1 | 25.70 | 28.20 | 39.04 | 38.91 |
| | **ORT** | 1 | 102.10 | 117.50 | 9.91 | 9.79 |
| | **ORT** | 2 | 88.80 | 100.30 | 11.50 | 11.26 |
| | **TFJS** | 2 | 24.20 | 40.40 | 39.22 | 41.32 |
| | **Pooled TFJS** | - | **24.90** | **28.50** | **39.13** | **40.16** |
| | **Pooled ORT** | - | **93.60** | **116.30** | **10.64** | **10.68** |
| **Batch 4** | **TFJS** | 1 | 84.80 | 99.20 | 48.25 | 47.17 |
| | **ORT** | 1 | 258.60 | 284.30 | 15.39 | 15.47 |
| | **ORT** | 2 | 321.90 | 378.10 | 12.39 | 12.43 |
| | **TFJS** | 2 | 78.70 | 91.30 | 50.35 | 50.83 |
| | **Pooled TFJS** | - | **84.20** | **97.10** | **49.28** | **47.51** |
| | **Pooled ORT** | - | **295.30** | **339.60** | **13.73** | **13.55** |
| **Batch 8 (Production)** | **TFJS** | 1 | 131.30 | 145.20 | 60.55 | 60.93 |
| | **ORT** | 1 | 386.20 | 439.40 | 20.50 | 20.71 |
| | **ORT** | 2 | 409.70 | 438.40 | 19.55 | 19.53 |
| | **TFJS** | 2 | 133.00 | 150.50 | 59.55 | 60.15 |
| | **Pooled TFJS** | - | **132.50** | **146.50** | **60.05** | **60.38** |
| | **Pooled ORT** | - | **399.50** | **439.40** | **20.01** | **20.03** |

---

## 6. Metric Formulas & Relative Comparisons

Using the prompt's standard comparison formulas:
- $\text{Speedup Factor} = \frac{\text{TFJS Median Batch ms}}{\text{ORT Median Batch ms}}$
- $\text{Throughput Increase \%} = (\text{Speedup Factor} - 1) \times 100$
- $\text{Latency Reduction \%} = \left(1 - \frac{\text{ORT Median Batch ms}}{\text{TFJS Median Batch ms}}\right) \times 100$

| Metric | Batch 1 | Batch 4 | Batch 8 (Production) |
|---|---|---|---|
| **Faster Runtime** | **TFJS** | **TFJS** | **TFJS** |
| **TFJS Speedup vs ORT** | **3.76x faster** | **3.51x faster** | **3.02x faster** |
| **Speedup Factor (TFJS / ORT)** | 0.2660 | 0.2851 | 0.3317 |
| **Throughput Increase %** | -73.40% | -71.49% | -66.83% |
| **Latency Reduction %** | -275.90% | -250.71% | -201.51% |
| **TFJS Latency Advantage** | **73.4% lower latency** | **71.5% lower latency** | **66.8% lower latency** |
| **TFJS Throughput Advantage** | **+268% throughput** | **+259% throughput** | **+200% throughput** |
| **Statistical Conclusiveness** | Conclusive ($> 5\%$) | Conclusive ($> 5\%$) | Conclusive ($> 5\%$) |

---

## 7. How to Reproduce

Execute the single command runner:

```powershell
node benchmarks/onnx-vs-tfjs/runner.mjs
```

This will launch headless Chrome, serve the test harness and models via Vite, run diagnostics and parity verification, execute the 12 benchmark blocks, and output `benchmarks/onnx-vs-tfjs/results.json`.
