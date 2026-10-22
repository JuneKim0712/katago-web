import { createServer } from 'vite';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from '@playwright/test';

// Verify model files
const binGzPath = [
  process.env.KATAGO_MODEL,
  'C:/Users/junwo/Downloads/kata1-tf2-b10c384-s2941M-d5872M.bin.gz',
  resolve('models/kata1-tf2-b10c384-s2941M-d5872M.bin.gz'),
].find(p => p && existsSync(p));

if (!binGzPath) {
  console.error('ERROR: KataGo .bin.gz model not found.');
  process.exit(1);
}

const onnxPath = [
  process.env.KATAGO_ONNX,
  'C:/Users/junwo/.gemini/antigravity/scratch/kata1-b10c384.onnx',
  resolve('benchmarks/onnx-vs-tfjs/kata1-b10c384.onnx'),
].find(p => p && existsSync(p));

if (!onnxPath) {
  console.error('ERROR: Exported ONNX model not found. Run "katago dumponnx -model <bin.gz> -out <onnx>" first.');
  process.exit(1);
}

// Compute digests
const binGzBuffer = readFileSync(binGzPath);
const onnxBuffer = readFileSync(onnxPath);
const binGzSha256 = createHash('sha256').update(binGzBuffer).digest('hex');
const onnxSha256 = createHash('sha256').update(onnxBuffer).digest('hex');

console.log('=== KataGo TFJS vs ONNX Runtime WebGPU Benchmark ===');
console.log(`Source Model: ${binGzPath}`);
console.log(`  Size: ${binGzBuffer.length} bytes`);
console.log(`  SHA-256: ${binGzSha256}`);
console.log(`ONNX Graph: ${onnxPath}`);
console.log(`  Size: ${onnxBuffer.length} bytes`);
console.log(`  SHA-256: ${onnxSha256}`);

// Load canonical benchmark features
const featureFile = 'C:/Users/junwo/.gemini/antigravity/scratch/benchmark-features.json';
if (!existsSync(featureFile)) {
  console.error('ERROR: benchmark-features.json not found. Run generate-features first.');
  process.exit(1);
}
const featureData = JSON.parse(readFileSync(featureFile, 'utf8'));
console.log(`Position: ${featureData.caseId} (${featureData.moves.length} moves, rules: ${featureData.rules}, komi: ${featureData.komi})`);
console.log(`Feature SHA-256: ${featureData.combinedSha256}`);

// Start local Vite dev server
const vite = await createServer({
  server: { host: '127.0.0.1', port: 0 },
  optimizeDeps: {
    include: ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgpu', 'pako', 'onnxruntime-web/webgpu'],
  },
  plugins: [{
    name: 'serve-models',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === '/model.onnx') {
          res.setHeader('Content-Type', 'application/octet-stream');
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          res.end(onnxBuffer);
        } else if (req.url === '/model.bin.gz') {
          res.setHeader('Content-Type', 'application/gzip');
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          res.end(binGzBuffer);
        } else {
          next();
        }
      });
    }
  }]
});
await vite.listen();
const port = vite.httpServer.address().port;
const serverUrl = `http://127.0.0.1:${port}`;
console.log(`Harness server running at: ${serverUrl}`);

// Launch automated Chrome
const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu'],
});

const page = await browser.newPage();
const consoleLogs = [];
page.on('console', msg => consoleLogs.push(`[${msg.type()}] ${msg.text()}`));
page.on('pageerror', err => console.error('Browser page error:', err));

await page.goto(`${serverUrl}/benchmarks/onnx-vs-tfjs/harness.html`);
await page.waitForSelector('#status');

// 1. Untimed Hardware & Provider Diagnostics Pass
const diag = await page.evaluate(async () => {
  if (!navigator.gpu) return { error: 'navigator.gpu is undefined' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { error: 'requestAdapter returned null' };
  const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    features: Array.from(adapter.features),
    limits: {
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    }
  };
});
console.log('\nHardware Diagnostics:');
console.log(`  GPU Vendor: ${diag.vendor}, Architecture: ${diag.architecture}`);
console.log(`  Max Buffer Size: ${diag.limits.maxBufferSize} bytes`);
console.log(`  Subgroups Supported: ${diag.features.includes('subgroups')}`);

// 2. Numerical Parity Pass (Batch 1)
console.log('\nRunning Numerical Parity Check on shared input (Case: opening-white)...');
const parityRun = await page.evaluate(async (cfg) => {
  return await window.runBenchmark({
    batchSizes: [1],
    modelUrlTfjs: '/model.bin.gz',
    modelUrlOrt: '/model.onnx',
    warmupCount: 2,
    iterationCount: 1,
    featureData: cfg.featureData,
  });
}, { featureData });

const b1Blocks = parityRun.batchResults[1];
const tfjsOut = b1Blocks.find(b => b.runtime === 'TFJS').outputs;
const ortOut = b1Blocks.find(b => b.runtime === 'ORT').outputs;
const nativeExp = featureData.nativeExpected;

// Helper to compute numerical metrics
function computeMetrics(tfjs, ort, native) {
  // Value softmax
  const maxV_tf = Math.max(...tfjs.value);
  const expV_tf = tfjs.value.map(v => Math.exp(v - maxV_tf));
  const sumExp_tf = expV_tf.reduce((a, b) => a + b, 0);
  const probV_tf = expV_tf.map(v => v / sumExp_tf);

  const maxV_ort = Math.max(...ort.value);
  const expV_ort = ort.value.map(v => Math.exp(v - maxV_ort));
  const sumExp_ort = expV_ort.reduce((a, b) => a + b, 0);
  const probV_ort = expV_ort.map(v => v / sumExp_ort);

  // Policy: legal moves softmax
  const legalIndices = [];
  for (let i = 0; i < 361; i++) {
    if (native.policy[i] !== null) legalIndices.push(i);
  }
  legalIndices.push(361); // pass

  const polTfLogits = legalIndices.map(idx => idx === 361 ? tfjs.policyPass[0] : tfjs.policy[idx * 2]);
  const maxPolTf = Math.max(...polTfLogits);
  const expPolTf = polTfLogits.map(l => Math.exp(l - maxPolTf));
  const probPolTf = expPolTf.map(e => e / expPolTf.reduce((a, b) => a + b, 0));

  const polOrtLogits = legalIndices.map(idx => idx === 361 ? ort.policyPass[0] : ort.policy[idx * 2]);
  const maxPolOrt = Math.max(...polOrtLogits);
  const expPolOrt = polOrtLogits.map(l => Math.exp(l - maxPolOrt));
  const probPolOrt = expPolOrt.map(e => e / expPolOrt.reduce((a, b) => a + b, 0));

  // Top 5 ranking
  const top5Tf = legalIndices.slice(0, -1)
    .map(idx => ({ idx, p: tfjs.policy[idx * 2] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5)
    .map(m => m.idx);

  const top5Ort = legalIndices.slice(0, -1)
    .map(idx => ({ idx, p: ort.policy[idx * 2] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5)
    .map(m => m.idx);

  const top5Native = native.policy
    .map((p, idx) => ({ idx, p: p ?? -999 }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5)
    .map(m => m.idx);

  // Score lead & mean (with multiplier 20.0 and noResult factor)
  const leadTf = tfjs.scoreValue[2] * 20.0 * (1.0 - probV_tf[2]);
  const leadOrt = ort.scoreValue[2] * 20.0 * (1.0 - probV_ort[2]);
  const scoreMeanTf = tfjs.scoreValue[0] * 20.0 * (1.0 - probV_tf[2]);
  const scoreMeanOrt = ort.scoreValue[0] * 20.0 * (1.0 - probV_ort[2]);

  // Ownership (post-tanh)
  let maxOwnDiff = 0, sumOwnDiff = 0;
  for (let i = 0; i < 361; i++) {
    const ownTf = Math.tanh(tfjs.ownership[i]);
    const ownOrt = Math.tanh(ort.ownership[i]);
    const diff = Math.abs(ownTf - ownOrt);
    if (diff > maxOwnDiff) maxOwnDiff = diff;
    sumOwnDiff += diff;
  }

  // Policy diff between TFJS and ORT
  let maxPolDiff = 0, sumPolDiff = 0;
  for (let k = 0; k < legalIndices.length; k++) {
    const diff = Math.abs(probPolTf[k] - probPolOrt[k]);
    if (diff > maxPolDiff) maxPolDiff = diff;
    sumPolDiff += diff;
  }

  // Value diff
  const winRateDiff = Math.abs(probV_tf[0] - probV_ort[0]);
  const scoreLeadDiff = Math.abs(leadTf - leadOrt);
  const scoreMeanDiff = Math.abs(scoreMeanTf - scoreMeanOrt);

  return {
    winRateDiff,
    scoreLeadDiff,
    scoreMeanDiff,
    maxPolicyProbDiff: maxPolDiff,
    meanPolicyProbDiff: sumPolDiff / legalIndices.length,
    maxOwnershipDiff: maxOwnDiff,
    meanOwnershipDiff: sumOwnDiff / 361,
    top5Tf,
    top5Ort,
    top5Native,
    top5Agreement: top5Tf.every((v, idx) => v === top5Ort[idx]),
    tfjsWinProb: probV_tf[0],
    ortWinProb: probV_ort[0],
    nativeWinProb: native.whiteWin,
    tfjsLead: leadTf,
    ortLead: leadOrt,
    nativeLead: native.whiteLead,
  };
}

const parityMetrics = computeMetrics(tfjsOut, ortOut, nativeExp);
console.log('Numerical Check Results (TFJS WebGPU vs ORT WebGPU vs Native):');
console.log(`  Top-5 Legal Moves Agreement: ${parityMetrics.top5Agreement ? '100% MATCH' : 'MISMATCH'}`);
console.log(`    Native: [${parityMetrics.top5Native.join(', ')}]`);
console.log(`    TFJS:   [${parityMetrics.top5Tf.join(', ')}]`);
console.log(`    ORT:    [${parityMetrics.top5Ort.join(', ')}]`);
console.log(`  Winrate: TFJS=${parityMetrics.tfjsWinProb.toFixed(5)}, ORT=${parityMetrics.ortWinProb.toFixed(5)}, Native=${parityMetrics.nativeWinProb.toFixed(5)}`);
console.log(`    TFJS vs ORT diff: ${parityMetrics.winRateDiff.toExponential(4)}`);
console.log(`  Score Lead: TFJS=${parityMetrics.tfjsLead.toFixed(4)}, ORT=${parityMetrics.ortLead.toFixed(4)}, Native=${parityMetrics.nativeLead.toFixed(4)}`);
console.log(`    TFJS vs ORT diff: ${parityMetrics.scoreLeadDiff.toExponential(4)}`);
console.log(`  Policy Prob Max Error: ${parityMetrics.maxPolicyProbDiff.toExponential(4)}, Mean Error: ${parityMetrics.meanPolicyProbDiff.toExponential(4)}`);
console.log(`  Ownership Max Error:   ${parityMetrics.maxOwnershipDiff.toExponential(4)}, Mean Error:   ${parityMetrics.meanOwnershipDiff.toExponential(4)}`);

// 3. Timed Benchmark Execution
// Batch sizes: batch 1 and batch 8 (probed production batch size), plus batch 4 provisional
const batchSizes = [1, 4, 8];
console.log(`\nExecuting Benchmark across batch sizes: [${batchSizes.join(', ')}]`);
console.log('Protocol: For each batch, Order = TFJS -> ORT -> ORT -> TFJS');
console.log('Per block: 10 untimed warmups, 50 sequential timed iterations\n');

const benchmarkResults = await page.evaluate(async (cfg) => {
  return await window.runBenchmark({
    batchSizes: cfg.batchSizes,
    modelUrlTfjs: '/model.bin.gz',
    modelUrlOrt: '/model.onnx',
    warmupCount: 10,
    iterationCount: 50,
    featureData: cfg.featureData,
  });
}, { batchSizes, featureData });

// Format and pool results
const summary = {
  hardware: {
    gpu: `${diag.vendor} ${diag.architecture}`,
    adapterLimits: diag.limits,
    os: process.platform,
    browser: 'Chrome 155.0.8059.86',
  },
  model: {
    source: binGzPath,
    sourceSha256: binGzSha256,
    onnx: onnxPath,
    onnxSha256: onnxSha256,
  },
  numericalAgreement: parityMetrics,
  batches: {},
};

for (const batch of batchSizes) {
  const blocks = benchmarkResults.batchResults[batch];
  const tfjsBlocks = blocks.filter(b => b.runtime === 'TFJS');
  const ortBlocks = blocks.filter(b => b.runtime === 'ORT');

  // Pooled timings
  const tfjsTimings = tfjsBlocks.flatMap(b => b.timings).sort((a, b) => a - b);
  const ortTimings = ortBlocks.flatMap(b => b.timings).sort((a, b) => a - b);

  const tfjsMedian = tfjsTimings[Math.floor(tfjsTimings.length / 2)];
  const tfjsP95 = tfjsTimings[Math.floor(tfjsTimings.length * 0.95)];
  const tfjsSum = tfjsTimings.reduce((a, b) => a + b, 0);
  const tfjsObservedPosPerSec = (batch * tfjsTimings.length * 1000) / tfjsSum;
  const tfjsMedianPosPerSec = (batch * 1000) / tfjsMedian;

  const ortMedian = ortTimings[Math.floor(ortTimings.length / 2)];
  const ortP95 = ortTimings[Math.floor(ortTimings.length * 0.95)];
  const ortSum = ortTimings.reduce((a, b) => a + b, 0);
  const ortObservedPosPerSec = (batch * ortTimings.length * 1000) / ortSum;
  const ortMedianPosPerSec = (batch * 1000) / ortMedian;

  const speedupFactor = tfjsMedian / ortMedian;
  const throughputIncreasePct = (speedupFactor - 1) * 100;
  const latencyReductionPct = (1 - (ortMedian / tfjsMedian)) * 100;

  summary.batches[batch] = {
    batchSize: batch,
    blocks: blocks.map(b => ({
      runtime: b.runtime,
      blockIndex: b.blockIndex,
      loadMs: b.loadMs,
      warmupMs: b.warmupMs,
      medianBatchMs: b.medianBatchMs,
      p95BatchMs: b.p95BatchMs,
      observedPosPerSec: b.observedPosPerSec,
      medianPosPerSec: b.medianPosPerSec,
    })),
    pooled: {
      tfjs: {
        medianBatchMs: tfjsMedian,
        p95BatchMs: tfjsP95,
        observedPosPerSec: tfjsObservedPosPerSec,
        medianPosPerSec: tfjsMedianPosPerSec,
      },
      ort: {
        medianBatchMs: ortMedian,
        p95BatchMs: ortP95,
        observedPosPerSec: ortObservedPosPerSec,
        medianPosPerSec: ortMedianPosPerSec,
      },
      comparison: {
        fasterRuntime: speedupFactor >= 1.0 ? 'ORT' : 'TFJS',
        speedupFactor,
        throughputIncreasePercent: throughputIncreasePct,
        latencyReductionPercent: latencyReductionPct,
        conclusive: Math.abs(speedupFactor - 1.0) >= 0.05,
      }
    }
  };
}

// Write machine-readable results
const outJsonPath = resolve('benchmarks/onnx-vs-tfjs/results.json');
writeFileSync(outJsonPath, JSON.stringify(summary, null, 2));
console.log(`\nMachine-readable benchmark results saved to: ${outJsonPath}`);

await browser.close();
await vite.close();
